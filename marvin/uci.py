"""UCI front end. Run: python -m marvin.uci"""

from __future__ import annotations

import sys
import threading
import time

import chess
import numpy as np

from marvin.engine import FLAG_MODES, THINK_MODES, Settings, centipawns, decide, explain, runs_out_of_time, typical_time
from marvin.model import ELO_MAX, ELO_MIN, Context, Model

STANDARD_BASES = [15, 30, 60, 120, 180, 300, 600, 900, 1200, 1800, 2700, 3600, 5400, 7200]

OPTIONS = [  # name, type, default, bounds/vars
    ("UCI_Elo", "spin", "1500", f"min {ELO_MIN} max {ELO_MAX}"),
    ("OpponentElo", "spin", "0", f"min 0 max {ELO_MAX}"),
    ("UCI_Opponent", "string", "", ""),
    ("Model", "combo", "79m", "var 79m var 23m"),
    ("Temperature", "string", "1.0", ""),
    ("TopP", "string", "1.0", ""),
    ("ThinkTime", "combo", "sample", " ".join(f"var {m}" for m in THINK_MODES)),
    ("TimeTemperature", "string", "1.0", ""),
    ("TimeTopP", "string", "1.0", ""),
    ("TimeScale", "string", "1.0", ""),
    ("Flag", "combo", "sample", " ".join(f"var {m}" for m in FLAG_MODES)),
    ("BaseTime", "spin", "0", "min 0 max 36000"),
    ("FallbackTimeControl", "string", "5+0", ""),
    ("MoveOverhead", "spin", "100", "min 0 max 10000"),
]


def say(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Engine:
    def __init__(self) -> None:
        self.opts = {name.lower(): default for name, _, default, _ in OPTIONS}
        self.model: Model | None = None
        self.rng = np.random.default_rng()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.debug = False
        self.new_game()

    # --- options -------------------------------------------------------------------------

    def opt(self, name: str) -> float:
        """Numeric option value; an unparsable value falls back to the default."""
        try:
            return float(self.opts[name.lower()])
        except ValueError:
            return float(next(d for n, _, d, _ in OPTIONS if n.lower() == name.lower()))

    def settings(self) -> Settings:
        mode = self.opts["thinktime"]
        return Settings(
            temperature=self.opt("Temperature"), top_p=self.opt("TopP"),
            think_time=mode if mode in THINK_MODES else "sample",
            time_temperature=self.opt("TimeTemperature"), time_top_p=self.opt("TimeTopP"),
            time_scale=max(0.0, self.opt("TimeScale")),
        )

    def elos(self) -> tuple[int, int]:
        clamp = lambda e: max(ELO_MIN, min(ELO_MAX, int(e)))
        bot = clamp(self.opt("UCI_Elo"))
        opp = int(self.opt("OpponentElo"))
        if opp == 0:  # auto: rating from UCI_Opponent ("<title> <elo> <computer|human> <name>"), else mirror
            parts = self.opts["uci_opponent"].split()
            opp = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else bot
        return bot, clamp(opp)

    def fallback_tc(self) -> tuple[float, float]:
        try:
            minutes, inc = self.opts["fallbacktimecontrol"].split("+")
            return float(minutes) * 60, float(inc)
        except ValueError:
            return 300.0, 0.0

    def setoption(self, line: str) -> None:
        rest = line.split(" name ", 1)[1] if " name " in line else ""
        name, _, value = rest.partition(" value ")
        key = name.strip().lower()
        if key in self.opts:
            self.opts[key] = value.strip()
            if key == "model":
                self.model = None

    def load_model(self) -> None:
        if self.model is None:
            try:
                self.model = Model(self.opts["model"])
            except Exception as e:
                say(f"info string {e}")

    # --- game state ----------------------------------------------------------------------

    def new_game(self) -> None:
        self.fen = chess.STARTING_FEN
        self.moves: list[str] = []
        self.times: list[float | None] = []  # seconds spent per ply; None = unknown
        self.last_go: tuple[int, float] | None = None  # (ply, opponent clock) at our previous go
        self.pending: tuple[int, str, float] | None = None  # our last move and its time
        self.base: float | None = None

    def set_position(self, line: str) -> None:
        tokens = line.split()
        fen = chess.STARTING_FEN
        if "fen" in tokens:
            end = tokens.index("moves") if "moves" in tokens else len(tokens)
            fen = " ".join(tokens[tokens.index("fen") + 1:end])
        moves = tokens[tokens.index("moves") + 1:] if "moves" in tokens else []

        if fen == self.fen and moves[: len(self.moves)] == self.moves:
            self.times += [None] * (len(moves) - len(self.moves))
            if self.pending and self.pending[0] < len(moves) and moves[self.pending[0]] == self.pending[1]:
                self.times[self.pending[0]] = self.pending[2]
        elif fen == self.fen and self.moves[: len(moves)] == moves:  # takeback
            self.times = self.times[: len(moves)]
        else:
            self.times = [None] * len(moves)
            self.last_go = None
        self.fen, self.moves, self.pending = fen, moves, None

    def infer_base(self, max_clock: float, ply: int) -> float:
        if self.opt("BaseTime") > 0:
            return self.opt("BaseTime")
        if self.base is None:
            # Before either side has moved, the larger clock is the base time.
            self.base = float(round(max_clock)) if ply <= 1 else float(
                next((b for b in STANDARD_BASES if b >= max_clock - 0.5), round(max_clock)))
        return self.base

    def fill_times(self, clocked: bool, real: dict, base: float, incs: dict, elos: dict) -> dict:
        """Estimate unknown move times with the model; return simulated clocks for the current ply.

        With a real clock only the last 8 plies matter (the model's time history); without one,
        every ply is needed to simulate both clocks from the fallback time control.
        """
        board = chess.Board(self.fen)
        sim = {chess.WHITE: base, chess.BLACK: base}
        first = max(0, len(self.moves) - 8) if clocked else 0
        for i, uci in enumerate(self.moves):
            mover = board.turn
            if i < 2:
                self.times[i] = incs[mover]  # Lichess records the increment for the first two plies
            elif i >= first and self.times[i] is None:
                clocks = real if clocked else sim
                ctx = Context(elos[mover], elos[not mover], clocks[mover], clocks[not mover],
                              incs[mover], incs[not mover], base,
                              [t or 0.0 for t in reversed(self.times[max(0, i - 8):i])])
                pred = self.model.predict(board, ctx)
                self.times[i] = typical_time(pred.time_logits[pred.moves.index(chess.Move.from_uci(uci))])
            if i >= 2:  # as on Lichess, the first move of each side doesn't touch the clock
                sim[mover] = max(0.0, sim[mover] - (self.times[i] or 0.0) + incs[mover])
            board.push_uci(uci)
        return sim

    # --- search --------------------------------------------------------------------------

    def go(self, line: str) -> None:
        started = time.monotonic()  # think time includes the time spent computing
        tokens = line.split()[1:]
        p = {k: float(v) for k, v in zip(tokens, tokens[1:]) if k in
             ("wtime", "btime", "winc", "binc", "movetime", "nodes", "depth", "movestogo")}
        infinite = "infinite" in tokens or "ponder" in tokens
        self.load_model()
        board = chess.Board(self.fen)
        for m in self.moves:
            board.push_uci(m)
        if self.model is None or board.is_game_over():
            say("bestmove 0000")
            return

        me, ply, s = board.turn, len(self.moves), self.settings()
        bot_elo, opp_elo = self.elos()
        elos = {me: bot_elo, not me: opp_elo}
        clocked = "wtime" in p and "btime" in p
        if clocked:
            real = {chess.WHITE: p["wtime"] / 1000, chess.BLACK: p["btime"] / 1000}
            incs = {chess.WHITE: p.get("winc", 0) / 1000, chess.BLACK: p.get("binc", 0) / 1000}
            base = self.infer_base(max(real.values()), ply)
            if self.last_go and self.last_go[0] == ply - 2 and self.times[ply - 1] is None:
                self.times[ply - 1] = max(0.0, self.last_go[1] + incs[not me] - real[not me])
            self.last_go = (ply, real[not me])
            self.fill_times(True, real, base, incs, elos)
            clocks = real
        else:
            base, inc = self.fallback_tc()
            incs = {chess.WHITE: inc, chess.BLACK: inc}
            clocks = self.fill_times(False, {}, base, incs, elos)

        ctx = Context(bot_elo, opp_elo, clocks[me], clocks[not me], incs[me], incs[not me], base,
                      [t or 0.0 for t in reversed(self.times[-8:])])
        overhead = self.opt("MoveOverhead") / 1000
        limit = float("inf")
        if clocked:
            limit = max(0.0, clocks[me] - overhead)
        elif "movetime" in p:
            limit = max(0.0, p["movetime"] / 1000 - overhead)
        d = decide(self.model, board, ctx, s, self.rng, limit)
        wait = 0.0 if s.think_time == "instant" or limit == float("inf") else d.think
        # Running out of time: hold the move until the clock has expired, so the GUI scores a loss
        # on time (a `stop` still gets an immediate bestmove, as the protocol requires).
        flag = clocked and ply >= 2 and runs_out_of_time(d.prediction, clocks[me], self.opts["flag"], self.rng)
        if flag:
            wait = clocks[me] + 1.0
        self.pending = (ply, d.move.uci(), d.think)

        say(f"info depth 1 nodes 1 score cp {centipawns(d.prediction)} pv {d.move.uci()}")
        say(f"info string p {d.prob:.3f} think {d.think:.2f}s flag {d.prediction.flag:.3f}"
            + (" (letting the clock run out)" if flag else ""))
        if self.debug:
            for text in explain(board, ctx, d, s):
                say(f"info string {text}")
        self.stop.clear()
        self.worker = threading.Thread(target=self._finish, args=(d.move.uci(), started + wait, infinite), daemon=True)
        self.worker.start()

    def _finish(self, move: str, until: float, infinite: bool) -> None:
        if infinite:
            self.stop.wait()
        else:
            self.stop.wait(max(0.0, until - time.monotonic()))
        say(f"bestmove {move}")

    def halt(self) -> None:
        self.stop.set()
        if self.worker:
            self.worker.join()
            self.worker = None

    # --- protocol ------------------------------------------------------------------------

    def loop(self) -> None:
        for raw in sys.stdin:
            line = raw.strip()
            cmd = line.split(" ", 1)[0]
            if cmd == "uci":
                say("id name Marvin")
                say("id author Victor Dziwniel")
                for name, kind, default, extra in OPTIONS:
                    shown = default or "<empty>"
                    say(f"option name {name} type {kind} default {shown} {extra}".rstrip())
                say("uciok")
            elif cmd == "debug":
                self.debug = line.endswith(" on")
            elif cmd == "isready":
                self.load_model()
                say("readyok")
            elif cmd == "setoption":
                self.setoption(line)
            elif cmd == "ucinewgame":
                self.halt()
                self.new_game()
            elif cmd == "position":
                self.halt()
                self.set_position(line)
            elif cmd == "go":
                self.halt()
                self.go(line)
            elif cmd == "stop":
                self.halt()
            elif cmd == "quit":
                break
        self.halt()


if __name__ == "__main__":
    Engine().loop()
