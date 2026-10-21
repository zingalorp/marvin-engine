"""Minimal browser GUI. Run: python -m marvin.gui [--model 79m] [--port 8000] [--host H] [--no-browser]"""

from __future__ import annotations

import argparse
import json
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import chess
import numpy as np

from marvin.engine import THINK_MODES, Settings, analyse, decide, explain, outcome, runs_out_of_time, time_stats
from marvin.model import Context, Model

HERE = Path(__file__).resolve().parent
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".mp3": "audio/mpeg"}
# Lichess's standard sound set. Its license is unclear, so it isn't bundled: each file is fetched from
# Lichess on first use and kept in static/sound.
SOUNDS = {"Move", "Capture", "Check", "Checkmate", "LowTime", "GenericNotify", "Victory", "Defeat", "Draw"}
SOUND_URL = "https://lichess1.org/assets/sound/standard/{}.mp3"
RESIGN_THRESHOLD = 0.9  # humans lost 98% of test-set positions where the head fires
RESIGN_MIN_PLY = 20
FLAG_MODE = "sample"  # see marvin.engine.FLAG_MODES
CLOCK_MARGIN = 0.3  # seconds kept back for request latency


def replay(moves: list[str]) -> tuple[chess.Board, list[str]]:
    board, san = chess.Board(), []
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {uci}")
        san.append(board.san(move))
        board.push(move)
    return board, san


def state(req: dict) -> dict:
    board, san = replay(req["moves"])
    end = outcome(board)
    return {
        "fen": board.fen(),
        "legal": [m.uci() for m in board.legal_moves],
        "san": san,
        "check": board.is_check(),
        "over": end and {"result": end.result(), "reason": end.termination.name.replace("_", " ").capitalize()},
        "material": {"w": not board.has_insufficient_material(chess.WHITE),
                     "b": not board.has_insufficient_material(chess.BLACK)},
    }


def settings(req: dict) -> Settings:
    s = req.get("settings", {})
    num = lambda key, lo, hi, default: min(hi, max(lo, float(s.get(key, default))))
    return Settings(
        temperature=num("temperature", 0, 5, 1), top_p=num("top_p", 0.01, 1, 1),
        think_time=s.get("think_time") if s.get("think_time") in THINK_MODES else "sample",
        time_temperature=num("time_temperature", 0, 5, 1), time_top_p=num("time_top_p", 0.01, 1, 1),
        time_scale=num("time_scale", 0, 10, 1),
    )


def context(req: dict, board: chess.Board) -> Context:
    """Context for the side to move. Clocks and ratings arrive keyed by colour ("w"/"b")."""
    me, them = ("w", "b") if board.turn == chess.WHITE else ("b", "w")
    return Context(req["elo"][me], req["elo"][them], req["clock"][me], req["clock"][them],
                   req["inc"], req["inc"], req["base"], req["times"][::-1])


def time_summary(time_logits: np.ndarray, offset=None) -> dict:
    """Think-time histogram for the popup: the fixed log-scale bars of marvin.engine.BAR_EDGES."""
    ts = time_stats(time_logits, offset)
    return {"bins": [round(float(x), 5) for x in ts.bins], "mode": round(ts.mode, 2),
            "typical": round(ts.typical, 2), "mean": round(ts.mean, 2)}


def fetch_sound(path: Path) -> None:
    if path.is_file():
        return
    try:
        with urllib.request.urlopen(SOUND_URL.format(path.stem), timeout=10) as r:
            data = r.read()
    except OSError:
        return  # offline: the page plays without this sound
    path.parent.mkdir(exist_ok=True)
    part = path.with_suffix(".part")
    part.write_bytes(data)
    part.replace(path)


class Handler(BaseHTTPRequestHandler):
    model: Model
    rng = np.random.default_rng()

    def log_message(self, *args) -> None:
        pass

    def reply(self, body: bytes, kind: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        name = "gui.html" if self.path == "/" else self.path.lstrip("/")
        path = (HERE / name).resolve()
        if path.parent == HERE / "static" / "sound" and path.stem in SOUNDS and path.suffix == ".mp3":
            fetch_sound(path)
        if path.parent not in (HERE, HERE / "static", HERE / "static" / "sound") or path.suffix not in TYPES \
                or not path.is_file():
            self.send_error(404)
            return
        self.reply(path.read_bytes(), TYPES[path.suffix])

    def do_POST(self) -> None:
        handlers = {"/state": state, "/move": self.move, "/analyse": self.analyse}
        if self.path not in handlers:
            self.send_error(404)
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            out = handlers[self.path](req)
        except Exception as e:  # report to the page instead of dropping the connection
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode())
            return
        self.reply(json.dumps(out).encode(), "application/json")

    def move(self, req: dict) -> dict:
        board, _ = replay(req["moves"])
        ctx, s = context(req, board), settings(req)
        limit = max(0.0, ctx.clock - CLOCK_MARGIN) if len(req["moves"]) >= 2 else float("inf")
        d = decide(self.model, board, ctx, s, self.rng, limit)
        resign = len(req["moves"]) >= RESIGN_MIN_PLY and d.prediction.resign >= RESIGN_THRESHOLD
        flag = len(req["moves"]) >= 2 and runs_out_of_time(d.prediction, ctx.clock, FLAG_MODE, self.rng)
        wait = 0.0 if s.think_time == "instant" else d.think
        return {"move": d.move.uci(), "think": d.think, "wait": wait, "resign": resign, "flag": flag,
                "debug": "\n".join(explain(board, ctx, d, s))}

    def analyse(self, req: dict) -> dict:
        board, _ = replay(req["moves"])
        pred, rows = analyse(self.model, board, context(req, board), float(req.get("top_p", 1.0)))
        white = board.turn == chess.WHITE
        as_white = lambda w, d, l: {"white": w if white else l, "draw": d, "black": l if white else w}
        moves = [{"uci": r.move.uci(), "san": board.san(r.move), "prob": r.prob,
                  "wdl": as_white(*r.wdl), "time": time_summary(r.time_logits, r.offset)} for r in rows]
        return {"moves": moves, "legal": len(pred.moves), "wdl": as_white(pred.win, pred.draw, pred.loss),
                "resign": pred.resign, "flag": pred.flag}


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Marvin in the browser.")
    ap.add_argument("--model", default="79m", help="79m, 23m, or a path to an .onnx file")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="localhost",
                    help="address to listen on; 0.0.0.0 makes the GUI reachable from other devices on your network")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser (e.g. when running as a service)")
    args = ap.parse_args()

    Handler.model = Model(args.model)
    url = f"http://localhost:{args.port}"
    print(f"Marvin ({Handler.model.name}) at {url}" + (f" (listening on {args.host})" if args.host != "localhost" else ""),
          flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
