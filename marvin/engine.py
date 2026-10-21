"""Move and think-time selection on top of the model's predictions."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from marvin.model import TIME_BINS, Context, Model, Prediction, encode
from marvin.offset import CENTRES

THINK_MODES = ("sample", "typical", "mean", "mode", "instant")
# How Marvin runs out of time instead of moving. "sample": with the flag head's probability, but
# only with at most FLAG_MAX_CLOCK seconds left, so it doesn't abandon games with time to spare.
# "threshold": whenever the head's training-time output (flag_raw) reaches FLAG_THRESHOLD.
FLAG_MODES = ("sample", "threshold", "never")
FLAG_MAX_CLOCK = 5.0
# On 20k held-out games, 92% of the games where the head fires were lost on time by that player,
# usually within 2 plies; it fires in 80% as many games as humans flag in.
FLAG_THRESHOLD = 0.9
TYPICAL_TEMPERATURE = 0.5  # "typical" = mean of the sharpened time distribution


@dataclass
class Settings:
    temperature: float = 1.0
    top_p: float = 1.0
    think_time: str = "sample"
    time_temperature: float = 1.0
    time_top_p: float = 1.0
    time_scale: float = 1.0


def outcome(board: chess.Board) -> chess.Outcome | None:
    """The game's end as Lichess applies it: threefold repetition and the fifty-move rule end the game
    automatically, but only once reached. (python-chess's claim_draw also counts a repetition the
    player to move could make, ending the game a move early.)"""
    if out := board.outcome():
        return out
    if board.is_repetition(3):
        return chess.Outcome(chess.Termination.THREEFOLD_REPETITION, None)
    if board.halfmove_clock >= 100:
        return chess.Outcome(chess.Termination.FIFTY_MOVES, None)
    return None


@dataclass
class Decision:
    move: chess.Move
    think: float  # seconds, after time_scale
    prob: float  # model probability of the chosen move
    prediction: Prediction


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = x / temperature
    p = np.exp(x - x.max())
    return p / p.sum()


def sample(logits: np.ndarray, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    p = _softmax(logits, temperature)
    if top_p < 1:
        order = np.argsort(-p)
        keep = order[: int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1]
        q = np.zeros_like(p)
        q[keep] = p[keep]
        p = q / q.sum()
    return int(rng.choice(len(p), p=p))


GRID = 0.04  # seconds; the offset head's resolution
# Think-time histograms use the same 40 bars for every move, equally wide in log(1 + t / 1 s) from
# 0 to 120 s: about 0.13 s wide at 0, 0.24 s at 1 s, 1.3 s at 10 s and 7 s at 60 s.
LOG_SCALE = 1.0
BAR_EDGES = LOG_SCALE * np.expm1(np.linspace(0.0, np.log1p((TIME_BINS - 1) / LOG_SCALE), 41))
# The mode is the peak of the same distribution on the same axis, smoothed (Gaussian, half a bar
# wide) and evaluated on a fine grid, so it is continuous rather than one of the 40 bar centres.
_U_MAX = float(np.log1p((TIME_BINS - 1) / LOG_SCALE))
_MODE_GRID = np.linspace(0.0, _U_MAX, 1201)
_MODE_SIGMA = _U_MAX / 40 / 2


def time_distribution(time_logits: np.ndarray, offset=None, temperature: float = 1.0,
                      min_prob: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Think-time distribution as (seconds, probabilities) on the 40 ms grid from 0 to 120 s.

    With an offset head, every k with probability >= `min_prob` is spread over (k - 1, k + 1) s
    as the head predicts. Otherwise (and for the rest) k is spread evenly over k +- 0.5 s, as
    think_time samples it. Times below 0 count as 0; 120 means 120 s or more. `temperature`
    sharpens the whole-second part only.
    """
    pk = _softmax(time_logits, temperature)
    per_second = round(1 / GRID)
    p = np.zeros(per_second * (TIME_BINS - 1) + 1)
    ks = np.arange(len(pk))
    head = (pk >= min_prob) & (ks < TIME_BINS - 1) if offset is not None else np.zeros(len(pk), bool)
    spread = np.arange(-(per_second // 2), per_second // 2 + 1)  # 25 points = one second
    idx = per_second * ks[~head, None] + spread[None, :]
    np.add.at(p, np.clip(idx, 0, len(p) - 1), np.repeat(pk[~head, None] / len(spread), len(spread), 1))
    if head.any():
        po = np.exp(offset(ks[head]).astype(np.float64))
        po /= po.sum(axis=1, keepdims=True)
        idx = per_second * ks[head, None] + np.arange(len(CENTRES))[None, :] - len(CENTRES) // 2
        np.add.at(p, np.clip(idx, 0, len(p) - 1), pk[head, None] * po)
    return np.arange(len(p)) * GRID, p


@dataclass
class TimeStats:
    bins: np.ndarray  # probability in each BAR_EDGES bar
    mode: float  # centre of the highest bar (on the log axis)
    typical: float
    mean: float


def time_stats(time_logits: np.ndarray, offset=None) -> TimeStats:
    """Histogram and summary of a move's think time. The same for every move: no switches."""
    t, p = time_distribution(time_logits, offset)
    t_typ, p_typ = time_distribution(time_logits, offset, TYPICAL_TEMPERATURE)
    # Lichess clocks move on an 80 ms grid, which shows up as alternating 40 ms points:
    # average neighbours before binning so the narrow bars near 0 don't alias it.
    smooth = np.convolve(p, [0.5, 0.5])[: len(p)]
    smooth[-1] += 0.5 * p[-1]  # keep all of the last point (120 s or more)
    n = len(BAR_EDGES) - 1
    bins = np.bincount(np.clip(np.searchsorted(BAR_EDGES, t, side="right") - 1, 0, n - 1), smooth, n)
    return TimeStats(bins, _log_mode(t, p), float(t_typ @ p_typ), float(t @ p))


def _log_mode(t: np.ndarray, p: np.ndarray) -> float:
    """Peak of the think-time density on the log axis of the histogram (see _MODE_GRID)."""
    step = _MODE_GRID[1]
    hist = np.bincount(np.rint(np.log1p(t / LOG_SCALE) / step).astype(int), p, len(_MODE_GRID))
    half = int(np.ceil(4 * _MODE_SIGMA / step))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * step / _MODE_SIGMA) ** 2)
    # Reflect at 0 s, so a spike at 0 isn't halved by the edge.
    padded = np.concatenate([hist[half:0:-1], [2 * hist[0]], hist[1:], np.zeros(half)])
    density = np.convolve(padded, kernel, mode="valid")[: len(_MODE_GRID)]
    return float(LOG_SCALE * np.expm1(_MODE_GRID[int(np.argmax(density))]))


def think_time(time_logits: np.ndarray, s: Settings, rng: np.random.Generator,
               limit: float = float("inf"), offset=None) -> float:
    """Seconds a human would spend on this move, according to `s.think_time`.

    `limit` is the time available before flagging. The distribution is cut off there, i.e. the
    time a human would take given that they move in time. `offset(k)`, if given, returns the
    log-probs of the sub-second offset from k whole seconds (see marvin.offset).
    """
    if s.time_scale > 0 and limit < (len(time_logits) - 1) * s.time_scale:
        time_logits = time_logits.copy()
        time_logits[int(max(0.0, limit) / s.time_scale) + 1:] = -np.inf
    if s.think_time == "mode":
        t = time_stats(time_logits, offset).mode
    elif s.think_time in ("typical", "mean"):
        t, p = time_distribution(time_logits, offset, TYPICAL_TEMPERATURE if s.think_time == "typical" else 1.0)
        t = float(t @ p)
    else:  # sample, instant
        k = sample(time_logits, s.time_temperature, s.time_top_p, rng)
        if offset is not None and k < TIME_BINS - 1:
            # The offset head places the time within (k - 1, k + 1) s on a 40 ms grid.
            p = np.exp(offset(k).astype(np.float64))
            j = rng.choice(len(CENTRES), p=p / p.sum())
            t = max(0.0, k + CENTRES[j] + rng.uniform(-0.02, 0.02))
        else:  # whole seconds only: jitter within the bin
            t = max(0.0, k + rng.uniform(-0.5, 0.5))
    return min(t * s.time_scale, max(0.0, limit))


def typical_time(time_logits: np.ndarray, offset=None) -> float:
    return think_time(time_logits, Settings(think_time="typical"), np.random.default_rng(), offset=offset)


def move_offset(pred: Prediction, i: int):
    """Move i's offset function for think_time / time_stats, or None without an offset head."""
    return (lambda k: pred.offset(i, k)) if pred.offset else None


def runs_out_of_time(pred: Prediction, clock: float, mode: str, rng: np.random.Generator) -> bool:
    """Whether to let the clock run out instead of moving. `clock`: seconds left before this move."""
    if mode == "threshold":
        return pred.flag_raw >= FLAG_THRESHOLD
    if mode == "sample":
        return clock <= FLAG_MAX_CLOCK and rng.random() < pred.flag
    return False


def decide(model: Model, board: chess.Board, ctx: Context, s: Settings, rng: np.random.Generator,
           limit: float = float("inf")) -> Decision:
    """Sample a move, then its think time. `limit`: seconds available before flagging."""
    pred = model.predict(board, ctx)
    i = sample(pred.logits, s.temperature, s.top_p, rng)
    return Decision(pred.moves[i], think_time(pred.time_logits[i], s, rng, limit, move_offset(pred, i)),
                    float(pred.probs[i]), pred)


@dataclass
class MoveAnalysis:
    move: chess.Move
    prob: float
    time_logits: np.ndarray
    wdl: tuple[float, float, float]  # win, draw, loss for the side making the move
    offset: object = None  # sub-second offset function (see move_offset), None without an offset head


def analyse(model: Model, board: chess.Board, ctx: Context,
            top_p: float = 1.0) -> tuple[Prediction, list[MoveAnalysis]]:
    """The model's prediction for `board`, plus each move's outcome from evaluating the position after it.

    Only the most likely moves covering `top_p` of the probability are analysed (each needs its own
    evaluation); their probabilities are still those of the full distribution. The position after a
    move is evaluated as the opponent would face it: ratings and clocks swapped, and the move's
    typical think time added to the time history.
    """
    pred = model.predict(board, ctx)
    probs = pred.probs
    order = np.argsort(-probs)
    covered = np.cumsum(probs[order])
    order = order[: int(np.searchsorted(covered, min(top_p, covered[-1]) - 1e-9)) + 1]
    rows, pending = [], []
    for i in order:
        move, tl = pred.moves[i], pred.time_logits[i]
        child = board.copy()
        child.push(move)
        end = outcome(child)
        wdl = None
        if end:
            wdl = (0.0, 1.0, 0.0) if end.winner is None else (1.0, 0.0, 0.0)
        else:
            t = typical_time(tl, move_offset(pred, i))
            spent = t - ctx.inc if board.ply() >= 2 else 0.0
            pending.append((len(rows), child, Context(ctx.opp_elo, ctx.elo, ctx.opp_clock, max(0.0, ctx.clock - spent),
                                                      ctx.opp_inc, ctx.inc, ctx.base, [t] + ctx.times)))
        rows.append(MoveAnalysis(move, float(probs[i]), tl, wdl, move_offset(pred, i)))
    replies = model.predict_many([(child, c) for _, child, c in pending])
    for (k, _, _), reply in zip(pending, replies):
        rows[k].wdl = (reply.loss, reply.draw, reply.win)  # the opponent's loss is the mover's win
    return pred, rows


def _placement(tokens: np.ndarray) -> str:
    if not tokens.any():
        return "-"
    board = chess.Board.empty()
    for sq, t in enumerate(tokens):
        if t:
            board.set_piece_at(sq, chess.Piece((int(t) - 1) % 6 + 1, chess.WHITE if t <= 6 else chess.BLACK))
    return board.board_fen()


def _nums(xs, fmt: str = "g") -> str:
    return " ".join(format(float(x), fmt) for x in xs)


def explain(board: chess.Board, ctx: Context, d: Decision, s: Settings) -> list[str]:
    """Human-readable dump of the exact model inputs and outputs behind a decision."""
    x = encode(board, ctx)
    sc = x["scalars"]
    ep = np.flatnonzero(x["ep_mask"])
    lines = ["INPUT  (boards from the side to move's view, newest first)"]
    lines += [f"  board[{i}]      {_placement(t)}" for i, t in enumerate(x["board_history"])]
    lines += [
        f"  time_history  {_nums(x['time_history'], '.2f')}",
        f"  rep_flags     {_nums(x['rep_flags'])}",
        f"  castling      {_nums(x['castling'])}   (white K Q, black K Q)",
        f"  ep_mask       {chess.square_name(int(ep[0])) if len(ep) else '-'}",
        f"  legal_mask    {int(x['legal_mask'].sum())} moves",
        f"  scalars       elo {sc[0]:+.3f} ({ctx.elo:g})   opp_elo {sc[1]:+.3f} ({ctx.opp_elo:g})",
        f"                ply {sc[2]:.2f}   halfmove {sc[7]:.2f}   base {sc[8]:g}",
        f"                clock {sc[3]:.1f}   opp_clock {sc[4]:.1f}   inc {sc[5]:g}   opp_inc {sc[6]:g}",
        "OUTPUT",
        f"  win {d.prediction.win:.3f}   draw {d.prediction.draw:.3f}   loss {d.prediction.loss:.3f}"
        f"   resign {d.prediction.resign:.3f}   flag {d.prediction.flag:.3f}",
    ]
    probs = d.prediction.probs
    for i in np.argsort(-probs)[:8]:
        ts = time_stats(d.prediction.time_logits[i], move_offset(d.prediction, i))
        lines.append(f"  {board.san(d.prediction.moves[i]):7s} {probs[i]:6.1%}   think: mode {ts.mode:.2f} s"
                     f"   typical {ts.typical:.2f} s   mean {ts.mean:.2f} s")
    lines += [
        "DECISION",
        f"  {board.san(d.move)} ({d.prob:.1%}), think {d.think:.2f} s",
        f"  {s.think_time} · temperature {s.temperature:g} · top_p {s.top_p:g} · time temperature"
        f" {s.time_temperature:g} · time top_p {s.time_top_p:g} · time scale {s.time_scale:g}",
    ]
    return lines


def centipawns(pred: Prediction) -> int:
    """lc0's WDL-to-centipawn mapping, for GUIs that expect a score."""
    q = float(np.clip(pred.win - pred.loss, -0.999, 0.999))
    return int(round(90 * np.tan(1.5637541897 * q)))
