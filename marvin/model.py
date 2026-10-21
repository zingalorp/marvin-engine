"""ONNX model wrapper: encodes a position exactly as in training and decodes the outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import chess
import numpy as np
import onnxruntime as ort

from marvin.offset import OffsetHead

HISTORY = 8
TIME_BINS = 121  # bin k = k seconds; the last bin means 120 s or more
# The flag head was trained with BCE pos_weight 50, which multiplies its odds by ~50. Dividing them
# back out gives probabilities that match real flag rates (checked on 2M low-clock positions).
# Current weight files do this inside the model (metadata flag_logit=calibrated); for older files
# the correction is applied here.
FLAG_POS_WEIGHT = 50.0
ELO_MIN, ELO_MAX = 1200, 2500
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

_PIECE = {chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3, chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6}
_PROMO = {chess.QUEEN: 0, chess.ROOK: 1, chess.BISHOP: 2, chess.KNIGHT: 3}


@dataclass
class Context:
    """Everything besides the board that the model conditions on. All times in seconds."""

    elo: float
    opp_elo: float
    clock: float
    opp_clock: float
    inc: float
    opp_inc: float
    base: float
    times: list[float] = field(default_factory=list)  # time spent on previous plies, newest first


@dataclass
class Prediction:
    moves: list[chess.Move]
    logits: np.ndarray  # one per move
    time_logits: np.ndarray  # (len(moves), TIME_BINS), conditioned on each move
    win: float
    draw: float
    loss: float
    resign: float
    flag: float  # probability of running out of time here instead of moving
    flag_raw: float = 0.0  # the head's training-time output, before the pos-weight correction
    # offset(i, k): log-probs of move i's sub-second offset given k whole seconds (marvin.offset);
    # None for models without an offset head.
    offset: Callable[[int, int], np.ndarray] | None = None

    @property
    def probs(self) -> np.ndarray:
        p = np.exp(self.logits - self.logits.max())
        return p / p.sum()


def _canonical(board: chess.Board) -> chess.Board:
    return board if board.turn == chess.WHITE else board.mirror()


def _tokens(board: chess.Board) -> np.ndarray:
    t = np.zeros(64, np.int64)
    for sq, p in board.piece_map().items():
        t[sq] = _PIECE[p.piece_type] + (0 if p.color == chess.WHITE else 6)
    return t


def _whole_seconds(t: float) -> float:
    """Training saw Lichess clocks, which round to whole seconds (half up); real clocks are finer."""
    return float(np.floor(max(0.0, t) + 0.5))


def encode(board: chess.Board, ctx: Context) -> dict[str, np.ndarray]:
    """Model inputs for the side to move. History comes from `board.move_stack`."""
    # Current position plus up to 7 predecessors, newest first, each seen from its own mover.
    b = board.copy()
    history = [_canonical(b.copy(stack=False))]
    while b.move_stack and len(history) < HISTORY:
        b.pop()
        history.append(_canonical(b.copy(stack=False)))

    boards = np.zeros((HISTORY, 64), np.int64)
    rep = np.zeros(HISTORY, np.float32)
    seen: set[str] = set()
    for i, h in enumerate(history):
        boards[i] = _tokens(h)
        rep[i] = h.board_fen() in seen
        seen.add(h.board_fen())

    times = np.zeros(HISTORY, np.float32)
    recent = ctx.times[:HISTORY]
    times[: len(recent)] = [_whole_seconds(t) for t in recent]

    legal = np.zeros(4096, bool)
    for mv in history[0].legal_moves:
        legal[mv.from_square * 64 + mv.to_square] = True

    ep = np.zeros(64, np.float32)
    if board.ep_square is not None:
        ep[board.ep_square] = 1.0

    ply = (board.fullmove_number - 1) * 2 + (board.turn == chess.BLACK)
    scalars = [
        (ctx.elo - 1900) / 700, (ctx.opp_elo - 1900) / 700, ply / 100,
        _whole_seconds(ctx.clock), _whole_seconds(ctx.opp_clock), ctx.inc, ctx.opp_inc,
        board.halfmove_clock / 100, ctx.base,
    ]
    castling = [
        board.has_kingside_castling_rights(chess.WHITE), board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK), board.has_queenside_castling_rights(chess.BLACK),
    ]
    return {
        "board_history": boards,
        "time_history": times,
        "rep_flags": rep,
        "castling": np.array(castling, np.float32),
        "ep_mask": ep,
        "scalars": np.array(scalars, np.float32),
        "legal_mask": legal,
    }


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def _sigmoid(x) -> float:
    return float(1 / (1 + np.exp(-float(np.ravel(x)[0]))))


class Model:
    def __init__(self, name_or_path: str = "79m", device: str = "auto"):
        path = Path(name_or_path)
        if not path.suffix:
            path = WEIGHTS_DIR / f"marvin_{name_or_path}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python download_weights.py")
        providers = ["CPUExecutionProvider"]
        if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")
        ort.set_default_logger_severity(3)  # the export triggers harmless ScatterND warnings
        self.session = ort.InferenceSession(str(path), providers=providers)
        self.name = path.stem
        offset_path = path.with_suffix(".offset.npz")
        has_features = {"time_hidden", "square_features"} <= {o.name for o in self.session.get_outputs()}
        self.offset_head = OffsetHead(offset_path) if offset_path.exists() and has_features else None
        calibrated = self.session.get_modelmeta().custom_metadata_map.get("flag_logit") == "calibrated"
        self.flag_shift = 0.0 if calibrated else float(np.log(FLAG_POS_WEIGHT))
        self.outputs = _OUTPUTS + (["time_hidden", "square_features"] if self.offset_head else [])

    def predict(self, board: chess.Board, ctx: Context) -> Prediction:
        return self.predict_many([(board, ctx)])[0]

    def predict_many(self, items: list[tuple[chess.Board, Context]]) -> list[Prediction]:
        """One batched forward pass for several positions."""
        if not items:
            return []
        encoded = [encode(board, ctx) for board, ctx in items]
        feed = {k: np.stack([e[k] for e in encoded]) for k in encoded[0]}
        outputs = self.session.run(self.outputs, feed)
        preds = []
        for i, (board, _) in enumerate(items):
            pred, canonical = _decode(board, *(o[i] for o in outputs[: len(_OUTPUTS)]), flag_shift=self.flag_shift)
            if self.offset_head:
                pred.offset = self._offset_fn(pred, canonical, outputs[8][i], outputs[9][i])
            preds.append(pred)
        return preds

    def _offset_fn(self, pred: Prediction, canonical: list[chess.Move], hidden: np.ndarray,
                   squares: np.ndarray) -> Callable[[int, int], np.ndarray]:
        def offset(i: int, k: int) -> np.ndarray:
            cm = canonical[i]
            return self.offset_head.log_probs(hidden, squares[cm.from_square], squares[cm.to_square],
                                              pred.time_logits[i], k)
        return offset


_OUTPUTS = ["move_logits", "wdl_logits", "resign_logit", "flag_logit",
            "time_baseline", "time_from_logits", "time_to_logits", "promo_logits"]


def _decode(board: chess.Board, move, wdl, resign, flag, t_base, t_from, t_to,
            promo, flag_shift: float = 0.0) -> tuple[Prediction, list[chess.Move]]:
    """The prediction, plus each move as seen from the side to move (what the model indexes by)."""
    promo = _log_softmax(promo)
    wdl = np.exp(_log_softmax(wdl))  # order: loss, draw, win
    moves, logits, time_logits, canonical = [], [], [], []
    flip = 56 if board.turn == chess.BLACK else 0
    for cm in _canonical(board).legal_moves:
        logit = move[cm.from_square * 64 + cm.to_square]
        if cm.promotion:
            logit += promo[cm.to_square - 56, _PROMO[cm.promotion]]
        moves.append(chess.Move(cm.from_square ^ flip, cm.to_square ^ flip, cm.promotion))
        canonical.append(cm)
        logits.append(logit)
        time_logits.append(t_base + t_from[cm.from_square] + t_to[cm.to_square])
    return Prediction(
        moves=moves,
        logits=np.array(logits, np.float64),
        time_logits=np.array(time_logits, np.float64).reshape(len(moves), TIME_BINS),
        win=float(wdl[2]), draw=float(wdl[1]), loss=float(wdl[0]),
        resign=_sigmoid(resign), flag=_sigmoid(float(np.ravel(flag)[0]) - flag_shift),
        flag_raw=_sigmoid(float(np.ravel(flag)[0]) - flag_shift + np.log(FLAG_POS_WEIGHT)),
    ), canonical
