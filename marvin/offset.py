"""Sub-second think time: the offset head, run in NumPy.

The time head predicts whole seconds k (training clocks were rounded to the
second). Models exported with a `<name>.offset.npz` next to the .onnx also have
a small head trained on centisecond clocks that predicts where the true time
lies within (k - 1 s, k + 1 s), in 40 ms steps, given the position and the move.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BINS = 49
CENTRES = (-96 + 4 * np.arange(BINS)) / 100.0  # seconds relative to k
_FLOOR = -30.0


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def _mish(x: np.ndarray) -> np.ndarray:
    return x * np.tanh(np.logaddexp(0.0, x))


class OffsetHead:
    def __init__(self, path: Path):
        self.w = dict(np.load(path))

    def log_probs(self, hidden: np.ndarray, x_from: np.ndarray, x_to: np.ndarray,
                  time_logits: np.ndarray, k) -> np.ndarray:
        """log p(offset bin | k) for one move: shape (BINS,) for an int k, (len(k), BINS) for an array.

        `time_logits`: that move's 121 whole-second logits. Only the k-dependent inputs differ
        between rows, so the shared part of the first layer is computed once.
        """
        w = self.w
        ks = np.atleast_1d(np.asarray(k, dtype=np.int64))
        tl = time_logits - time_logits.max()
        lp = np.maximum(tl - np.log(np.exp(tl).sum()), _FLOOR)
        lp = np.concatenate([[_FLOOR], lp, [_FLOOR]])
        l3 = np.stack([lp[ks], lp[ks + 1], lp[ks + 2]], axis=1)
        local = np.concatenate([l3, l3[:, :1] - l3[:, 1:2], l3[:, 2:] - l3[:, 1:2]], axis=1) / 10.0
        shared = np.concatenate([
            _layer_norm(hidden, w["ln_hidden.weight"], w["ln_hidden.bias"]),
            _layer_norm(x_from, w["ln_from.weight"], w["ln_from.bias"]),
            _layer_norm(x_to, w["ln_to.weight"], w["ln_to.bias"]),
        ])
        w1 = w["mlp.0.weight"]
        per_k = np.concatenate([w["k_emb.weight"][ks], local], axis=1)
        z = _mish(w1[:, :len(shared)] @ shared + per_k @ w1[:, len(shared):].T + w["mlp.0.bias"])
        z = _mish(z @ w["mlp.3.weight"].T + w["mlp.3.bias"])
        logits = z @ w["mlp.6.weight"].T + w["mlp.6.bias"]
        logits = logits - logits.max(axis=1, keepdims=True)
        out = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        return out[0] if np.ndim(k) == 0 else out
