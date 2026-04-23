"""Repetition code (n, 1).

Each information bit is repeated `n` times (n odd).
    BSC decoder : majority vote.
    BEC decoder : majority vote over non-erased copies; if every copy is
                  erased, a uniformly random bit is drawn.

Rate = 1/n. This is the simplest channel code and gives a clear
"coding-gain vs rate" trade-off curve on both channels.
"""
from __future__ import annotations

import numpy as np

from . import bec as bec_mod


def encode(bits: np.ndarray, n: int) -> np.ndarray:
    if n < 1 or n % 2 == 0:
        raise ValueError("n must be a positive odd integer")
    bits = np.asarray(bits, dtype=np.uint8)
    return np.repeat(bits, n)


def _reshape_groups(received: np.ndarray, n: int) -> np.ndarray:
    k = len(received) // n
    return np.asarray(received[: k * n], dtype=np.int16).reshape(k, n)


def decode_bsc(received: np.ndarray, n: int) -> np.ndarray:
    groups = _reshape_groups(received, n)
    votes = groups.sum(axis=1)
    return (votes > (n // 2)).astype(np.uint8)


def decode_bec(
    received: np.ndarray, n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    groups = _reshape_groups(received, n)
    erased = groups == int(bec_mod.ERASURE)
    ones = ((groups == 1) & ~erased).sum(axis=1)
    zeros = ((groups == 0) & ~erased).sum(axis=1)

    out = np.zeros(groups.shape[0], dtype=np.uint8)
    decided = ones != zeros
    out[decided] = (ones[decided] > zeros[decided]).astype(np.uint8)

    tie = ~decided
    if tie.any():
        out[tie] = rng.integers(0, 2, size=int(tie.sum()), dtype=np.uint8)
    return out
