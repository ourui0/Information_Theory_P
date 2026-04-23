"""Binary Erasure Channel: each bit is erased (→ 2) independently with prob p.

Received symbols are encoded as uint8 values in {0, 1, 2}, where 2 = erasure.
"""
from __future__ import annotations

import numpy as np

ERASURE = np.uint8(2)


def transmit(bits: np.ndarray, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    rng = rng or np.random.default_rng()
    bits = np.asarray(bits, dtype=np.uint8)
    out = bits.copy()
    erased = rng.random(bits.shape) < p
    out[erased] = ERASURE
    return out


def fill_erasures_random(
    received: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Final fallback: replace remaining erasures (value 2) with random bits.

    Used only when no channel code is applied, so the pipeline can still
    produce a decoded image.
    """
    rng = rng or np.random.default_rng()
    out = received.copy()
    mask = out == ERASURE
    if mask.any():
        out[mask] = rng.integers(0, 2, size=int(mask.sum()), dtype=np.uint8)
    return out
