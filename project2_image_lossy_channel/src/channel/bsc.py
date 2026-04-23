"""Binary Symmetric Channel: each bit is independently flipped with prob p."""
from __future__ import annotations

import numpy as np


def transmit(bits: np.ndarray, p: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Simulate transmission of `bits` over BSC(p). Returns received bits."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    rng = rng or np.random.default_rng()
    bits = np.asarray(bits, dtype=np.uint8)
    flips = (rng.random(bits.shape) < p).astype(np.uint8)
    return bits ^ flips
