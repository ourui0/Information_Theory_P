"""Evaluation metrics: MSE, PSNR, bit-error rate, accuracy."""
from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two images (cast to float)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio in dB.

    Returns +inf when the two images are identical.
    """
    e = mse(a, b)
    if e == 0.0:
        return float("inf")
    return 10.0 * np.log10((max_val ** 2) / e)


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit Error Rate between transmitted and received bit sequences."""
    tx = np.asarray(tx_bits, dtype=np.int8)
    rx = np.asarray(rx_bits, dtype=np.int8)
    n = min(len(tx), len(rx))
    if n == 0:
        return 0.0
    return float(np.mean(tx[:n] != rx[:n]))


def pixel_accuracy(a: np.ndarray, b: np.ndarray, tol: int = 0) -> float:
    """Fraction of pixels whose absolute difference is ≤ tol."""
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float(np.mean(diff <= tol))
