"""Uniform scalar quantization of grayscale images.

Each 8-bit pixel is mapped to a `k`-bit index (k ∈ {1,...,8}). The decoder
maps indices back to the centre of each quantization bin. This is the
simplest lossy source coder and serves as a baseline.

Bit layout:
    header (32 bits) = 8 bits k | 12 bits H | 12 bits W
    payload          = H*W indices, each k bits, MSB first, row-major
"""
from __future__ import annotations

import numpy as np

from .. import bitio


HEADER_BITS = 8 + 12 + 12


def _pack_header(k: int, h: int, w: int) -> np.ndarray:
    if not (1 <= k <= 8):
        raise ValueError("k must be in 1..8")
    if h >= (1 << 12) or w >= (1 << 12):
        raise ValueError("image too large for scalar-quantizer header (max 4095x4095)")
    hdr = (k << 24) | (h << 12) | w
    bits = np.array(
        [(hdr >> i) & 1 for i in range(HEADER_BITS - 1, -1, -1)],
        dtype=np.uint8,
    )
    return bits


def _unpack_header(bits: np.ndarray) -> tuple[int, int, int]:
    hdr = 0
    for b in bits[:HEADER_BITS]:
        hdr = (hdr << 1) | int(b)
    k = (hdr >> 24) & 0xFF
    h = (hdr >> 12) & 0xFFF
    w = hdr & 0xFFF
    return k, h, w


def encode(image: np.ndarray, k: int) -> np.ndarray:
    """Encode a grayscale (H,W) uint8 image into a bit sequence."""
    if image.ndim != 2:
        raise ValueError("expected a 2D grayscale image")
    if not (1 <= k <= 8):
        raise ValueError("k must be in 1..8")
    h, w = image.shape
    levels = 1 << k
    step = 256 // levels
    idx = np.clip(image.astype(np.int32) // step, 0, levels - 1).astype(np.uint32)
    flat = idx.reshape(-1)

    shifts = np.arange(k - 1, -1, -1, dtype=np.uint32)
    payload = ((flat[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)

    return np.concatenate([_pack_header(k, h, w), payload])


def decode(bits: np.ndarray) -> np.ndarray:
    """Decode a bit sequence produced by `encode` back to an image."""
    k, h, w = _unpack_header(bits)
    levels = 1 << k
    step = 256 // levels
    body = bits[HEADER_BITS : HEADER_BITS + k * h * w]
    body = body[: (len(body) // k) * k].reshape(-1, k).astype(np.uint32)
    shifts = np.arange(k - 1, -1, -1, dtype=np.uint32)
    idx = (body << shifts).sum(axis=1)
    centres = (idx * step + step // 2).astype(np.int32)
    centres = np.clip(centres, 0, 255).astype(np.uint8)
    return centres.reshape(h, w)
