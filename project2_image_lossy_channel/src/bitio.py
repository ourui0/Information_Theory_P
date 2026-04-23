"""Bit-level packing / unpacking utilities.

Bits are represented as numpy uint8 arrays with values in {0,1}. A ternary
representation (0/1/2) is used on the BEC, where `2` means "erased".
"""
from __future__ import annotations

import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Unpack a bytes object to a uint8 array of 0/1 (MSB first)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Pack a uint8 array of 0/1 back into bytes (MSB first, zero padded)."""
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


def ints_to_bits(values: np.ndarray, bits_per_value: int) -> np.ndarray:
    """Encode signed integers as two's complement bits (MSB first).

    Values are clipped to the representable range of `bits_per_value` bits.
    Output shape: (len(values) * bits_per_value,) uint8 in {0,1}.
    """
    if bits_per_value <= 0 or bits_per_value > 32:
        raise ValueError("bits_per_value must be in 1..32")
    lo = -(1 << (bits_per_value - 1))
    hi = (1 << (bits_per_value - 1)) - 1
    v = np.clip(np.asarray(values, dtype=np.int64), lo, hi)
    mask = (1 << bits_per_value) - 1
    u = (v & mask).astype(np.uint64)
    shifts = np.arange(bits_per_value - 1, -1, -1, dtype=np.uint64)
    bits = ((u[:, None] >> shifts) & 1).astype(np.uint8)
    return bits.reshape(-1)


def bits_to_ints(bits: np.ndarray, bits_per_value: int) -> np.ndarray:
    """Inverse of ints_to_bits — recover signed integers from a bitstream."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = len(bits) // bits_per_value
    bits = bits[: n * bits_per_value].reshape(n, bits_per_value).astype(np.uint64)
    shifts = np.arange(bits_per_value - 1, -1, -1, dtype=np.uint64)
    u = (bits << shifts).sum(axis=1)
    sign_bit = 1 << (bits_per_value - 1)
    mask = (1 << bits_per_value) - 1
    v = np.where(u & sign_bit, (u | ~np.uint64(mask)).astype(np.int64), u.astype(np.int64))
    return v


def pad_to_multiple(bits: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """Right-pad a bit array with zeros so its length is a multiple of k."""
    pad = (-len(bits)) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits, pad
