"""Systematic Hamming(7,4) code.

Generator and parity-check matrices (rows are systematic form):
    G = [I_4 | P]       shape (4, 7)
    H = [P^T | I_3]     shape (3, 7)

Rate = 4/7. Corrects 1 bit error (BSC) or up to 2 erasures reliably (BEC);
3 erasures can often be corrected when the corresponding columns of H are
linearly independent, which is the case for most patterns.

On the BSC we use classical syndrome decoding (single-error correction).
On the BEC we enumerate the 2^e candidate fillings of the e erased bits
(capped at 2**MAX_ERASE_ENUM) and keep the first that has zero syndrome.
When the erased pattern is uncorrectable, remaining bits are drawn at random.
"""
from __future__ import annotations

import numpy as np

from . import bec as bec_mod


G = np.array(
    [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.uint8,
)

H = np.array(
    [
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1],
    ],
    dtype=np.uint8,
)


def _syndrome_table() -> dict[tuple[int, int, int], int]:
    """Map a syndrome (s0,s1,s2) → index of the single-bit error (or -1)."""
    table: dict[tuple[int, int, int], int] = {(0, 0, 0): -1}
    for j in range(7):
        s = tuple(int(x) for x in H[:, j])
        table[s] = j
    return table


SYNDROME = _syndrome_table()
MAX_ERASE_ENUM = 5   # 2**5 = 32 candidate fillings per 7-bit word — tiny


def encode(bits: np.ndarray) -> np.ndarray:
    """Encode a bit stream with systematic Hamming(7,4).

    Length is padded with zeros to a multiple of 4. The padding is later
    discarded by the decoder using an explicit `n_info_bits` parameter.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-len(bits)) % 4
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    blocks = bits.reshape(-1, 4)
    codewords = (blocks @ G) % 2
    return codewords.reshape(-1).astype(np.uint8)


def decode_bsc(received: np.ndarray, n_info_bits: int) -> np.ndarray:
    r = np.asarray(received, dtype=np.uint8)
    r = r[: (len(r) // 7) * 7].reshape(-1, 7)
    corrected = r.copy()
    syn = (r @ H.T) % 2
    for i, s in enumerate(syn):
        pos = SYNDROME.get(tuple(int(x) for x in s), -1)
        if pos >= 0:
            corrected[i, pos] ^= 1
    info = corrected[:, :4].reshape(-1)
    return info[:n_info_bits]


def decode_bec(
    received: np.ndarray,
    n_info_bits: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    r = np.asarray(received, dtype=np.int16)
    r = r[: (len(r) // 7) * 7].reshape(-1, 7)
    out = np.zeros(r.shape, dtype=np.uint8)

    for i, word in enumerate(r):
        erased = np.where(word == int(bec_mod.ERASURE))[0]
        base = np.where(word == int(bec_mod.ERASURE), 0, word).astype(np.uint8)

        if len(erased) == 0:
            out[i] = base
            continue

        if len(erased) > MAX_ERASE_ENUM:
            base[erased] = rng.integers(0, 2, size=len(erased), dtype=np.uint8)
            out[i] = base
            continue

        solved = False
        for mask in range(1 << len(erased)):
            cand = base.copy()
            for j, pos in enumerate(erased):
                cand[pos] = (mask >> j) & 1
            if np.all((H @ cand) % 2 == 0):
                out[i] = cand
                solved = True
                break
        if not solved:
            base[erased] = rng.integers(0, 2, size=len(erased), dtype=np.uint8)
            out[i] = base

    info = out[:, :4].reshape(-1)
    return info[:n_info_bits]
