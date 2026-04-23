"""Block DCT + JPEG-style quantization lossy codec.

Pipeline (encode):
    image (uint8)
      → centre to [-128, 127]
      → split into 8x8 blocks (zero padded to multiples of 8)
      → 2D DCT-II per block (separable, orthonormal)
      → divide by Q (JPEG luminance matrix scaled by quality factor)
      → round to nearest integer, clip to [-128, 127]
      → pack as 8-bit two's-complement integers, block-major zigzag order

Decode is the exact inverse: unpack → dequantize → inverse DCT → clip & round.

Bit layout:
    36-bit header = 4 bits b | 8 bits Q | 12 bits H | 12 bits W
                    where b ∈ {2,...,12} is the per-coefficient bit width,
                    chosen at encode time to just fit the quantized range.
    payload       = 64 * n_blocks coefficients, b bits each (signed two's
                    complement), block-major and zigzag-ordered within a block.

Because `b` shrinks as the quality factor decreases, the total bit count
reflects a genuine rate-distortion trade-off without needing a full entropy
coder.
"""
from __future__ import annotations

import numpy as np

from .. import bitio


BLOCK = 8
MIN_COEFF_BITS = 2
MAX_COEFF_BITS = 12
HEADER_BITS = 4 + 8 + 12 + 12


JPEG_LUMA_Q50 = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def _zigzag_indices() -> np.ndarray:
    """Return a (64,) array that permutes a flat 8x8 block into zigzag order."""
    order: list[tuple[int, int]] = []
    for s in range(2 * BLOCK - 1):
        diag = [(i, s - i) for i in range(BLOCK) if 0 <= s - i < BLOCK]
        if s % 2 == 0:
            diag.reverse()
        order.extend(diag)
    return np.array([i * BLOCK + j for i, j in order], dtype=np.int64)


ZIGZAG = _zigzag_indices()
INV_ZIGZAG = np.argsort(ZIGZAG)


def _dct_matrix(n: int = BLOCK) -> np.ndarray:
    k = np.arange(n)
    i = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[:, 0] *= 1.0 / np.sqrt(2.0)
    m *= np.sqrt(2.0 / n)
    return m.T  # so that y = D @ x computes DCT-II


DCT_M = _dct_matrix()


def _dct2(block: np.ndarray) -> np.ndarray:
    return DCT_M @ block @ DCT_M.T


def _idct2(block: np.ndarray) -> np.ndarray:
    return DCT_M.T @ block @ DCT_M


def quality_to_qmatrix(quality: int) -> np.ndarray:
    """JPEG-style scaling of the luminance quantization matrix.

    quality ∈ [1,99]. Higher quality → smaller Q entries → finer quantization.
    """
    q = int(np.clip(quality, 1, 99))
    if q < 50:
        scale = 5000.0 / q
    else:
        scale = 200.0 - 2 * q
    qm = np.floor((JPEG_LUMA_Q50 * scale + 50) / 100)
    qm = np.clip(qm, 1, 255)
    return qm.astype(np.float64)


def _pack_header(b: int, q: int, h: int, w: int) -> np.ndarray:
    if not (1 <= q <= 99):
        raise ValueError("quality must be in 1..99")
    if not (MIN_COEFF_BITS <= b <= MAX_COEFF_BITS):
        raise ValueError(f"coeff bits must be in {MIN_COEFF_BITS}..{MAX_COEFF_BITS}")
    if h >= (1 << 12) or w >= (1 << 12):
        raise ValueError("image too large for DCT codec header (max 4095x4095)")
    hdr = (b << 32) | (q << 24) | (h << 12) | w
    return np.array(
        [(hdr >> i) & 1 for i in range(HEADER_BITS - 1, -1, -1)], dtype=np.uint8
    )


def _unpack_header(bits: np.ndarray) -> tuple[int, int, int, int]:
    hdr = 0
    for bit in bits[:HEADER_BITS]:
        hdr = (hdr << 1) | int(bit)
    b = (hdr >> 32) & 0xF
    q = (hdr >> 24) & 0xFF
    h = (hdr >> 12) & 0xFFF
    w = hdr & 0xFFF
    return b, q, h, w


def _bits_for_range(values: np.ndarray) -> int:
    """Smallest b in [MIN_COEFF_BITS, MAX_COEFF_BITS] that fits all values."""
    if len(values) == 0:
        return MIN_COEFF_BITS
    m = int(np.max(np.abs(values)))
    for b in range(MIN_COEFF_BITS, MAX_COEFF_BITS + 1):
        if m <= (1 << (b - 1)) - 1:
            return b
    return MAX_COEFF_BITS


def encode(image: np.ndarray, quality: int = 50) -> np.ndarray:
    """Encode a grayscale (H,W) uint8 image into a bit sequence."""
    if image.ndim != 2:
        raise ValueError("expected a 2D grayscale image")
    h, w = image.shape
    hp = ((h + BLOCK - 1) // BLOCK) * BLOCK
    wp = ((w + BLOCK - 1) // BLOCK) * BLOCK

    padded = np.zeros((hp, wp), dtype=np.float64)
    padded[:h, :w] = image
    if hp > h:
        padded[h:, :w] = image[-1:, :]   # edge replication keeps DCT smooth
    if wp > w:
        padded[:h, w:] = padded[:h, w - 1 : w]
    if hp > h and wp > w:
        padded[h:, w:] = padded[h - 1, w - 1]

    padded -= 128.0
    qm = quality_to_qmatrix(quality)

    nby = hp // BLOCK
    nbx = wp // BLOCK
    coeffs = np.empty(nby * nbx * BLOCK * BLOCK, dtype=np.int32)

    k = 0
    for by in range(nby):
        for bx in range(nbx):
            blk = padded[by * BLOCK : (by + 1) * BLOCK, bx * BLOCK : (bx + 1) * BLOCK]
            c = np.round(_dct2(blk) / qm).astype(np.int32)
            coeffs[k : k + BLOCK * BLOCK] = c.reshape(-1)[ZIGZAG]
            k += BLOCK * BLOCK

    coeff_bits = _bits_for_range(coeffs)
    payload = bitio.ints_to_bits(coeffs, coeff_bits)
    return np.concatenate([_pack_header(coeff_bits, quality, h, w), payload])


def decode(bits: np.ndarray) -> np.ndarray:
    """Inverse of `encode`."""
    coeff_bits, q, h, w = _unpack_header(bits)
    hp = ((h + BLOCK - 1) // BLOCK) * BLOCK
    wp = ((w + BLOCK - 1) // BLOCK) * BLOCK
    nby = hp // BLOCK
    nbx = wp // BLOCK
    n_coeffs = nby * nbx * BLOCK * BLOCK

    body = bits[HEADER_BITS : HEADER_BITS + n_coeffs * coeff_bits]
    coeffs = bitio.bits_to_ints(body, coeff_bits)
    if len(coeffs) < n_coeffs:
        coeffs = np.concatenate(
            [coeffs, np.zeros(n_coeffs - len(coeffs), dtype=np.int64)]
        )

    qm = quality_to_qmatrix(q)
    out = np.zeros((hp, wp), dtype=np.float64)

    k = 0
    for by in range(nby):
        for bx in range(nbx):
            zz = coeffs[k : k + BLOCK * BLOCK]
            blk = np.zeros(BLOCK * BLOCK, dtype=np.float64)
            blk[ZIGZAG] = zz
            blk = blk.reshape(BLOCK, BLOCK) * qm
            out[by * BLOCK : (by + 1) * BLOCK, bx * BLOCK : (bx + 1) * BLOCK] = _idct2(
                blk
            )
            k += BLOCK * BLOCK

    out += 128.0
    out = np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out[:h, :w]
