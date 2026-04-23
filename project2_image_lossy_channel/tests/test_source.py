"""Tests for the lossy source coders (uniform quantizer, DCT codec)."""
from __future__ import annotations

import unittest

import numpy as np

from . import _common  # noqa: F401
from src import metrics
from src.source import dct_codec, quantize


def _random_image(shape: tuple[int, int] = (32, 32), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _smooth_image(shape: tuple[int, int] = (32, 32)) -> np.ndarray:
    h, w = shape
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    img = 127.5 * (1 + np.cos(2 * np.pi * xx) * np.cos(2 * np.pi * yy))
    return np.clip(img, 0, 255).astype(np.uint8)


class TestScalarQuantizer(unittest.TestCase):
    def test_lossless_at_k8(self) -> None:
        img = _random_image()
        bits = quantize.encode(img, k=8)
        out = quantize.decode(bits)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(metrics.mse(img, out), 0.0)

    def test_bit_budget(self) -> None:
        img = _random_image((16, 24))
        for k in (1, 2, 4, 6, 8):
            bits = quantize.encode(img, k=k)
            expected = quantize.HEADER_BITS + k * img.size
            self.assertEqual(len(bits), expected)

    def test_psnr_monotonic_in_k(self) -> None:
        """Lower k should never give higher PSNR on a smooth image."""
        img = _smooth_image()
        psnrs = []
        for k in (1, 2, 4, 6, 8):
            out = quantize.decode(quantize.encode(img, k=k))
            psnrs.append(metrics.psnr(img, out))
        for i in range(len(psnrs) - 1):
            self.assertLessEqual(psnrs[i], psnrs[i + 1] + 1e-6)

    def test_shape_preserved_non_aligned(self) -> None:
        img = _random_image((17, 23))
        out = quantize.decode(quantize.encode(img, k=4))
        self.assertEqual(out.shape, img.shape)

    def test_invalid_k(self) -> None:
        img = _random_image()
        with self.assertRaises(ValueError):
            quantize.encode(img, k=0)
        with self.assertRaises(ValueError):
            quantize.encode(img, k=9)


class TestDCTCodec(unittest.TestCase):
    def test_high_quality_psnr_above_40(self) -> None:
        """Quality=95 should produce a near-perfect reconstruction."""
        img = _smooth_image()
        out = dct_codec.decode(dct_codec.encode(img, quality=95))
        self.assertEqual(out.shape, img.shape)
        self.assertGreater(metrics.psnr(img, out), 40.0)

    def test_psnr_increases_with_quality(self) -> None:
        img = _smooth_image()
        p_low = metrics.psnr(img, dct_codec.decode(dct_codec.encode(img, quality=10)))
        p_high = metrics.psnr(img, dct_codec.decode(dct_codec.encode(img, quality=90)))
        self.assertLess(p_low, p_high)

    def test_bit_count_increases_with_quality(self) -> None:
        img = _smooth_image()
        n_low = len(dct_codec.encode(img, quality=10))
        n_high = len(dct_codec.encode(img, quality=90))
        self.assertLess(n_low, n_high)

    def test_non_aligned_image_shape(self) -> None:
        img = _random_image((21, 30))
        out = dct_codec.decode(dct_codec.encode(img, quality=70))
        self.assertEqual(out.shape, img.shape)

    def test_zigzag_is_permutation(self) -> None:
        self.assertEqual(sorted(dct_codec.ZIGZAG.tolist()), list(range(64)))
        np.testing.assert_array_equal(
            dct_codec.ZIGZAG[dct_codec.INV_ZIGZAG], np.arange(64)
        )

    def test_dct_matrix_is_orthonormal(self) -> None:
        m = dct_codec.DCT_M
        np.testing.assert_allclose(m.T @ m, np.eye(m.shape[0]), atol=1e-10)

    def test_quality_matrix_scales(self) -> None:
        q_low = dct_codec.quality_to_qmatrix(10)
        q_high = dct_codec.quality_to_qmatrix(90)
        self.assertTrue((q_low >= q_high).all())

    def test_header_roundtrip(self) -> None:
        """The first HEADER_BITS bits must round-trip q, h and w exactly."""
        img = _random_image((40, 24))
        for q in (1, 50, 95):
            bits = dct_codec.encode(img, quality=q)
            b, q_hat, h, w = dct_codec._unpack_header(bits)
            self.assertEqual(q_hat, q)
            self.assertEqual(h, 40)
            self.assertEqual(w, 24)
            self.assertGreaterEqual(b, dct_codec.MIN_COEFF_BITS)
            self.assertLessEqual(b, dct_codec.MAX_COEFF_BITS)


if __name__ == "__main__":
    unittest.main()
