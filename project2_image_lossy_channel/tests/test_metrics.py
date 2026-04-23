"""Tests for the evaluation metrics."""
from __future__ import annotations

import math
import unittest

import numpy as np

from . import _common  # noqa: F401
from src import metrics


class TestMSE(unittest.TestCase):
    def test_zero_for_identical(self) -> None:
        a = np.full((8, 8), 123, dtype=np.uint8)
        self.assertEqual(metrics.mse(a, a.copy()), 0.0)

    def test_matches_manual(self) -> None:
        a = np.array([[0, 0], [0, 0]], dtype=np.uint8)
        b = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        expected = (1 + 4 + 9 + 16) / 4
        self.assertAlmostEqual(metrics.mse(a, b), expected, places=9)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            metrics.mse(np.zeros((4, 4)), np.zeros((4, 5)))


class TestPSNR(unittest.TestCase):
    def test_identical_returns_inf(self) -> None:
        a = np.zeros((4, 4), dtype=np.uint8)
        self.assertEqual(metrics.psnr(a, a.copy()), float("inf"))

    def test_known_value(self) -> None:
        """MSE = 1 should give PSNR = 20·log10(255) ≈ 48.131 dB."""
        a = np.zeros((4, 4), dtype=np.uint8)
        b = a.copy()
        b[0, 0] = 4          # MSE = 16/16 = 1
        self.assertAlmostEqual(metrics.psnr(a, b), 20 * math.log10(255), places=5)


class TestBER(unittest.TestCase):
    def test_identical(self) -> None:
        tx = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        self.assertEqual(metrics.ber(tx, tx.copy()), 0.0)

    def test_all_flipped(self) -> None:
        tx = np.array([0, 1, 0, 1], dtype=np.uint8)
        rx = np.array([1, 0, 1, 0], dtype=np.uint8)
        self.assertEqual(metrics.ber(tx, rx), 1.0)

    def test_partial(self) -> None:
        tx = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
        rx = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.assertAlmostEqual(metrics.ber(tx, rx), 0.2, places=9)

    def test_handles_mismatched_length(self) -> None:
        tx = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        rx = np.array([0, 1, 0], dtype=np.uint8)
        self.assertEqual(metrics.ber(tx, rx), 0.0)


class TestAccuracy(unittest.TestCase):
    def test_exact(self) -> None:
        a = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        b = np.array([[10, 21], [30, 40]], dtype=np.uint8)
        self.assertAlmostEqual(metrics.pixel_accuracy(a, b, tol=0), 0.75)
        self.assertAlmostEqual(metrics.pixel_accuracy(a, b, tol=1), 1.0)


if __name__ == "__main__":
    unittest.main()
