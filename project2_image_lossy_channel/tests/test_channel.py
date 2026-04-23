"""Tests for channels (BSC, BEC) and channel codes (repetition, Hamming)."""
from __future__ import annotations

import unittest

import numpy as np

from . import _common  # noqa: F401
from src import metrics
from src.channel import bec, bsc, hamming, repetition


class TestBSC(unittest.TestCase):
    def test_p0_is_transparent(self) -> None:
        bits = np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        rx = bsc.transmit(bits, p=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(rx, bits)

    def test_p1_flips_every_bit(self) -> None:
        bits = np.array([0, 1, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        rx = bsc.transmit(bits, p=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(rx, 1 - bits)

    def test_empirical_ber_close_to_p(self) -> None:
        rng = np.random.default_rng(1)
        bits = np.zeros(200_000, dtype=np.uint8)
        rx = bsc.transmit(bits, p=0.07, rng=rng)
        self.assertAlmostEqual(float(rx.mean()), 0.07, places=2)

    def test_invalid_probability(self) -> None:
        with self.assertRaises(ValueError):
            bsc.transmit(np.array([0]), p=-0.1)
        with self.assertRaises(ValueError):
            bsc.transmit(np.array([0]), p=1.1)


class TestBEC(unittest.TestCase):
    def test_p0_is_transparent(self) -> None:
        bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        rx = bec.transmit(bits, p=0.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(rx, bits)

    def test_p1_erases_every_bit(self) -> None:
        bits = np.array([0, 1, 0, 1, 1], dtype=np.uint8)
        rx = bec.transmit(bits, p=1.0, rng=np.random.default_rng(0))
        self.assertTrue((rx == int(bec.ERASURE)).all())

    def test_empirical_erasure_rate(self) -> None:
        rng = np.random.default_rng(2)
        bits = np.zeros(200_000, dtype=np.uint8)
        rx = bec.transmit(bits, p=0.2, rng=rng)
        self.assertAlmostEqual(
            float(np.mean(rx == int(bec.ERASURE))), 0.2, places=2
        )

    def test_non_erased_match_input(self) -> None:
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, size=10_000, dtype=np.uint8)
        rx = bec.transmit(bits, p=0.3, rng=rng)
        mask = rx != int(bec.ERASURE)
        np.testing.assert_array_equal(rx[mask], bits[mask])

    def test_fill_erasures_random(self) -> None:
        rx = np.array([0, 2, 1, 2, 2], dtype=np.uint8)
        out = bec.fill_erasures_random(rx, rng=np.random.default_rng(0))
        self.assertEqual(int(out[0]), 0)
        self.assertEqual(int(out[2]), 1)
        self.assertTrue(((out == 0) | (out == 1)).all())


class TestRepetition(unittest.TestCase):
    def test_encode_shape(self) -> None:
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        self.assertEqual(len(repetition.encode(bits, 3)), 12)
        self.assertEqual(len(repetition.encode(bits, 5)), 20)

    def test_encode_rejects_even(self) -> None:
        with self.assertRaises(ValueError):
            repetition.encode(np.array([0, 1]), n=2)

    def test_bsc_decode_zero_noise(self) -> None:
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=1000, dtype=np.uint8)
        for n in (3, 5, 7):
            codeword = repetition.encode(bits, n)
            decoded = repetition.decode_bsc(codeword, n)
            np.testing.assert_array_equal(decoded, bits)

    def test_bsc_corrects_minority_flips(self) -> None:
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=5000, dtype=np.uint8)
        codeword = repetition.encode(bits, 5)
        noisy = bsc.transmit(codeword, p=0.1, rng=rng)
        decoded = repetition.decode_bsc(noisy, 5)
        raw_ber = metrics.ber(codeword, noisy)
        dec_ber = metrics.ber(bits, decoded)
        self.assertGreater(raw_ber, dec_ber * 5)   # substantial coding gain

    def test_bec_recovers_partial_erasures(self) -> None:
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=2000, dtype=np.uint8)
        codeword = repetition.encode(bits, 5)
        noisy = bec.transmit(codeword, p=0.3, rng=rng)
        decoded = repetition.decode_bec(noisy, 5, rng=rng)
        self.assertLess(metrics.ber(bits, decoded), 1e-2)

    def test_bec_decode_all_erased_is_random(self) -> None:
        """When every copy of a bit is erased, output must still be in {0,1}."""
        erased = np.full(15, int(bec.ERASURE), dtype=np.uint8)
        decoded = repetition.decode_bec(erased, n=5, rng=np.random.default_rng(0))
        self.assertEqual(len(decoded), 3)
        self.assertTrue(((decoded == 0) | (decoded == 1)).all())


class TestHamming(unittest.TestCase):
    def test_generator_parity_orthogonality(self) -> None:
        """H · Gᵀ must equal the zero matrix over GF(2)."""
        prod = (hamming.H @ hamming.G.T) % 2
        np.testing.assert_array_equal(prod, np.zeros_like(prod))

    def test_encode_shape_and_rate(self) -> None:
        bits = np.zeros(12, dtype=np.uint8)
        code = hamming.encode(bits)
        self.assertEqual(len(code), 21)          # 3 codewords × 7 bits
        self.assertEqual(len(code) // 7, len(bits) // 4)

    def test_noiseless_roundtrip(self) -> None:
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=400, dtype=np.uint8)
        code = hamming.encode(bits)
        decoded = hamming.decode_bsc(code, n_info_bits=len(bits))
        np.testing.assert_array_equal(decoded, bits)

    def test_corrects_exactly_one_flip_per_word(self) -> None:
        """Flipping a single bit in each 7-bit codeword must be corrected."""
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, size=400, dtype=np.uint8)
        code = hamming.encode(bits).reshape(-1, 7)
        for i in range(code.shape[0]):
            pos = rng.integers(0, 7)
            code[i, pos] ^= 1
        decoded = hamming.decode_bsc(code.reshape(-1), n_info_bits=len(bits))
        np.testing.assert_array_equal(decoded, bits)

    def test_coding_gain_on_bsc(self) -> None:
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, size=4000, dtype=np.uint8)
        code = hamming.encode(bits)
        noisy = bsc.transmit(code, p=0.01, rng=rng)
        decoded = hamming.decode_bsc(noisy, n_info_bits=len(bits))
        raw = metrics.ber(code, noisy)
        dec = metrics.ber(bits, decoded)
        self.assertLess(dec, raw / 3)            # expect ≥ 3× BER reduction

    def test_recovers_two_erasures_per_word(self) -> None:
        """Any 2 erasures in a 7-bit codeword must be perfectly recovered."""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, size=400, dtype=np.uint8)
        code = hamming.encode(bits).reshape(-1, 7).astype(np.int16)
        for i in range(code.shape[0]):
            positions = rng.choice(7, size=2, replace=False)
            code[i, positions] = int(bec.ERASURE)
        decoded = hamming.decode_bec(
            code.reshape(-1), n_info_bits=len(bits), rng=rng
        )
        np.testing.assert_array_equal(decoded, bits)


if __name__ == "__main__":
    unittest.main()
