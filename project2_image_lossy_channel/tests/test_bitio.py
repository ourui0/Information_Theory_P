"""Bit-level packing / unpacking tests."""
from __future__ import annotations

import unittest

import numpy as np

from . import _common  # noqa: F401  – ensures project root is importable
from src import bitio


class TestBytesBits(unittest.TestCase):
    def test_roundtrip_random_bytes(self) -> None:
        rng = np.random.default_rng(0)
        for n in (1, 7, 8, 9, 64, 255, 1024):
            data = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
            bits = bitio.bytes_to_bits(data)
            self.assertEqual(len(bits), 8 * n)
            self.assertTrue(((bits == 0) | (bits == 1)).all())
            got = bitio.bits_to_bytes(bits)
            self.assertEqual(got, data)

    def test_bits_to_bytes_pads_with_zero(self) -> None:
        """Short tails are zero-padded on the right (MSB first convention)."""
        bits = np.array([1, 0, 1], dtype=np.uint8)
        got = bitio.bits_to_bytes(bits)
        self.assertEqual(got, bytes([0b10100000]))


class TestIntsBits(unittest.TestCase):
    def test_signed_roundtrip_known(self) -> None:
        values = np.array([0, 1, -1, 2, -2, 127, -128], dtype=np.int64)
        bits = bitio.ints_to_bits(values, 8)
        self.assertEqual(len(bits), 8 * 7)
        back = bitio.bits_to_ints(bits, 8)
        np.testing.assert_array_equal(back, values)

    def test_signed_roundtrip_random(self) -> None:
        rng = np.random.default_rng(42)
        for b in (2, 4, 6, 8, 12, 16):
            lo, hi = -(1 << (b - 1)), (1 << (b - 1)) - 1
            vals = rng.integers(lo, hi + 1, size=500, dtype=np.int64)
            back = bitio.bits_to_ints(bitio.ints_to_bits(vals, b), b)
            np.testing.assert_array_equal(back, vals)

    def test_out_of_range_is_clipped(self) -> None:
        vals = np.array([1000, -1000, 42], dtype=np.int64)
        back = bitio.bits_to_ints(bitio.ints_to_bits(vals, 8), 8)
        self.assertEqual(back[0], 127)
        self.assertEqual(back[1], -128)
        self.assertEqual(back[2], 42)

    def test_bits_per_value_validation(self) -> None:
        with self.assertRaises(ValueError):
            bitio.ints_to_bits(np.array([0]), 0)
        with self.assertRaises(ValueError):
            bitio.ints_to_bits(np.array([0]), 33)


class TestPadToMultiple(unittest.TestCase):
    def test_no_padding_when_already_aligned(self) -> None:
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        out, pad = bitio.pad_to_multiple(bits, 4)
        self.assertEqual(pad, 0)
        np.testing.assert_array_equal(out, bits)

    def test_padding_length(self) -> None:
        bits = np.array([1, 0, 1], dtype=np.uint8)
        out, pad = bitio.pad_to_multiple(bits, 4)
        self.assertEqual(pad, 1)
        self.assertEqual(len(out), 4)
        self.assertEqual(int(out[-1]), 0)


if __name__ == "__main__":
    unittest.main()
