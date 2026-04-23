"""Unit tests for the information-theoretic helpers and bit I/O.

Uses only the Python standard library (``unittest``).
"""

from __future__ import annotations

import math
import random
import unittest

from src import bitio, entropy


class EntropyTests(unittest.TestCase):
    def test_entropy_of_empty_is_zero(self):
        self.assertEqual(entropy.entropy({}), 0.0)

    def test_entropy_of_deterministic_source_is_zero(self):
        self.assertEqual(entropy.entropy({"a": 1.0}), 0.0)

    def test_entropy_of_fair_coin_is_one_bit(self):
        self.assertAlmostEqual(entropy.entropy({"H": 0.5, "T": 0.5}), 1.0)

    def test_entropy_of_uniform_alphabet_is_log2_n(self):
        n = 8
        probs = {i: 1 / n for i in range(n)}
        self.assertAlmostEqual(entropy.entropy(probs), math.log2(n))

    def test_symbol_probs_sum_to_one(self):
        text = "abracadabra"
        probs = entropy.symbol_probs(text)
        self.assertAlmostEqual(sum(probs.values()), 1.0)
        counts = entropy.symbol_counts(text)
        self.assertEqual(counts["a"], 5)
        self.assertEqual(counts["b"], 2)
        self.assertEqual(counts["r"], 2)

    def test_avg_code_length_and_efficiency(self):
        # Source: a b c with probabilities 1/2, 1/4, 1/4. H = 1.5 bit/sym.
        probs = {"a": 0.5, "b": 0.25, "c": 0.25}
        code_lengths = {"a": 1, "b": 2, "c": 2}  # optimal Huffman code
        l_bar = entropy.avg_code_length(probs, code_lengths)
        h = entropy.entropy(probs)
        self.assertAlmostEqual(l_bar, 1.5)
        self.assertAlmostEqual(h, 1.5)
        self.assertAlmostEqual(entropy.efficiency(h, l_bar), 1.0)
        self.assertAlmostEqual(entropy.redundancy(h, l_bar), 0.0)

    def test_compression_ratio_edge_cases(self):
        self.assertEqual(entropy.compression_ratio(1000, 500), 2.0)
        self.assertTrue(math.isinf(entropy.compression_ratio(100, 0)))


class BitIOTests(unittest.TestCase):
    def test_bitwriter_pads_last_byte_with_zeros(self):
        bw = bitio.BitWriter()
        bw.write_bits([1, 0, 1])
        self.assertEqual(bw.to_bytes(), bytes([0b10100000]))
        self.assertEqual(bw.bit_length, 3)

    def test_bit_round_trip_random(self):
        rng = random.Random(42)
        bits = [rng.randint(0, 1) for _ in range(1000)]
        bw = bitio.BitWriter()
        for b in bits:
            bw.write_bit(b)
        reader = bitio.BitReader(bw.to_bytes(), bit_length=bw.bit_length)
        recovered = [reader.read_bit() for _ in range(len(bits))]
        self.assertEqual(recovered, bits)
        self.assertTrue(reader.eof())

    def test_uint_round_trip_across_widths(self):
        bw = bitio.BitWriter()
        values = [(0, 1), (1, 1), (5, 3), (255, 8), (0x1234, 16), (0xDEADBEEF, 32)]
        for v, w in values:
            bw.write_uint(v, w)
        reader = bitio.BitReader(bw.to_bytes(), bit_length=bw.bit_length)
        for v, w in values:
            self.assertEqual(reader.read_uint(w), v)

    def test_write_code_from_string(self):
        bw = bitio.BitWriter()
        bw.write_code("110100")
        reader = bitio.BitReader(bw.to_bytes(), bit_length=bw.bit_length)
        self.assertEqual(
            [reader.read_bit() for _ in range(6)], [1, 1, 0, 1, 0, 0]
        )

    def test_write_uint_rejects_overflow(self):
        bw = bitio.BitWriter()
        with self.assertRaises(ValueError):
            bw.write_uint(256, 8)

    def test_reader_raises_on_eof(self):
        reader = bitio.BitReader(b"", bit_length=0)
        with self.assertRaises(EOFError):
            reader.read_bit()


if __name__ == "__main__":
    unittest.main()
