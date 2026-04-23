"""Tests for the three codecs and a full sample x method integration sweep.

Uses only the Python standard library (``unittest``).
"""

from __future__ import annotations

import random
import unittest
from pathlib import Path

from src import entropy, huffman, lzw, shannon_fano

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

ROUND_TRIP_INPUTS = [
    ("empty", ""),
    ("single_char", "a"),
    ("single_char_repeated", "aaaaaaaaaa"),
    ("two_symbols", "abababab"),
    ("ascii_sentence", "the quick brown fox jumps over the lazy dog 0123456789"),
    ("unicode_cjk", "信息论项目：无损信源编码的实验验证。"),
    ("mixed", "Shannon 1948 —— H(X) = -Σ p log₂ p  ✓"),
]


def _is_prefix_free(codebook: dict) -> bool:
    codes = sorted(codebook.values(), key=len)
    for i, a in enumerate(codes):
        for b in codes[i + 1 :]:
            if b.startswith(a):
                return False
    return True


def _kraft_sum(codebook: dict) -> float:
    return sum(2 ** (-len(c)) for c in codebook.values())


class HuffmanTests(unittest.TestCase):
    def test_single_symbol_gets_length_one_code(self):
        _, codebook = huffman.encode_to_bytes("aaaaa")
        self.assertEqual(codebook, {"a": "0"})

    def test_codebook_is_prefix_free_and_kraft_tight(self):
        text = "the quick brown fox jumps over the lazy dog"
        _, codebook = huffman.encode_to_bytes(text)
        self.assertTrue(_is_prefix_free(codebook))
        # For a full Huffman tree Kraft's sum equals 1.
        self.assertAlmostEqual(_kraft_sum(codebook), 1.0)

    def test_satisfies_source_coding_bound(self):
        """Shannon's source coding theorem: H(X) <= L_bar < H(X) + 1."""
        data = (SAMPLES_DIR / "english.txt").read_text(encoding="utf-8")
        _, codebook = huffman.encode_to_bytes(data)
        probs = entropy.symbol_probs(data)
        h = entropy.entropy(probs)
        l_bar = entropy.avg_code_length(probs, huffman.code_lengths(codebook))
        self.assertLessEqual(h, l_bar + 1e-9)
        self.assertLess(l_bar, h + 1.0)

    def test_beats_or_ties_shannon_fano(self):
        """Huffman is optimal among prefix codes: L_H <= L_SF."""
        text = (SAMPLES_DIR / "english.txt").read_text(encoding="utf-8")
        probs = entropy.symbol_probs(text)
        _, hc = huffman.encode_to_bytes(text)
        _, sfc = shannon_fano.encode_to_bytes(text)
        l_h = entropy.avg_code_length(probs, huffman.code_lengths(hc))
        l_sf = entropy.avg_code_length(
            probs, {s: len(c) for s, c in sfc.items()}
        )
        self.assertLessEqual(l_h, l_sf + 1e-9)

    def test_rejects_stream_with_wrong_magic(self):
        with self.assertRaises(ValueError):
            huffman.decode_from_bytes(b"NOPE\x00\x00\x00\x00")

    def test_round_trip_inputs(self):
        for name, text in ROUND_TRIP_INPUTS:
            with self.subTest(name=name):
                blob, _ = huffman.encode_to_bytes(text)
                self.assertEqual(huffman.decode_from_bytes(blob), text)


class ShannonFanoTests(unittest.TestCase):
    def test_codebook_is_prefix_free(self):
        text = "the quick brown fox jumps over the lazy dog"
        _, codebook = shannon_fano.encode_to_bytes(text)
        self.assertTrue(_is_prefix_free(codebook))
        self.assertLessEqual(_kraft_sum(codebook), 1.0 + 1e-9)

    def test_rejects_stream_with_wrong_magic(self):
        with self.assertRaises(ValueError):
            shannon_fano.decode_from_bytes(b"NOPE\x00\x00\x00\x00")

    def test_round_trip_inputs(self):
        for name, text in ROUND_TRIP_INPUTS:
            with self.subTest(name=name):
                blob, _ = shannon_fano.encode_to_bytes(text)
                self.assertEqual(shannon_fano.decode_from_bytes(blob), text)


class LZWTests(unittest.TestCase):
    def test_rejects_stream_with_wrong_magic(self):
        with self.assertRaises(ValueError):
            lzw.decode_from_bytes(b"NOPEnope")

    def test_round_trip_inputs(self):
        for name, text in ROUND_TRIP_INPUTS:
            with self.subTest(name=name):
                blob = lzw.encode_to_bytes(text)
                self.assertEqual(lzw.decode_from_bytes(blob), text)

    def test_round_trip_across_width_boundaries(self):
        """Random data stresses the 9->10->11->12->13 code-width transitions."""
        rng = random.Random(1234)
        for n in (600, 2048, 8192, 32768):
            with self.subTest(n=n):
                data = bytes(rng.randrange(256) for _ in range(n))
                blob = lzw.encode_bytes(data)
                self.assertEqual(lzw.decode_bytes(blob), data)

    def test_round_trip_highly_repetitive(self):
        data = ("ABCABCABCDABCDEABCDEFABCDEFG" * 2000).encode("utf-8")
        blob = lzw.encode_bytes(data)
        self.assertEqual(lzw.decode_bytes(blob), data)


class IntegrationTests(unittest.TestCase):
    """Every sample text round-trips through every codec."""

    def _encode_decode_pairs(self):
        return [
            ("huffman",
             lambda t: huffman.encode_to_bytes(t)[0],
             huffman.decode_from_bytes),
            ("shannon_fano",
             lambda t: shannon_fano.encode_to_bytes(t)[0],
             shannon_fano.decode_from_bytes),
            ("lzw", lzw.encode_to_bytes, lzw.decode_from_bytes),
        ]

    def test_all_samples_round_trip(self):
        samples = sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix == ".txt")
        self.assertGreater(len(samples), 0, "no sample files found")
        for sample in samples:
            text = sample.read_text(encoding="utf-8")
            for name, encode, decode in self._encode_decode_pairs():
                with self.subTest(sample=sample.name, codec=name):
                    blob = encode(text)
                    self.assertEqual(
                        decode(blob), text,
                        f"{name} round-trip failed on {sample.name}",
                    )


if __name__ == "__main__":
    unittest.main()
