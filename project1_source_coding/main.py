"""Command-line interface for the lossless-coding project.

Usage examples::

    python main.py encode --method huffman --input samples/english.txt --output out.huf
    python main.py decode --method huffman --input out.huf --output recovered.txt
    python main.py verify --input samples/english.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, Tuple

from src import entropy, huffman, lzw, shannon_fano

METHODS = ("huffman", "shannon_fano", "lzw")


def _huffman_encode(text: str) -> Tuple[bytes, Dict[str, str]]:
    return huffman.encode_to_bytes(text)


def _sf_encode(text: str) -> Tuple[bytes, Dict[str, str]]:
    return shannon_fano.encode_to_bytes(text)


def _lzw_encode(text: str) -> Tuple[bytes, Dict[str, str]]:
    return lzw.encode_to_bytes(text), {}


ENCODERS: Dict[str, Callable[[str], Tuple[bytes, Dict[str, str]]]] = {
    "huffman": _huffman_encode,
    "shannon_fano": _sf_encode,
    "lzw": _lzw_encode,
}

DECODERS: Dict[str, Callable[[bytes], str]] = {
    "huffman": huffman.decode_from_bytes,
    "shannon_fano": shannon_fano.decode_from_bytes,
    "lzw": lzw.decode_from_bytes,
}


def cmd_encode(args: argparse.Namespace) -> int:
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    blob, _ = ENCODERS[args.method](text)
    with open(args.output, "wb") as f:
        f.write(blob)
    print(
        f"[{args.method}] {args.input} ({len(text.encode('utf-8'))} B) "
        f"-> {args.output} ({len(blob)} B)"
    )
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    with open(args.input, "rb") as f:
        blob = f.read()
    text = DECODERS[args.method](blob)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    print(f"[{args.method}] {args.input} -> {args.output} ({len(text)} chars)")
    return 0


def _verify_huffman(text: str) -> Dict[str, float]:
    blob, codebook = huffman.encode_to_bytes(text)
    recovered = huffman.decode_from_bytes(blob)
    assert recovered == text, "Huffman round-trip failed"
    metrics = entropy.summarise(text, huffman.code_lengths(codebook))
    metrics["compressed_bytes"] = len(blob)
    metrics["payload_bits_per_symbol"] = metrics["L_bar"]
    return metrics


def _verify_shannon_fano(text: str) -> Dict[str, float]:
    blob, codebook = shannon_fano.encode_to_bytes(text)
    recovered = shannon_fano.decode_from_bytes(blob)
    assert recovered == text, "Shannon-Fano round-trip failed"
    metrics = entropy.summarise(text, {s: len(c) for s, c in codebook.items()})
    metrics["compressed_bytes"] = len(blob)
    metrics["payload_bits_per_symbol"] = metrics["L_bar"]
    return metrics


def _verify_lzw(text: str) -> Dict[str, float]:
    blob = lzw.encode_to_bytes(text)
    recovered = lzw.decode_from_bytes(blob)
    assert recovered == text, "LZW round-trip failed"
    probs = entropy.symbol_probs(text)
    h = entropy.entropy(probs)
    raw_bits = len(text.encode("utf-8")) * 8
    comp_bits = len(blob) * 8
    l_bar = comp_bits / len(text) if text else 0.0
    return {
        "H": h,
        "L_bar": l_bar,
        "efficiency": (h / l_bar) if l_bar > 0 else 0.0,
        "redundancy": 1 - ((h / l_bar) if l_bar > 0 else 0.0),
        "num_symbols": len(probs),
        "length": len(text),
        "compressed_bytes": len(blob),
        "payload_bits_per_symbol": l_bar,
        "raw_bits": raw_bits,
    }


VERIFIERS = {
    "huffman": _verify_huffman,
    "shannon_fano": _verify_shannon_fano,
    "lzw": _verify_lzw,
}


def cmd_verify(args: argparse.Namespace) -> int:
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    raw_bytes = len(text.encode("utf-8"))
    print(f"Source: {args.input}")
    print(f"  characters: {len(text):,}   UTF-8 bytes: {raw_bytes:,}")

    header = (
        f"{'method':<14}{'H (b/sym)':>12}{'L_bar':>10}{'eta':>10}"
        f"{'rho':>10}{'CR (UTF8)':>12}{'out bytes':>12}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for method in METHODS:
        m = VERIFIERS[method](text)
        cr = (raw_bytes / m["compressed_bytes"]) if m["compressed_bytes"] else 0.0
        print(
            f"{method:<14}{m['H']:>12.4f}{m['L_bar']:>10.4f}"
            f"{m['efficiency']:>10.4f}{m['redundancy']:>10.4f}"
            f"{cr:>12.3f}{m['compressed_bytes']:>12d}"
        )
    print("-" * len(header))
    print("(round-trip decoding passed for every method)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Project 1: lossless source coding for text.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("encode", help="Compress a text file.")
    pe.add_argument("--method", choices=METHODS, required=True)
    pe.add_argument("--input", required=True)
    pe.add_argument("--output", required=True)
    pe.set_defaults(func=cmd_encode)

    pd = sub.add_parser("decode", help="Decompress a previously encoded file.")
    pd.add_argument("--method", choices=METHODS, required=True)
    pd.add_argument("--input", required=True)
    pd.add_argument("--output", required=True)
    pd.set_defaults(func=cmd_decode)

    pv = sub.add_parser("verify", help="Run all methods and print metrics.")
    pv.add_argument("--input", required=True)
    pv.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
