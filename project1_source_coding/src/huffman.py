"""Character-level Huffman coding with a self-contained file format.

Algorithm
---------
* Count character frequencies of the source string.
* Build a Huffman tree by repeatedly merging the two lowest-frequency nodes
  (priority queue tie-breaking keeps the algorithm deterministic).
* Derive a prefix-free code by traversing the tree (left=0, right=1).
* Special case: a single-symbol source is assigned the 1-bit code "0" so the
  encoder/decoder still produce a positive number of bits.

File format (self-describing, produced by :func:`encode_to_bytes`):
    magic        : 4 bytes  b"HUF1"
    num_syms     : 2 bytes  big-endian, number of distinct symbols (>=1)
    total_syms   : 4 bytes  big-endian, number of symbols in the source
    payload_bits : 4 bytes  big-endian, valid bit length of the payload
    table        : for each distinct symbol
                     2 bytes  UTF-8 length of symbol (L)
                     L bytes  UTF-8 encoded symbol
                     4 bytes  big-endian frequency
    payload      : bit-packed Huffman codes, MSB first (padding on the right)
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .bitio import BitReader, BitWriter

MAGIC = b"HUF1"


@dataclass
class _Node:
    freq: int
    symbol: Optional[str] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def build_tree(freqs: Dict[str, int]) -> _Node:
    """Build a Huffman tree from a non-empty frequency table."""
    if not freqs:
        raise ValueError("empty frequency table")

    counter = itertools.count()
    heap: List[Tuple[int, int, _Node]] = []
    for sym, f in freqs.items():
        heapq.heappush(heap, (f, next(counter), _Node(freq=f, symbol=sym)))

    if len(heap) == 1:
        # Degenerate source: wrap the single leaf so it still has depth 1.
        f, _, leaf = heap[0]
        return _Node(freq=f, left=leaf, right=None)

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)
        merged = _Node(freq=f1 + f2, left=n1, right=n2)
        heapq.heappush(heap, (merged.freq, next(counter), merged))

    return heap[0][2]


def build_codebook(tree: _Node) -> Dict[str, str]:
    """Return symbol -> bitstring mapping for a Huffman tree."""
    codes: Dict[str, str] = {}

    def walk(node: _Node, prefix: str) -> None:
        if node.is_leaf:
            assert node.symbol is not None
            codes[node.symbol] = prefix or "0"
            return
        if node.left is not None:
            walk(node.left, prefix + "0")
        if node.right is not None:
            walk(node.right, prefix + "1")

    walk(tree, "")
    return codes


def code_lengths(codebook: Dict[str, str]) -> Dict[str, int]:
    return {s: len(c) for s, c in codebook.items()}


# ---------------------------------------------------------------------------
# Encoding / decoding against an in-memory bytes object
# ---------------------------------------------------------------------------


def encode_to_bytes(text: str) -> Tuple[bytes, Dict[str, str]]:
    """Compress ``text`` to a self-contained bytes blob.

    Returns the payload plus the codebook (so callers can inspect it without
    re-parsing the file header).
    """
    if text == "":
        # Empty source: num_syms=0, total_syms=0, payload_bits=0.
        return MAGIC + b"\x00\x00" + b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00", {}

    freqs: Dict[str, int] = {}
    for ch in text:
        freqs[ch] = freqs.get(ch, 0) + 1

    tree = build_tree(freqs)
    codebook = build_codebook(tree)

    body = BitWriter()
    for ch in text:
        body.write_code(codebook[ch])
    payload = body.to_bytes()
    payload_bits = body.bit_length

    out = bytearray()
    out += MAGIC
    out += len(freqs).to_bytes(2, "big")
    out += len(text).to_bytes(4, "big")
    out += payload_bits.to_bytes(4, "big")
    for sym, f in freqs.items():
        encoded = sym.encode("utf-8")
        out += len(encoded).to_bytes(2, "big")
        out += encoded
        out += f.to_bytes(4, "big")
    out += payload
    return bytes(out), codebook


def decode_from_bytes(blob: bytes) -> str:
    """Inverse of :func:`encode_to_bytes`. Raises ``ValueError`` on bad input."""
    if not blob.startswith(MAGIC):
        raise ValueError("not a HUF1 stream")
    p = len(MAGIC)
    num_syms = int.from_bytes(blob[p : p + 2], "big"); p += 2
    total_syms = int.from_bytes(blob[p : p + 4], "big"); p += 4
    payload_bits = int.from_bytes(blob[p : p + 4], "big"); p += 4

    if num_syms == 0:
        return ""

    freqs: Dict[str, int] = {}
    for _ in range(num_syms):
        slen = int.from_bytes(blob[p : p + 2], "big"); p += 2
        sym = blob[p : p + slen].decode("utf-8"); p += slen
        f = int.from_bytes(blob[p : p + 4], "big"); p += 4
        freqs[sym] = f

    tree = build_tree(freqs)
    payload = blob[p:]
    reader = BitReader(payload, bit_length=payload_bits)

    out: List[str] = []
    for _ in range(total_syms):
        node = tree
        while not node.is_leaf:
            bit = reader.read_bit()
            node = node.left if bit == 0 else (node.right or node.left)
            assert node is not None
        assert node.symbol is not None
        out.append(node.symbol)
    return "".join(out)


def encode_file(src_path: str, dst_path: str) -> Dict[str, str]:
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    blob, codebook = encode_to_bytes(text)
    with open(dst_path, "wb") as f:
        f.write(blob)
    return codebook


def decode_file(src_path: str, dst_path: str) -> None:
    with open(src_path, "rb") as f:
        blob = f.read()
    text = decode_from_bytes(blob)
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
