"""Shannon-Fano coding used as a classical comparison to Huffman.

Algorithm
---------
1. Sort symbols by decreasing probability.
2. Split the sorted list at the index that minimises the absolute difference
   of the two halves' total probability.
3. Prepend '0' to every symbol in the upper half, '1' to the lower half.
4. Recurse on each half until only one symbol remains.

Shannon-Fano is prefix-free but not guaranteed optimal, so its average code
length is >= Huffman's.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .bitio import BitReader, BitWriter

MAGIC = b"SF01"


def _split(items: List[Tuple[str, float]]) -> int:
    """Return the split index that minimises |sum(left) - sum(right)|."""
    total = sum(p for _, p in items)
    best_idx = 1
    best_diff = float("inf")
    running = 0.0
    for i in range(len(items) - 1):
        running += items[i][1]
        diff = abs(running - (total - running))
        if diff < best_diff:
            best_diff = diff
            best_idx = i + 1
    return best_idx


def build_codebook(probs: Dict[str, float]) -> Dict[str, str]:
    if not probs:
        return {}
    if len(probs) == 1:
        sym = next(iter(probs))
        return {sym: "0"}

    items = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
    codes: Dict[str, str] = {s: "" for s, _ in items}

    def recurse(sub: List[Tuple[str, float]]) -> None:
        if len(sub) <= 1:
            return
        idx = _split(sub)
        left, right = sub[:idx], sub[idx:]
        for s, _ in left:
            codes[s] += "0"
        for s, _ in right:
            codes[s] += "1"
        recurse(left)
        recurse(right)

    recurse(items)
    return codes


def encode_to_bytes(text: str) -> Tuple[bytes, Dict[str, str]]:
    if text == "":
        return MAGIC + b"\x00\x00" + b"\x00\x00\x00\x00" + b"\x00\x00\x00\x00", {}

    freqs: Dict[str, int] = {}
    for ch in text:
        freqs[ch] = freqs.get(ch, 0) + 1
    total = sum(freqs.values())
    probs = {s: c / total for s, c in freqs.items()}
    codebook = build_codebook(probs)

    bw = BitWriter()
    for ch in text:
        bw.write_code(codebook[ch])
    payload = bw.to_bytes()
    payload_bits = bw.bit_length

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
    if not blob.startswith(MAGIC):
        raise ValueError("not a SF01 stream")
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

    total = sum(freqs.values())
    probs = {s: c / total for s, c in freqs.items()}
    codebook = build_codebook(probs)
    # Invert into a prefix trie for O(L) decoding.
    lookup: Dict[str, str] = {c: s for s, c in codebook.items()}

    payload = blob[p:]
    reader = BitReader(payload, bit_length=payload_bits)
    out: List[str] = []
    buf = []
    remaining = total_syms
    while remaining > 0:
        buf.append("1" if reader.read_bit() else "0")
        key = "".join(buf)
        if key in lookup:
            out.append(lookup[key])
            buf.clear()
            remaining -= 1
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
