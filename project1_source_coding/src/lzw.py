"""Variable-width LZW coding of a UTF-8 byte stream.

LZW is a dictionary method: it exploits inter-symbol dependencies (i.e. the
source memory) that a memoryless Huffman coder cannot. The dictionary is
initialised with all 256 single bytes. New phrases are added as they appear
and the code width grows with the dictionary (9 .. ``MAX_BITS`` bits).

Stream format::
    magic       : 4 bytes  b"LZW1"
    payload_bits: 4 bytes  big-endian number of valid payload bits
    payload     : variable-width LZW codes (MSB first, right-zero padded)
"""

from __future__ import annotations

from typing import Dict, List

from .bitio import BitReader, BitWriter

MAGIC = b"LZW1"
INITIAL_BITS = 9
MAX_BITS = 16
MAX_DICT = 1 << MAX_BITS


def _required_bits(dict_size: int) -> int:
    bits = INITIAL_BITS
    while (1 << bits) < dict_size + 1 and bits < MAX_BITS:
        bits += 1
    return bits


def encode_bytes(data: bytes) -> bytes:
    """Variable-width LZW compression.

    The encoder widens the code after it has added a new phrase to the
    dictionary. To keep the decoder in sync despite its natural one-step
    lag, the decoder widens one phrase earlier (see :func:`decode_bytes`).
    """
    bw = BitWriter()
    if not data:
        return MAGIC + b"\x00\x00\x00\x00"

    table: Dict[bytes, int] = {bytes([i]): i for i in range(256)}
    w = b""
    bits = INITIAL_BITS
    for byte in data:
        wc = w + bytes([byte])
        if wc in table:
            w = wc
        else:
            bw.write_uint(table[w], bits)
            if len(table) < MAX_DICT:
                table[wc] = len(table)
                bits = _required_bits(len(table))
            w = bytes([byte])
    if w:
        bw.write_uint(table[w], bits)

    payload = bw.to_bytes()
    payload_bits = bw.bit_length
    return MAGIC + payload_bits.to_bytes(4, "big") + payload


def decode_bytes(blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("not an LZW1 stream")
    payload_bits = int.from_bytes(blob[4:8], "big")
    if payload_bits == 0:
        return b""

    reader = BitReader(blob[8:], bit_length=payload_bits)
    table: List[bytes] = [bytes([i]) for i in range(256)]
    bits = INITIAL_BITS

    first = reader.read_uint(bits)
    w = table[first]
    out = bytearray(w)
    # The decoder lags one phrase behind the encoder, so its widen rule
    # anticipates the next insertion: use len(table) + 1 when picking the
    # width for the NEXT code to be read.
    bits = _required_bits(len(table) + 1)

    while not reader.eof():
        try:
            k = reader.read_uint(bits)
        except EOFError:
            break
        if k < len(table):
            entry = table[k]
        elif k == len(table):
            entry = w + w[:1]
        else:
            raise ValueError(f"invalid LZW code {k}")
        out.extend(entry)
        if len(table) < MAX_DICT:
            table.append(w + entry[:1])
            bits = _required_bits(len(table) + 1)
        w = entry
    return bytes(out)


def encode_to_bytes(text: str) -> bytes:
    return encode_bytes(text.encode("utf-8"))


def decode_from_bytes(blob: bytes) -> str:
    return decode_bytes(blob).decode("utf-8")


def encode_file(src_path: str, dst_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()
    with open(dst_path, "wb") as f:
        f.write(encode_to_bytes(text))


def decode_file(src_path: str, dst_path: str) -> None:
    with open(src_path, "rb") as f:
        blob = f.read()
    with open(dst_path, "w", encoding="utf-8", newline="") as f:
        f.write(decode_from_bytes(blob))
