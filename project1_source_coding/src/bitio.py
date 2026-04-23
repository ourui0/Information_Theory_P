"""Minimal MSB-first bit writer / reader used to pack variable-length codes."""

from __future__ import annotations

from typing import Iterable, List


class BitWriter:
    """Accumulates individual bits and emits a ``bytes`` payload.

    The last byte is zero-padded on the right (LSB side). ``bit_length``
    stores the exact number of valid bits so the reader can stop on time.
    """

    __slots__ = ("_buf", "_cur", "_nbits", "bit_length")

    def __init__(self) -> None:
        self._buf: List[int] = []
        self._cur = 0
        self._nbits = 0
        self.bit_length = 0

    def write_bit(self, bit: int) -> None:
        self._cur = (self._cur << 1) | (bit & 1)
        self._nbits += 1
        self.bit_length += 1
        if self._nbits == 8:
            self._buf.append(self._cur)
            self._cur = 0
            self._nbits = 0

    def write_bits(self, bits: Iterable[int]) -> None:
        for b in bits:
            self.write_bit(b)

    def write_code(self, code: str) -> None:
        """Write a code given as a string of '0'/'1'."""
        for ch in code:
            self.write_bit(1 if ch == "1" else 0)

    def write_uint(self, value: int, width: int) -> None:
        """Write ``value`` as a ``width``-bit unsigned integer (MSB first)."""
        if value < 0 or value >= (1 << width):
            raise ValueError(f"value {value} does not fit in {width} bits")
        for i in range(width - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def to_bytes(self) -> bytes:
        """Flush and return the accumulated payload."""
        if self._nbits > 0:
            pad = 8 - self._nbits
            self._buf.append(self._cur << pad)
            self._cur = 0
            self._nbits = 0
        return bytes(self._buf)


class BitReader:
    """MSB-first bit reader matching :class:`BitWriter`."""

    __slots__ = ("_data", "_bit_length", "_pos")

    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        self._data = data
        self._bit_length = bit_length if bit_length is not None else len(data) * 8
        self._pos = 0

    @property
    def position(self) -> int:
        return self._pos

    def eof(self) -> bool:
        return self._pos >= self._bit_length

    def read_bit(self) -> int:
        if self._pos >= self._bit_length:
            raise EOFError("no more bits")
        byte = self._data[self._pos >> 3]
        bit = (byte >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit

    def read_uint(self, width: int) -> int:
        v = 0
        for _ in range(width):
            v = (v << 1) | self.read_bit()
        return v
