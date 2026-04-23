"""Information-theoretic measurements for a memoryless source.

Provides utilities to compute:
  * empirical symbol probabilities
  * source entropy  H(X) = -sum p_i log2 p_i         (bit/symbol)
  * average code length L_bar = sum p_i * len(c_i)   (bit/symbol)
  * coding efficiency  eta = H(X) / L_bar
  * redundancy         rho = 1 - eta
  * compression ratio  CR  = original_bits / compressed_bits
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Hashable, Mapping, Sequence


def symbol_counts(seq: Sequence[Hashable]) -> Counter:
    """Return a Counter of symbol occurrences."""
    return Counter(seq)


def symbol_probs(seq: Sequence[Hashable]) -> Dict[Hashable, float]:
    """Return empirical probability of each symbol in ``seq``."""
    counts = symbol_counts(seq)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {s: c / total for s, c in counts.items()}


def entropy(probs: Mapping[Hashable, float]) -> float:
    """Shannon entropy (base 2) in bit/symbol."""
    h = 0.0
    for p in probs.values():
        if p > 0.0:
            h -= p * math.log2(p)
    return h


def avg_code_length(
    probs: Mapping[Hashable, float], code_lengths: Mapping[Hashable, int]
) -> float:
    """Average code length sum_i p_i * l_i in bit/symbol."""
    l_bar = 0.0
    for s, p in probs.items():
        if p <= 0.0:
            continue
        if s not in code_lengths:
            raise KeyError(f"symbol {s!r} missing from code_lengths")
        l_bar += p * code_lengths[s]
    return l_bar


def efficiency(h: float, l_bar: float) -> float:
    """Coding efficiency eta = H / L_bar in [0, 1]."""
    if l_bar <= 0.0:
        return 0.0
    return h / l_bar


def redundancy(h: float, l_bar: float) -> float:
    """Redundancy rho = 1 - eta."""
    return 1.0 - efficiency(h, l_bar)


def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """Ratio of original size to compressed size (the larger the better)."""
    if compressed_bits <= 0:
        return float("inf")
    return original_bits / compressed_bits


def summarise(
    seq: Sequence[Hashable], code_lengths: Mapping[Hashable, int]
) -> Dict[str, float]:
    """Convenience helper returning a metric dictionary for ``seq``."""
    probs = symbol_probs(seq)
    h = entropy(probs)
    l_bar = avg_code_length(probs, code_lengths)
    return {
        "H": h,
        "L_bar": l_bar,
        "efficiency": efficiency(h, l_bar),
        "redundancy": redundancy(h, l_bar),
        "num_symbols": len(probs),
        "length": len(seq),
    }
