"""End-to-end experiment runner.

* Runs every sample file in ``samples/`` through all three codecs.
* Verifies that decode(encode(x)) == x (lossless guarantee).
* Checks the Huffman bound H <= L_bar < H + 1.
* Writes a CSV summary and four figures (character frequency, code-length
  histogram, bits/symbol comparison, compression ratio comparison) to
  ``report/``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import entropy, huffman, lzw, shannon_fano

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
REPORT = ROOT / "report"
REPORT.mkdir(exist_ok=True)

# Matplotlib: use a font that can render CJK glyphs when available.
for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
    try:
        plt.rcParams["font.sans-serif"] = [candidate]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def _run_huffman(text: str):
    blob, codebook = huffman.encode_to_bytes(text)
    recovered = huffman.decode_from_bytes(blob)
    assert recovered == text
    lengths = huffman.code_lengths(codebook)
    metrics = entropy.summarise(text, lengths)
    metrics["compressed_bytes"] = len(blob)
    metrics["codebook"] = codebook
    return metrics


def _run_shannon_fano(text: str):
    blob, codebook = shannon_fano.encode_to_bytes(text)
    recovered = shannon_fano.decode_from_bytes(blob)
    assert recovered == text
    lengths = {s: len(c) for s, c in codebook.items()}
    metrics = entropy.summarise(text, lengths)
    metrics["compressed_bytes"] = len(blob)
    metrics["codebook"] = codebook
    return metrics


def _run_lzw(text: str):
    blob = lzw.encode_to_bytes(text)
    recovered = lzw.decode_from_bytes(blob)
    assert recovered == text
    probs = entropy.symbol_probs(text)
    h = entropy.entropy(probs)
    l_bar = (len(blob) * 8) / len(text) if text else 0.0
    return {
        "H": h,
        "L_bar": l_bar,
        "efficiency": (h / l_bar) if l_bar > 0 else 0.0,
        "redundancy": 1 - ((h / l_bar) if l_bar > 0 else 0.0),
        "num_symbols": len(probs),
        "length": len(text),
        "compressed_bytes": len(blob),
        "codebook": None,
    }


RUNNERS = {
    "Huffman": _run_huffman,
    "Shannon-Fano": _run_shannon_fano,
    "LZW": _run_lzw,
}


def _fig_char_frequency(text: str, name: str, top_n: int = 25) -> Path:
    counts = entropy.symbol_counts(text)
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    labels = [repr(s) if s in "\n\r\t " else s for s, _ in items]
    freqs = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(freqs)), freqs, color="#3274A1")
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(labels)
    ax.set_title(f"Top-{top_n} character frequency ({name})")
    ax.set_xlabel("Character")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out = REPORT / f"char_freq_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _fig_length_histogram(codebook: Dict[str, str], name: str) -> Path:
    lengths = [len(c) for c in codebook.values()]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=range(min(lengths), max(lengths) + 2), color="#E1812C",
            edgecolor="black")
    ax.set_title(f"Huffman code-length distribution ({name})")
    ax.set_xlabel("Code length (bits)")
    ax.set_ylabel("Number of symbols")
    fig.tight_layout()
    out = REPORT / f"codelen_hist_{name}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _fig_bits_per_symbol(rows: List[dict]) -> Path:
    names = sorted({r["sample"] for r in rows})
    methods = ["Huffman", "Shannon-Fano", "LZW"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.22
    x = range(len(names))
    for i, m in enumerate(methods):
        ys = []
        for n in names:
            match = next(r for r in rows if r["sample"] == n and r["method"] == m)
            ys.append(match["L_bar"])
        ax.bar([xi + (i - 1) * width for xi in x], ys, width=width, label=m)
    h_vals = []
    for n in names:
        match = next(r for r in rows if r["sample"] == n and r["method"] == "Huffman")
        h_vals.append(match["H"])
    ax.plot(list(x), h_vals, "k--o", label="H(X) lower bound")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Average code length (bit / symbol)")
    ax.set_title("Compression cost vs. source entropy")
    ax.legend()
    fig.tight_layout()
    out = REPORT / "bits_per_symbol.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _fig_compression_ratio(rows: List[dict]) -> Path:
    names = sorted({r["sample"] for r in rows})
    methods = ["Huffman", "Shannon-Fano", "LZW"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.25
    x = range(len(names))
    for i, m in enumerate(methods):
        ys = []
        for n in names:
            match = next(r for r in rows if r["sample"] == n and r["method"] == m)
            ys.append(match["CR"])
        ax.bar([xi + (i - 1) * width for xi in x], ys, width=width, label=m)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Compression ratio  (UTF-8 bytes / output bytes)")
    ax.set_title("Overall compression ratio")
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend()
    fig.tight_layout()
    out = REPORT / "compression_ratio.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def run() -> None:
    samples = sorted(p for p in SAMPLES.iterdir() if p.suffix == ".txt")
    if not samples:
        raise SystemExit(f"No .txt samples found in {SAMPLES}")

    rows: List[dict] = []
    for sample in samples:
        text = sample.read_text(encoding="utf-8")
        raw_bytes = len(text.encode("utf-8"))
        name = sample.stem
        print(f"\n=== {name} ===")
        print(f"  chars: {len(text):,}   UTF-8 bytes: {raw_bytes:,}")
        _fig_char_frequency(text, name)

        for method, runner in RUNNERS.items():
            m = runner(text)
            cr = raw_bytes / m["compressed_bytes"] if m["compressed_bytes"] else 0.0
            row = {
                "sample": name,
                "method": method,
                "chars": m["length"],
                "alphabet": m["num_symbols"],
                "H": m["H"],
                "L_bar": m["L_bar"],
                "efficiency": m["efficiency"],
                "redundancy": m["redundancy"],
                "compressed_bytes": m["compressed_bytes"],
                "CR": cr,
            }
            rows.append(row)
            print(
                f"  {method:<13} H={m['H']:.4f}  L_bar={m['L_bar']:.4f}  "
                f"eta={m['efficiency']:.4f}  CR={cr:.3f}  out={m['compressed_bytes']} B"
            )
            if method == "Huffman":
                h = m["H"]
                l_bar = m["L_bar"]
                # Shannon's source-coding theorem for a memoryless source:
                #   H(X) <= L_bar < H(X) + 1
                ok_lower = l_bar + 1e-9 >= h
                ok_upper = l_bar < h + 1.0 + 1e-9
                status = "OK" if (ok_lower and ok_upper) else "FAIL"
                print(
                    f"    Huffman bound H <= L_bar < H+1 : "
                    f"{h:.4f} <= {l_bar:.4f} < {h + 1:.4f}  [{status}]"
                )
                if m["codebook"] is not None:
                    _fig_length_histogram(m["codebook"], name)

    csv_path = REPORT / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {_fig_bits_per_symbol(rows)}")
    print(f"Wrote {_fig_compression_ratio(rows)}")


if __name__ == "__main__":
    run()
