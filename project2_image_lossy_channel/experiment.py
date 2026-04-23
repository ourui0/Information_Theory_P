"""Full experimental evaluation for Project 2.

Runs four families of sweeps over every sample image under
`samples/*.png` and writes both a CSV of raw results and several PNG plots
into `report/`:

    1.  Rate-distortion : DCT quality ∈ {10..95} with no channel noise.
                           Plots PSNR vs bits-per-pixel.
    2.  Source coder    : DCT vs uniform scalar quantizer at matched bpp.
    3.  BSC sweep       : crossover p ∈ {0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}
                           for {none, repetition(3), repetition(5), hamming(7,4)}.
    4.  BEC sweep       : erasure p ∈ {0, 1e-2, 3e-2, 1e-1, 2e-1, 3e-1}
                           for {none, repetition(3), repetition(5), hamming(7,4)}.

For each configuration a recovered-image PNG is also saved so the final
report can display qualitative results alongside the quantitative ones.
"""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src import pipeline


ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
REPORT = ROOT / "report"
IMAGES = REPORT / "images"


def _load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _save_gray(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _row(image_name: str, experiment: str, rep) -> dict:
    d = {"image": image_name, "experiment": experiment}
    cfg = rep.config
    d.update(
        {
            "source": cfg.source,
            "source_param": cfg.source_param,
            "channel_code": cfg.channel_code,
            "channel_param": cfg.channel_param,
            "channel": cfg.channel,
            "channel_p": cfg.channel_p,
            "psnr_db": rep.psnr_db,
            "mse": rep.mse,
            "accuracy_exact": rep.accuracy_exact,
            "accuracy_tol5": rep.accuracy_tol5,
            "ber_raw": rep.ber_raw,
            "ber_decoded": rep.ber_after_channel_dec,
            "n_info_bits": rep.n_info_bits,
            "n_tx_bits": rep.n_tx_bits,
            "code_rate": rep.code_rate,
            "compression_ratio": rep.compression_ratio,
            "encode_ms": rep.encode_time_s * 1000.0,
            "decode_ms": rep.decode_time_s * 1000.0,
        }
    )
    return d


def sweep_rate_distortion(image: np.ndarray, name: str, rows: list[dict]) -> None:
    for q in [5, 10, 20, 30, 50, 70, 85, 95]:
        cfg = pipeline.Config(
            source="dct", source_param=q,
            channel_code="none", channel="bsc", channel_p=0.0, seed=0,
        )
        rep = pipeline.run(image, cfg)
        rows.append(_row(name, "rd_dct", rep))
        _save_gray(IMAGES / f"{name}__rd_dct_q{q}.png", rep.recovered)

    for k in [1, 2, 3, 4, 5, 6, 8]:
        cfg = pipeline.Config(
            source="quantize", source_param=k,
            channel_code="none", channel="bsc", channel_p=0.0, seed=0,
        )
        rep = pipeline.run(image, cfg)
        rows.append(_row(name, "rd_quantize", rep))
        _save_gray(IMAGES / f"{name}__rd_quant_k{k}.png", rep.recovered)


def sweep_channel(
    image: np.ndarray,
    name: str,
    rows: list[dict],
    channel: str,
    p_list: list[float],
    quality: int = 60,
) -> None:
    channel_codes = [
        ("none", 1),
        ("repetition", 3),
        ("repetition", 5),
        ("hamming", 4),
    ]
    for code, param in channel_codes:
        for p in p_list:
            cfg = pipeline.Config(
                source="dct", source_param=quality,
                channel_code=code, channel_param=param,
                channel=channel, channel_p=p, seed=0,
            )
            rep = pipeline.run(image, cfg)
            rows.append(_row(name, f"{channel}_sweep", rep))
            tag = f"{code}{param}" if code == "repetition" else code
            _save_gray(IMAGES / f"{name}__{channel}_{tag}_p{p}.png", rep.recovered)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def plot_rate_distortion(rows: list[dict]) -> None:
    images = sorted({r["image"] for r in rows if r["experiment"].startswith("rd_")})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, src, title in zip(
        axes, ("dct", "quantize"), ("DCT codec", "Uniform scalar quantizer")
    ):
        for name in images:
            sel = [
                r
                for r in rows
                if r["image"] == name
                and r["source"] == src
                and r["experiment"].startswith("rd_")
            ]
            sel.sort(key=lambda r: r["n_info_bits"])
            bpp = [r["n_info_bits"] / (128 * 128) for r in sel]
            psnr = [r["psnr_db"] for r in sel]
            ax.plot(bpp, psnr, marker="o", label=name)
        ax.set_xlabel("bits per pixel (info)")
        ax.set_title(f"Rate-distortion: {title}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT / "rate_distortion.png", dpi=150)
    plt.close(fig)
    print(f"wrote {REPORT / 'rate_distortion.png'}")


def plot_channel_sweep(rows: list[dict], channel: str) -> None:
    sub = [r for r in rows if r["experiment"] == f"{channel}_sweep"]
    if not sub:
        return
    images = sorted({r["image"] for r in sub})
    codes = [("none", 1), ("repetition", 3), ("repetition", 5), ("hamming", 4)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    ax_psnr, ax_ber = axes

    for name in images:
        for code, param in codes:
            sel = [
                r
                for r in sub
                if r["image"] == name
                and r["channel_code"] == code
                and r["channel_param"] == param
            ]
            sel.sort(key=lambda r: r["channel_p"])
            ps = [r["channel_p"] for r in sel]
            psnr = [r["psnr_db"] for r in sel]
            ber = [max(r["ber_decoded"], 1e-6) for r in sel]
            label = f"{name}/{code}{param}" if code == "repetition" else f"{name}/{code}"
            ax_psnr.plot(ps, psnr, marker="o", label=label, alpha=0.7)
            ax_ber.plot(ps, ber, marker="o", label=label, alpha=0.7)

    for ax in axes:
        ax.set_xlabel("channel parameter p")
        ax.grid(True, which="both", alpha=0.3)
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.set_title(f"{channel.upper()}: PSNR vs p")
    ax_ber.set_yscale("log")
    ax_ber.set_ylabel("BER after channel decoding")
    ax_ber.set_title(f"{channel.upper()}: decoded BER vs p")
    ax_ber.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORT / f"channel_sweep_{channel}.png", dpi=150)
    plt.close(fig)
    print(f"wrote {REPORT / f'channel_sweep_{channel}.png'}")


def plot_complexity(rows: list[dict]) -> None:
    rd = [r for r in rows if r["experiment"].startswith("rd_")]
    if not rd:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for src, marker in (("dct", "o"), ("quantize", "s")):
        sel = [r for r in rd if r["source"] == src]
        xs = [r["n_info_bits"] / (128 * 128) for r in sel]
        ys = [r["encode_ms"] + r["decode_ms"] for r in sel]
        ax.scatter(xs, ys, label=src, marker=marker, alpha=0.6)
    ax.set_xlabel("bits per pixel (info)")
    ax.set_ylabel("encode + decode time (ms)")
    ax.set_title("Algorithm complexity at 128x128")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "complexity.png", dpi=150)
    plt.close(fig)
    print(f"wrote {REPORT / 'complexity.png'}")


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)

    sample_paths = sorted(SAMPLES.glob("*.png"))
    if not sample_paths:
        raise SystemExit(
            "No samples found. Run  python samples/generate_samples.py  first."
        )

    rows: list[dict] = []
    for p in sample_paths:
        name = p.stem
        img = _load_gray(p)
        print(f"\n=== {name} ({img.shape}) ===")
        print("  rate-distortion sweep ...")
        sweep_rate_distortion(img, name, rows)
        print("  BSC sweep ...")
        sweep_channel(img, name, rows, "bsc", [0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
        print("  BEC sweep ...")
        sweep_channel(img, name, rows, "bec", [0.0, 1e-2, 3e-2, 1e-1, 2e-1, 3e-1])

    write_csv(rows, REPORT / "metrics.csv")
    plot_rate_distortion(rows)
    plot_channel_sweep(rows, "bsc")
    plot_channel_sweep(rows, "bec")
    plot_complexity(rows)
    print("\nDone. See report/ directory for all outputs.")


if __name__ == "__main__":
    main()
