"""Command-line entry point for Project 2.

Examples
--------
# Round-trip a single image through a DCT source coder, Hamming(7,4) and a
# BSC with flip probability 1%:
python main.py run --input samples/lena.png --source dct --source-param 50 \
                   --channel-code hamming --channel bsc --channel-p 0.01 \
                   --output report/lena_out.png

# Same image through repetition(5) over a BEC with erasure probability 10%:
python main.py run --input samples/cameraman.png --source dct --source-param 60 \
                   --channel-code repetition --channel-param 5 \
                   --channel bec --channel-p 0.1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

from src import pipeline


def _load_gray(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def _save_gray(path: str, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def _print_report(rep: pipeline.Report) -> None:
    c = rep.config
    print("Configuration:")
    print(f"  source         = {c.source}(param={c.source_param})")
    print(f"  channel coder  = {c.channel_code}(param={c.channel_param})")
    print(f"  channel        = {c.channel}(p={c.channel_p})")
    print("Bit counts:")
    print(f"  info bits      = {rep.n_info_bits}")
    print(f"  tx bits        = {rep.n_tx_bits}")
    print(f"  code rate      = {rep.code_rate:.4f}")
    print(f"  compression CR = {rep.compression_ratio:.3f}  (input bits / info bits)")
    print("Quality:")
    print(f"  PSNR           = {rep.psnr_db:.2f} dB")
    print(f"  MSE            = {rep.mse:.3f}")
    print(f"  accuracy @0    = {rep.accuracy_exact:.4f}")
    print(f"  accuracy @±5   = {rep.accuracy_tol5:.4f}")
    print("Channel:")
    print(f"  BER raw        = {rep.ber_raw:.4e}")
    print(f"  BER decoded    = {rep.ber_after_channel_dec:.4e}")
    print("Complexity:")
    print(f"  encode time    = {rep.encode_time_s*1000:.2f} ms")
    print(f"  decode time    = {rep.decode_time_s*1000:.2f} ms")


def cmd_run(args: argparse.Namespace) -> None:
    image = _load_gray(args.input)
    cfg = pipeline.Config(
        source=args.source,
        source_param=args.source_param,
        channel_code=args.channel_code,
        channel_param=args.channel_param,
        channel=args.channel,
        channel_p=args.channel_p,
        seed=args.seed,
    )
    rep = pipeline.run(image, cfg)
    _print_report(rep)
    if args.output:
        _save_gray(args.output, rep.recovered)
        print(f"Recovered image written to {args.output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Project 2 pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="encode, transmit, decode an image")
    run_p.add_argument("--input", required=True)
    run_p.add_argument("--output", default=None)
    run_p.add_argument("--source", choices=pipeline.SOURCE_CODERS, default="dct")
    run_p.add_argument(
        "--source-param",
        type=int,
        default=50,
        help="DCT quality (1..99) or quantize bits/pixel (1..8)",
    )
    run_p.add_argument(
        "--channel-code", choices=pipeline.CHANNEL_CODERS, default="hamming"
    )
    run_p.add_argument(
        "--channel-param",
        type=int,
        default=3,
        help="repetition factor n (only for --channel-code=repetition, must be odd)",
    )
    run_p.add_argument("--channel", choices=pipeline.CHANNELS, default="bsc")
    run_p.add_argument("--channel-p", type=float, default=0.01)
    run_p.add_argument("--seed", type=int, default=0)
    run_p.set_defaults(func=cmd_run)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
