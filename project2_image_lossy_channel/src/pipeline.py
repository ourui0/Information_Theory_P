"""End-to-end image transmission pipeline.

    image ──► source encoder ──► info bits ──► channel encoder ──► tx bits
                                                                      │
                                                               BSC(p) or BEC(p)
                                                                      │
    image ◄── source decoder ◄── info bits ◄── channel decoder ◄── rx bits

Each configuration is a simple named tuple describing which modules to use;
`run()` glues everything together and returns an evaluation report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .source import dct_codec, quantize
from .channel import bec, bsc, hamming, repetition


SOURCE_CODERS = ("dct", "quantize")
CHANNEL_CODERS = ("none", "repetition", "hamming")
CHANNELS = ("bsc", "bec")


@dataclass
class Config:
    source: str = "dct"
    source_param: int = 50           # quality (dct) or bits/pixel (quantize)
    channel_code: str = "hamming"
    channel_param: int = 3           # repetition factor n (only for repetition)
    channel: str = "bsc"
    channel_p: float = 0.01
    seed: int | None = 0


@dataclass
class Report:
    config: Config
    psnr_db: float
    mse: float
    accuracy_exact: float
    accuracy_tol5: float
    ber_raw: float
    ber_after_channel_dec: float
    n_info_bits: int
    n_tx_bits: int
    code_rate: float
    compression_ratio: float
    encode_time_s: float
    decode_time_s: float
    recovered: np.ndarray = field(repr=False)


def _source_encode(image: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.source == "dct":
        return dct_codec.encode(image, quality=cfg.source_param)
    if cfg.source == "quantize":
        return quantize.encode(image, k=cfg.source_param)
    raise ValueError(f"unknown source coder: {cfg.source}")


def _source_decode(bits: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.source == "dct":
        return dct_codec.decode(bits)
    if cfg.source == "quantize":
        return quantize.decode(bits)
    raise ValueError(f"unknown source coder: {cfg.source}")


def _channel_encode(bits: np.ndarray, cfg: Config) -> tuple[np.ndarray, float]:
    if cfg.channel_code == "none":
        return bits, 1.0
    if cfg.channel_code == "repetition":
        return repetition.encode(bits, cfg.channel_param), 1.0 / cfg.channel_param
    if cfg.channel_code == "hamming":
        return hamming.encode(bits), 4.0 / 7.0
    raise ValueError(f"unknown channel coder: {cfg.channel_code}")


def _channel_decode(
    received: np.ndarray,
    n_info_bits: int,
    cfg: Config,
    rng: np.random.Generator,
) -> np.ndarray:
    is_bec = cfg.channel == "bec"
    if cfg.channel_code == "none":
        if is_bec:
            return bec.fill_erasures_random(received, rng=rng)[:n_info_bits]
        return received[:n_info_bits]

    if cfg.channel_code == "repetition":
        n = cfg.channel_param
        if is_bec:
            return repetition.decode_bec(received, n, rng=rng)[:n_info_bits]
        return repetition.decode_bsc(received, n)[:n_info_bits]

    if cfg.channel_code == "hamming":
        if is_bec:
            return hamming.decode_bec(received, n_info_bits, rng=rng)
        return hamming.decode_bsc(received, n_info_bits)

    raise ValueError(f"unknown channel coder: {cfg.channel_code}")


def _transmit(bits: np.ndarray, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    if cfg.channel == "bsc":
        return bsc.transmit(bits, cfg.channel_p, rng=rng)
    if cfg.channel == "bec":
        return bec.transmit(bits, cfg.channel_p, rng=rng)
    raise ValueError(f"unknown channel: {cfg.channel}")


def run(image: np.ndarray, cfg: Config) -> Report:
    """Run the full encode → transmit → decode pipeline and evaluate."""
    rng = np.random.default_rng(cfg.seed)

    t0 = time.perf_counter()
    info_bits = _source_encode(image, cfg)
    tx_bits, code_rate = _channel_encode(info_bits, cfg)
    encode_time = time.perf_counter() - t0

    received = _transmit(tx_bits, cfg, rng)

    t0 = time.perf_counter()
    recovered_info = _channel_decode(received, len(info_bits), cfg, rng)
    recovered = _source_decode(recovered_info, cfg)
    decode_time = time.perf_counter() - t0

    if cfg.channel == "bsc":
        ber_raw = metrics.ber(tx_bits, received)
    else:
        ber_raw = float(np.mean(received == int(bec.ERASURE)))
    ber_corrected = metrics.ber(info_bits, recovered_info)

    if recovered.shape != image.shape:
        h = min(image.shape[0], recovered.shape[0])
        w = min(image.shape[1], recovered.shape[1])
        a = image[:h, :w]
        b = recovered[:h, :w]
    else:
        a, b = image, recovered

    n_input_bits = image.size * 8
    comp_ratio = n_input_bits / max(len(info_bits), 1)

    return Report(
        config=cfg,
        psnr_db=metrics.psnr(a, b),
        mse=metrics.mse(a, b),
        accuracy_exact=metrics.pixel_accuracy(a, b, tol=0),
        accuracy_tol5=metrics.pixel_accuracy(a, b, tol=5),
        ber_raw=ber_raw,
        ber_after_channel_dec=ber_corrected,
        n_info_bits=int(len(info_bits)),
        n_tx_bits=int(len(tx_bits)),
        code_rate=code_rate,
        compression_ratio=comp_ratio,
        encode_time_s=encode_time,
        decode_time_s=decode_time,
        recovered=recovered,
    )
