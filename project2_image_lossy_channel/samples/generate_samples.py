"""Generate a small set of synthetic grayscale test images.

Run:  python samples/generate_samples.py
Outputs are written next to this file as 128x128 uint8 PNGs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 128
OUT = Path(__file__).parent


def _save(name: str, arr: np.ndarray) -> None:
    img = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(OUT / f"{name}.png")
    print(f"  wrote samples/{name}.png   shape={img.shape}, mean={img.mean():.1f}")


def gradient() -> np.ndarray:
    x = np.linspace(0, 255, SIZE)
    y = np.linspace(0, 255, SIZE)
    xx, yy = np.meshgrid(x, y)
    return 0.5 * xx + 0.5 * yy


def checkerboard(cell: int = 16) -> np.ndarray:
    yy, xx = np.indices((SIZE, SIZE))
    return np.where(((xx // cell) + (yy // cell)) % 2 == 0, 30.0, 220.0)


def sinusoid() -> np.ndarray:
    x = np.linspace(-np.pi, np.pi, SIZE)
    y = np.linspace(-np.pi, np.pi, SIZE)
    xx, yy = np.meshgrid(x, y)
    z = np.cos(3 * xx) * np.cos(3 * yy)
    return 128.0 + 120.0 * z


def shapes() -> np.ndarray:
    img = np.full((SIZE, SIZE), 40.0)
    yy, xx = np.indices((SIZE, SIZE))
    r = np.hypot(xx - 40, yy - 40)
    img[r < 24] = 220
    img[(xx > 70) & (xx < 110) & (yy > 70) & (yy < 110)] = 150
    img[(yy > 20) & (yy < 25)] = 255
    return img


def noise_pattern() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = 128 + 30 * np.sin(np.linspace(0, 6 * np.pi, SIZE))[None, :]
    base = base + 20 * np.cos(np.linspace(0, 4 * np.pi, SIZE))[:, None]
    base = base + rng.normal(0, 10, size=(SIZE, SIZE))
    return base


def main() -> None:
    print("generating synthetic test images →")
    _save("gradient", gradient())
    _save("checkerboard", checkerboard())
    _save("sinusoid", sinusoid())
    _save("shapes", shapes())
    _save("noise", noise_pattern())


if __name__ == "__main__":
    main()
