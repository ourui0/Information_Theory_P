"""End-to-end pipeline tests covering every combination of modules."""
from __future__ import annotations

import unittest

import numpy as np

from . import _common  # noqa: F401
from src import pipeline


def _smooth_image(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    h, w = shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    z = 127.5 * (1 + np.cos(np.pi * xx) * np.cos(np.pi * yy))
    return np.clip(z, 0, 255).astype(np.uint8)


class TestPipelineNoNoise(unittest.TestCase):
    """Without channel errors every combination must reproduce the image
    exactly (modulo lossy-source-coder distortion only)."""

    def test_dct_none_bsc_p0(self) -> None:
        img = _smooth_image()
        cfg = pipeline.Config(
            source="dct", source_param=85,
            channel_code="none", channel="bsc", channel_p=0.0, seed=0,
        )
        rep = pipeline.run(img, cfg)
        self.assertGreater(rep.psnr_db, 35.0)
        self.assertEqual(rep.ber_raw, 0.0)
        self.assertEqual(rep.ber_after_channel_dec, 0.0)

    def test_quantize_none_bec_p0(self) -> None:
        img = _smooth_image()
        cfg = pipeline.Config(
            source="quantize", source_param=8,
            channel_code="none", channel="bec", channel_p=0.0, seed=0,
        )
        rep = pipeline.run(img, cfg)
        self.assertEqual(rep.mse, 0.0)           # 8-bit quantizer is lossless

    def test_all_channel_codes_noiseless(self) -> None:
        img = _smooth_image()
        for code, param in (("none", 1), ("repetition", 3), ("repetition", 5), ("hamming", 4)):
            for ch in ("bsc", "bec"):
                cfg = pipeline.Config(
                    source="dct", source_param=85,
                    channel_code=code, channel_param=param,
                    channel=ch, channel_p=0.0, seed=0,
                )
                rep = pipeline.run(img, cfg)
                with self.subTest(code=code, param=param, channel=ch):
                    self.assertEqual(rep.ber_after_channel_dec, 0.0)
                    self.assertGreater(rep.psnr_db, 35.0)


class TestPipelineCodingGain(unittest.TestCase):
    """Under noise a proper channel code must reduce decoded BER below the
    raw BER and improve the reconstructed PSNR."""

    def test_hamming_improves_bsc(self) -> None:
        img = _smooth_image()
        base = pipeline.Config(
            source="dct", source_param=60,
            channel_code="none", channel="bsc", channel_p=0.01, seed=7,
        )
        coded = pipeline.Config(
            source="dct", source_param=60,
            channel_code="hamming", channel="bsc", channel_p=0.01, seed=7,
        )
        r_base = pipeline.run(img, base)
        r_coded = pipeline.run(img, coded)
        self.assertLess(r_coded.ber_after_channel_dec, r_base.ber_after_channel_dec)
        self.assertGreater(r_coded.psnr_db, r_base.psnr_db)

    def test_repetition5_improves_bec(self) -> None:
        img = _smooth_image()
        base = pipeline.Config(
            source="dct", source_param=60,
            channel_code="none", channel="bec", channel_p=0.1, seed=11,
        )
        coded = pipeline.Config(
            source="dct", source_param=60,
            channel_code="repetition", channel_param=5,
            channel="bec", channel_p=0.1, seed=11,
        )
        r_base = pipeline.run(img, base)
        r_coded = pipeline.run(img, coded)
        self.assertLess(r_coded.ber_after_channel_dec, r_base.ber_after_channel_dec)
        self.assertGreater(r_coded.psnr_db, r_base.psnr_db + 5.0)


class TestPipelineReportFields(unittest.TestCase):
    def test_report_has_all_fields(self) -> None:
        img = _smooth_image((48, 48))
        cfg = pipeline.Config(
            source="dct", source_param=50,
            channel_code="hamming", channel="bsc", channel_p=0.0, seed=0,
        )
        rep = pipeline.run(img, cfg)
        for attr in (
            "psnr_db", "mse", "accuracy_exact", "accuracy_tol5",
            "ber_raw", "ber_after_channel_dec",
            "n_info_bits", "n_tx_bits", "code_rate",
            "compression_ratio", "encode_time_s", "decode_time_s",
        ):
            self.assertTrue(hasattr(rep, attr), f"missing attribute {attr}")
        self.assertAlmostEqual(rep.code_rate, 4 / 7, places=3)
        self.assertEqual(rep.recovered.shape, img.shape)
        self.assertGreaterEqual(rep.compression_ratio, 0.0)

    def test_determinism_with_seed(self) -> None:
        img = _smooth_image((48, 48))
        cfg = pipeline.Config(
            source="dct", source_param=40,
            channel_code="hamming", channel="bsc", channel_p=0.05, seed=123,
        )
        r1 = pipeline.run(img, cfg)
        r2 = pipeline.run(img, cfg)
        np.testing.assert_array_equal(r1.recovered, r2.recovered)
        self.assertEqual(r1.ber_after_channel_dec, r2.ber_after_channel_dec)


if __name__ == "__main__":
    unittest.main()
