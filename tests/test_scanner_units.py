"""Fast scanner tests on synthetically rendered IMBs (no PDF corpus needed)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import cli_app
import intelligent_mail_barcode as imb

TRACKING = "00271000128815033722"
FADT = imb.encode(0, 271, 128, 815033722, "")


def _render(fadt: str, pitch: int = 15, bar_w: int = 7) -> np.ndarray:
    """IMB on a white page-like canvas, with a text-like line above it."""
    img = np.full((400, 65 * pitch + 400), 255, np.uint8)
    x0, y0 = 200, 200
    for i, ch in enumerate(fadt):
        x = x0 + i * pitch
        top = y0 if ch in "AF" else y0 + 20
        bottom = y0 + 60 if ch in "DF" else y0 + 40
        img[top:bottom, x:x + bar_w] = 0
    cv2.putText(img, "PO BOX 4175 CAROL STREAM IL", (x0, y0 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)
    return img


def _read_from_fadt(fadt: str) -> cli_app.BarRead:
    asc = np.array([c in "AF" for c in fadt])
    desc = np.array([c in "DF" for c in fadt])
    return cli_app.BarRead(asc, desc, np.ones(65), np.ones(65))


def _trackings(gray: np.ndarray) -> list[str]:
    return [r["tracking"] for r in cli_app.scan_image_robust_all(gray)]


def test_decodes_rendered_barcode():
    assert _trackings(_render(FADT)) == [TRACKING]


@pytest.mark.parametrize("angle", [-6, -3, -1, 1, 3, 6])
def test_decodes_skewed_barcode(angle):
    assert TRACKING in _trackings(cli_app.rotate_image(_render(FADT), angle))


@pytest.mark.parametrize("rotation", [
    cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE,
])
def test_decodes_rotated_barcode(rotation):
    assert TRACKING in _trackings(cv2.rotate(_render(FADT), rotation))


def test_decodes_large_image_after_scale_normalization():
    big = cv2.resize(_render(FADT), None, fx=3.5, fy=3.5, interpolation=cv2.INTER_NEAREST)
    assert max(big.shape) > cli_app.WORKING_MAX_SIDE
    assert TRACKING in _trackings(big)


def test_decodes_noisy_barcode():
    noisy = _render(FADT) + np.random.default_rng(0).normal(0, 25, (400, 1375))
    assert TRACKING in _trackings(np.clip(noisy, 0, 255).astype(np.uint8))


def test_skew_estimate_inverts_rotation():
    rotated = cli_app.rotate_image(_render(FADT), 4)
    assert cli_app.estimate_skew_angle(rotated) == pytest.approx(-4, abs=0.6)


def test_flipped_read_is_180_degree_view():
    upside_down = _read_from_fadt(FADT).flipped()
    assert imb.decode(upside_down.fadt()) is None
    assert upside_down.flipped().fadt() == FADT


def _corrupt(read: cli_app.BarRead, bars: list[int]) -> cli_app.BarRead:
    """Flip the ascender bit of each bar and mark it low-confidence."""
    for b in bars:
        read.asc[b] = not read.asc[b]
        read.asc_conf[b] = 0.05
    return read


def test_error_correction_repairs_weak_bars():
    read = _corrupt(_read_from_fadt(FADT), [3, 40])
    assert cli_app._decode_fadt(read.fadt()) is None
    result = cli_app._error_correct(read)
    assert result["tracking"] == TRACKING
    assert result["corrected_bars"] == 2


def test_error_correction_works_upside_down():
    read = _corrupt(_read_from_fadt(FADT), [10]).flipped()
    assert cli_app._error_correct(read)["tracking"] == TRACKING


def test_error_correction_ignores_confident_bars():
    read = _corrupt(_read_from_fadt(FADT), [3])
    read.asc_conf[:] = 1.0
    read.desc_conf[:] = 0.05
    assert cli_app._error_correct(read) is None


def test_error_correction_gives_up_on_too_many_bad_codewords():
    read = _read_from_fadt(FADT)
    bars = []
    codewords_hit = set()
    for i in range(65):
        cw = imb.tableA[i][0]
        if cw not in codewords_hit:
            codewords_hit.add(cw)
            bars.append(i)
        if len(bars) > cli_app.EC_MAX_BAD_CODEWORDS:
            break
    assert cli_app._error_correct(_corrupt(read, bars)) is None


def test_noise_estimate_separates_clean_from_noisy():
    clean = _render(FADT)
    noisy = np.clip(clean + np.random.default_rng(0).normal(0, 12, clean.shape), 0, 255)
    assert cli_app.estimate_noise_sigma(clean) < cli_app.NOISE_SIGMA_DENOISE
    assert cli_app.estimate_noise_sigma(noisy.astype(np.uint8)) > cli_app.NOISE_SIGMA_DENOISE
