"""Robustness regression: decode rate of the PDF corpus under seeded distortions.

Each test applies one deterministic distortion (rotation, scale, blur, JPEG,
noise, shading, perspective, or a phone-photo combination) to every
known-good page and asserts the decode rate stays at or above its floor.
Raise a floor when an improvement lands; never lower one to make CI pass.

Slow (~10 min serial), so excluded by default. Run with:
    uv run pytest -m robustness -n auto
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from cli_app import scan_image_robust_all
from tests.test_pdf_detection import PDF_FILES, _parse_expected, _pdf_to_images, _scan

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.robustness


def _rotate(g, deg):
    h, w = g.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1)
    return cv2.warpAffine(g, m, (w, h), borderValue=255)


def _jpeg(g, quality):
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _noise(g, sigma):
    noisy = g + np.random.default_rng(0).normal(0, sigma, g.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _shade(g):
    h, w = g.shape
    return (g * np.tile(np.linspace(0.45, 1.0, w), (h, 1))).astype(np.uint8)


def _perspective(g, k):
    h, w = g.shape
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * k, 0], [w * (1 - k), h * k], [w * (1 - k), h * (1 - k)], [w * k, h]])
    return cv2.warpPerspective(g, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)


def _scale(g, f):
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(g, None, fx=f, fy=f, interpolation=interp)


def _phone(g):
    return _jpeg(_noise(_shade(_perspective(_rotate(_scale(g, 1.4), 3), 0.04)), 8), 50)


# name -> (distortion, minimum pages decoded out of the 17-page corpus)
DISTORTIONS = {
    "rotate_1deg": (lambda g: _rotate(g, 1), 17),
    "rotate_4deg": (lambda g: _rotate(g, 4), 17),
    "rotate_90": (lambda g: cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE), 17),
    "rotate_180": (lambda g: cv2.rotate(g, cv2.ROTATE_180), 17),
    "scale_0.33": (lambda g: _scale(g, 0.33), 17),
    "scale_2.0": (lambda g: _scale(g, 2.0), 17),
    "gaussian_blur_5": (lambda g: cv2.GaussianBlur(g, (5, 5), 0), 17),
    "motion_blur_7": (lambda g: cv2.filter2D(g, -1, np.ones((1, 7)) / 7), 17),
    "jpeg_q30": (lambda g: _jpeg(g, 30), 17),
    "jpeg_q15": (lambda g: _jpeg(g, 15), 12),
    "noise_sigma20": (lambda g: _noise(g, 20), 17),
    "shading": (_shade, 16),
    "perspective_8pct": (lambda g: _perspective(g, 0.08), 17),
    "phone_combo": (_phone, 17),
}


@pytest.fixture(scope="session")
def known_good_pages() -> list[tuple[np.ndarray, str]]:
    """(grayscale page, expected tracking) for the page carrying each IMB."""
    pages = []
    for pdf_path in PDF_FILES:
        expected = _parse_expected(pdf_path)["tracking"]
        images = sorted(_pdf_to_images(pdf_path), key=lambda it: it[1].size[0] <= it[1].size[1])
        for _, img in images:
            if any(r["tracking"] == expected for r in _scan(img)):
                gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
                pages.append((gray, expected))
                break
    return pages


@pytest.mark.parametrize("name", list(DISTORTIONS))
def test_decode_rate_under_distortion(name, known_good_pages, record_property):
    distort, floor = DISTORTIONS[name]
    decoded = sum(
        any(r["tracking"] == expected for r in scan_image_robust_all(distort(gray)))
        for gray, expected in known_good_pages
    )
    total = len(known_good_pages)
    record_property("decoded", decoded)
    record_property("total", total)
    logger.info("robustness | distortion=%s decoded=%d/%d floor=%d", name, decoded, total, floor)
    assert decoded >= floor, f"{name}: decoded {decoded}/{total}, floor {floor}"
