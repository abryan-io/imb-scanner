"""IMB detection tests over the real-mailpiece PDF corpus.

Each fixture PDF is named BarcodeID_STID_MID_Serial.pdf so the expected
decode is encoded in the filename. Any page of the PDF that yields a
CRC-valid decode matching the expected tuple passes the test.
"""
from __future__ import annotations

import glob
import logging
import os
import time
from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest
from PIL import Image

from cli_app import scan_image_robust_all

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
PDF_FILES = sorted(glob.glob(str(FIXTURES_DIR / "*.pdf")))


def _parse_expected(pdf_path: str) -> dict | None:
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    parts = base.split("_")
    if len(parts) != 4:
        return None
    bid, stid, mid, serial = parts
    return {
        "barcode_id": bid,
        "service_type": stid,
        "mailer_id": mid,
        "serial": serial,
        "tracking": bid + stid + mid + serial,
    }


def _pdf_to_images(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        out.append((page_num, img))
    doc.close()
    return out


def _scan(pil_img: Image.Image) -> list:
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return scan_image_robust_all(gray)


@pytest.mark.parametrize("pdf_path", PDF_FILES, ids=lambda p: os.path.basename(p))
def test_pdf_decodes_expected_imb(pdf_path: str, run_id_short: str):
    expected = _parse_expected(pdf_path)
    assert expected is not None, f"filename does not encode expected values: {pdf_path}"

    start = time.time()
    pages = _pdf_to_images(pdf_path)
    # Landscape (envelopes) first — empirical hot path.
    pages.sort(key=lambda item: 0 if item[1].size[0] > item[1].size[1] else 1)

    decodes: list[dict] = []
    for page_num, img in pages:
        for r in _scan(img):
            if not r.get("crc_ok"):
                continue
            if any(d["tracking"] == r["tracking"] for d in decodes):
                continue
            decodes.append(r)
            if (
                r["barcode_id"] == expected["barcode_id"]
                and r["service_type"] == expected["service_type"]
                and r["mailer_id"] == expected["mailer_id"]
                and r["serial"] == expected["serial"]
            ):
                elapsed = time.time() - start
                logger.info(
                    "pdf pass | file=%s page=%d elapsed=%.2fs run_id_short=%s",
                    os.path.basename(pdf_path), page_num, elapsed, run_id_short,
                )
                return

    elapsed = time.time() - start
    found = ", ".join(d["tracking"] for d in decodes) or "none"
    logger.warning(
        "pdf fail | file=%s elapsed=%.2fs decodes=%s expected=%s",
        os.path.basename(pdf_path), elapsed, found, expected["tracking"],
    )
    pytest.fail(
        f"expected tracking={expected['tracking']} not found. "
        f"Decoded: [{found}] in {elapsed:.2f}s"
    )
