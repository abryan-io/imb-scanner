"""Streamlit app behavior, driven through AppTest with failed-scan capture off."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from streamlit.testing.v1 import AppTest

import cli_app
from tests.test_pdf_detection import FIXTURES_DIR, _parse_expected

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("FAILED_SCAN_BACKEND", "off")
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    return at


def _pdf_bytes(pages: int, **save_kwargs) -> bytes:
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page(width=72, height=72)
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data


def _upload(at: AppTest, name: str, data: bytes, mime: str) -> AppTest:
    return at.file_uploader(key="file_uploader").upload(name, data, mime).run()


def test_pdf_upload_decodes(app):
    pdf = FIXTURES_DIR / "00_271_000128_815033722.pdf"

    at = _upload(app, pdf.name, pdf.read_bytes(), "application/pdf")

    assert not at.exception
    assert any(_parse_expected(str(pdf))["tracking"] in s.value for s in at.success)


def test_image_upload_decodes(app):
    png = FIXTURES_DIR / "0024300086772377558932526296091.png"

    at = _upload(app, png.name, png.read_bytes(), "image/png")

    assert not at.exception
    assert any(f"Decoded: `{png.stem}`" in s.value for s in at.success)


def test_long_pdf_scans_only_first_pages(app, monkeypatch):
    scanned = []
    monkeypatch.setattr(cli_app, "scan_image_robust", lambda gray, *a: scanned.append(1))

    at = _upload(app, "long.pdf", _pdf_bytes(25), "application/pdf")

    assert not at.exception
    assert len(scanned) == 20
    assert any("first 20 of 25 pages" in w.value for w in at.warning)


def test_password_protected_pdf_shows_error(app):
    data = _pdf_bytes(1, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="o", user_pw="u")

    at = _upload(app, "locked.pdf", data, "application/pdf")

    assert not at.exception
    assert any("password-protected" in e.value for e in at.error)


def test_corrupt_image_shows_error(app):
    at = _upload(app, "broken.png", b"not an image", "image/png")

    assert not at.exception
    assert any("Failed to open image" in e.value for e in at.error)


def test_sidebar_mid_rule_matches_decoder(app):
    text = " ".join(m.value for m in app.sidebar.markdown)

    assert "6-digit MID: first digit is 0-8" in text
    assert "9-digit MID: first digit is 9" in text
