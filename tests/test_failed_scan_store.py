"""Tests for failed_scan_store."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from failed_scan_store import record_failure
from logging_config import setup_logging


@pytest.fixture
def local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILED_SCAN_BACKEND", "local")
    monkeypatch.setenv("FAILED_SCAN_LOCAL_PATH", str(tmp_path))
    return tmp_path


def _read_sidecar(dir_: Path) -> dict:
    sidecars = list(dir_.glob("failed_*.json"))
    assert len(sidecars) == 1, f"expected 1 sidecar, got {len(sidecars)}"
    return json.loads(sidecars[0].read_text())


def test_local_capture_writes_image_and_sidecar(local_backend):
    img = np.full((100, 200), 180, dtype=np.uint8)
    location = record_failure(img, source="unit.png", reason="test")
    assert location is not None
    assert Path(location).exists()

    meta = _read_sidecar(local_backend)
    assert meta["source"] == "unit.png"
    assert meta["reason"] == "test"
    assert meta["width"] == 200 and meta["height"] == 100
    assert meta["backend"] == "local"


def test_backend_off_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILED_SCAN_BACKEND", "off")
    monkeypatch.setenv("FAILED_SCAN_LOCAL_PATH", str(tmp_path))
    img = np.zeros((10, 10), dtype=np.uint8)
    assert record_failure(img, source="x") is None
    assert list(tmp_path.glob("*")) == []


def test_log_tail_embedded_when_logging_configured(local_backend, tmp_path, monkeypatch):
    # Point setup_logging at a scratch log dir so we don't pollute ./logs
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    setup_logging(log_dir=tmp_path / "logs")
    logging.getLogger("test").info("sentinel pre-capture line")

    img = np.full((50, 80), 200, dtype=np.uint8)
    record_failure(img, source="with_logs.png", reason="sentinel_check")

    meta = _read_sidecar(local_backend)
    assert meta["log_tail"] is not None
    joined = "\n".join(meta["log_tail"])
    # Tail captures the trail *leading up to* the failure — the
    # "captured failed scan" log line fires after meta is built, so
    # it's intentionally not included.
    assert "sentinel pre-capture line" in joined


def test_log_tail_none_when_logging_not_configured(tmp_path, monkeypatch):
    # Clear any prior setup_logging state from other tests
    root = logging.getLogger()
    if hasattr(root, "_usps_imb_configured"):
        delattr(root, "_usps_imb_configured")
    if hasattr(root, "_usps_imb_log_file"):
        delattr(root, "_usps_imb_log_file")

    monkeypatch.setenv("FAILED_SCAN_BACKEND", "local")
    monkeypatch.setenv("FAILED_SCAN_LOCAL_PATH", str(tmp_path))
    img = np.zeros((10, 10), dtype=np.uint8)
    record_failure(img, source="x")

    meta = _read_sidecar(tmp_path)
    assert meta["log_tail"] is None
