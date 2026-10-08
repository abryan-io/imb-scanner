"""Tests for mid_lookup normalization and the unknown-MID log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import unknown_mid_log
from mid_lookup import load_mid_table, normalize_mid
from unknown_mid_log import list_unknown_mids, record_unknown_mid

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from unknown_mids import format_rows  # noqa: E402


@pytest.mark.parametrize(
    "raw, expected",
    [
        (128, "000128"),
        (128.0, "000128"),
        (" 000128 ", "000128"),
        (271128, "271128"),
        (906894001, "906894001"),
        ("abc", "abc"),
    ],
)
def test_normalize_mid_restores_leading_zeros(raw, expected):
    assert normalize_mid(raw) == expected


def test_lookup_table_matches_decoded_six_digit_mid():
    assert "000128" in load_mid_table()


@pytest.fixture
def local_log(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILED_SCAN_BACKEND", "local")
    monkeypatch.setattr(unknown_mid_log, "LOCAL_DIR", tmp_path)
    return tmp_path


def _result(tracking: str, mid: str = "123456", stid: str = "271") -> dict:
    return {
        "tracking": tracking,
        "mailer_id": mid,
        "service_type": stid,
        "barcode_id": "00",
        "routing": "80234275425",
        "serial": "999",
    }


def test_records_new_mid(local_log):
    location = record_unknown_mid(_result("00271123456000000001"), source="camera")
    record = json.loads(Path(location).read_text())
    assert record["mid"] == "123456"
    assert len(record["pieces"]) == 1
    assert record["service_types"] == ["271"]
    assert record["sources"] == ["camera"]
    assert "80234275425" not in json.dumps(record)


def test_same_piece_counted_once(local_log):
    r = _result("00271123456000000001")
    assert record_unknown_mid(r, source="camera") is not None
    assert record_unknown_mid(r, source="camera") is None
    assert len(list_unknown_mids()[0]["pieces"]) == 1


def test_distinct_pieces_accumulate(local_log):
    record_unknown_mid(_result("00271123456000000001"), source="a.pdf p1")
    record_unknown_mid(_result("00300123456000000002", stid="300"), source="b.pdf p1")
    record = list_unknown_mids()[0]
    assert len(record["pieces"]) == 2
    assert record["service_types"] == ["271", "300"]
    assert record["sources"] == ["a.pdf p1", "b.pdf p1"]


def test_list_orders_by_pieces(local_log):
    record_unknown_mid(_result("00271111111000000001", mid="111111"))
    for i in range(2):
        record_unknown_mid(_result(f"0027122222200000000{i}", mid="222222"))
    assert [r["mid"] for r in list_unknown_mids()] == ["222222", "111111"]


def test_off_backend_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILED_SCAN_BACKEND", "off")
    monkeypatch.setattr(unknown_mid_log, "LOCAL_DIR", tmp_path)
    assert record_unknown_mid(_result("00271123456000000001")) is None
    assert list(tmp_path.iterdir()) == []


def test_format_rows_hides_mids_now_in_lookup(local_log):
    record_unknown_mid(_result("00271123456000000001", mid="123456"))
    record_unknown_mid(_result("00271654321000000001", mid="654321"))
    records = list_unknown_mids()
    rows = format_rows(records, known_mids={"654321"}, include_known=False)
    assert len(rows) == 2 and rows[1].startswith("123456\t1\t")
    assert len(format_rows(records, known_mids={"654321"}, include_known=True)) == 3
