"""Unit tests for label_corpus validation logic."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/ is not a package; load the module directly
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from label_corpus import validate_labels, ValidationError  # noqa: E402


def test_valid_9_digit_mid():
    result = validate_labels({
        "barcode_id": "00",
        "stid": "243",
        "mid": "000867000",
        "serial": "723775",
        "routing": "",
    })
    assert result["tracking"] == "00243000867000723775"
    assert result["routing"] == ""


def test_valid_6_digit_mid():
    result = validate_labels({
        "barcode_id": "00",
        "stid": "243",
        "mid": "000867",
        "serial": "723775589",
        "routing": "32526296091",
    })
    assert result["tracking"] == "00243000867723775589"
    assert result["routing"] == "32526296091"


@pytest.mark.parametrize("routing", ["", "12345", "123456789", "12345678901"])
def test_valid_routing_lengths(routing):
    result = validate_labels({
        "barcode_id": "00", "stid": "243", "mid": "000867",
        "serial": "723775589", "routing": routing,
    })
    assert result["routing"] == routing


@pytest.mark.parametrize("bad_bid", ["0", "000", "ab", ""])
def test_bad_barcode_id(bad_bid):
    with pytest.raises(ValidationError):
        validate_labels({
            "barcode_id": bad_bid, "stid": "243", "mid": "000867",
            "serial": "723775589", "routing": "",
        })


@pytest.mark.parametrize("bad_stid", ["24", "2430", "abc"])
def test_bad_stid(bad_stid):
    with pytest.raises(ValidationError):
        validate_labels({
            "barcode_id": "00", "stid": bad_stid, "mid": "000867",
            "serial": "723775589", "routing": "",
        })


@pytest.mark.parametrize("bad_mid", ["00086", "0008670", "0008670000"])
def test_bad_mid_length(bad_mid):
    with pytest.raises(ValidationError):
        validate_labels({
            "barcode_id": "00", "stid": "243", "mid": bad_mid,
            "serial": "723775589", "routing": "",
        })


def test_serial_length_must_complement_mid():
    # 6-digit MID needs 9-digit serial
    with pytest.raises(ValidationError, match="serial must be 9"):
        validate_labels({
            "barcode_id": "00", "stid": "243", "mid": "000867",
            "serial": "723775", "routing": "",
        })
    # 9-digit MID needs 6-digit serial
    with pytest.raises(ValidationError, match="serial must be 6"):
        validate_labels({
            "barcode_id": "00", "stid": "243", "mid": "000867000",
            "serial": "723775589", "routing": "",
        })


@pytest.mark.parametrize("bad_routing", ["1", "1234", "123456", "123456789012"])
def test_bad_routing_length(bad_routing):
    with pytest.raises(ValidationError):
        validate_labels({
            "barcode_id": "00", "stid": "243", "mid": "000867",
            "serial": "723775589", "routing": bad_routing,
        })


def test_non_digit_routing():
    with pytest.raises(ValidationError):
        validate_labels({
            "barcode_id": "00", "stid": "243", "mid": "000867",
            "serial": "723775589", "routing": "abcde",
        })


def test_notes_passed_through():
    result = validate_labels({
        "barcode_id": "00", "stid": "243", "mid": "000867",
        "serial": "723775589", "routing": "", "notes": "blurry crop",
    })
    assert result["notes"] == "blurry crop"
