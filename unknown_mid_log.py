"""Log decoded Mailer IDs that aren't in MID_Lkp.xlsx, for manual lookup.

Uses the same FAILED_SCAN_BACKEND as failed_scan_store (local / r2 / off).
One JSON record per MID:
    local -> ./data/unknown_mids/{mid}.json
    r2    -> unknown_mids/{mid}.json in the R2 bucket

Each record counts distinct mail pieces (by hashed tracking code), so
Streamlit reruns or re-scanning the same piece don't inflate the count.
Routing codes and serials are not stored.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

from failed_scan_store import _backend, _r2_client

logger = logging.getLogger(__name__)

LOCAL_DIR = Path("data/unknown_mids")
R2_PREFIX = "unknown_mids/"
MAX_SOURCES = 10


def _load_local(mid: str) -> dict | None:
    path = LOCAL_DIR / f"{mid}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _save_local(record: dict) -> str:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCAL_DIR / f"{record['mid']}.json"
    path.write_text(json.dumps(record, indent=2))
    return str(path)


def _load_r2(mid: str) -> dict | None:
    client, bucket = _r2_client()
    try:
        obj = client.get_object(Bucket=bucket, Key=f"{R2_PREFIX}{mid}.json")
    except client.exceptions.NoSuchKey:
        return None
    return json.loads(obj["Body"].read())


def _save_r2(record: dict) -> str:
    client, bucket = _r2_client()
    key = f"{R2_PREFIX}{record['mid']}.json"
    client.put_object(Bucket=bucket, Key=key, Body=json.dumps(record, indent=2).encode("utf-8"),
                      ContentType="application/json")
    return f"r2://{bucket}/{key}"


def record_unknown_mid(result: dict, *, source: str | None = None) -> str | None:
    """Add a sighting of result['mailer_id']. Returns the location written,
    or None if disabled, already counted, or on error. Never raises."""
    backend = _backend()
    if backend not in ("local", "r2"):
        return None
    try:
        mid = result["mailer_id"]
        piece = hashlib.sha256(result["tracking"].encode()).hexdigest()[:12]
        load, save = (_load_local, _save_local) if backend == "local" else (_load_r2, _save_r2)
        now = datetime.now().isoformat(timespec="seconds")

        record = load(mid) or {"mid": mid, "first_seen": now, "pieces": [],
                               "service_types": [], "barcode_ids": [], "sources": []}
        if piece in record["pieces"]:
            return None
        record["pieces"].append(piece)
        record["last_seen"] = now
        for field, value in (("service_types", result.get("service_type")),
                             ("barcode_ids", result.get("barcode_id"))):
            if value and value not in record[field]:
                record[field] = sorted(record[field] + [value])
        if source:
            record["sources"] = (record["sources"] + [source])[-MAX_SOURCES:]

        location = save(record)
        logger.info("unknown MID logged | mid=%s pieces=%d location=%s",
                    mid, len(record["pieces"]), location)
        return location
    except Exception:
        logger.exception("unknown_mid_log: record failed (non-fatal)")
        return None


def list_unknown_mids() -> list[dict]:
    """All logged records from the configured backend, most-seen first."""
    backend = _backend()
    records = []
    if backend == "local":
        records = [json.loads(p.read_text()) for p in sorted(LOCAL_DIR.glob("*.json"))]
    elif backend == "r2":
        client, bucket = _r2_client()
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=R2_PREFIX):
            for obj in page.get("Contents", []):
                body = client.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                records.append(json.loads(body))
    return sorted(records, key=lambda r: (-len(r["pieces"]), r["mid"]))
