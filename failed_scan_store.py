"""Capture images the scanner could not decode, for later manual labeling.

Backend is selected by the FAILED_SCAN_BACKEND env var:
    local  -> writes under FAILED_SCAN_LOCAL_PATH (default ./data/failed_scans)
    r2     -> writes to a Cloudflare R2 bucket via the S3-compatible API
    off    -> no-op

Each failed scan produces two files:
    failed_{YYYYMMDD_HHMMSS}_{sha8}.png       the original image
    failed_{YYYYMMDD_HHMMSS}_{sha8}.json      metadata sidecar

Metadata includes source (filename / page), attempts tried, run_id,
image dimensions, and reason — enough to reconstruct what the scanner
saw when it gave up.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _backend() -> str:
    return os.environ.get("FAILED_SCAN_BACKEND", "local").lower()


def _local_path() -> Path:
    return Path(os.environ.get("FAILED_SCAN_LOCAL_PATH", "./data/failed_scans"))


def _to_png_bytes(image: Any) -> bytes:
    """Accept PIL.Image, numpy array (grayscale or RGB), or bytes. Return PNG bytes."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            pil = Image.fromarray(image, mode="L")
        else:
            pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil = image
    else:
        raise TypeError(f"unsupported image type: {type(image).__name__}")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def _filenames(png_bytes: bytes) -> tuple[str, str, str]:
    sha8 = hashlib.sha256(png_bytes).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"failed_{ts}_{sha8}"
    return stem, f"{stem}.png", f"{stem}.json"


def _write_local(png_bytes: bytes, meta: dict, img_name: str, meta_name: str) -> str:
    target = _local_path()
    target.mkdir(parents=True, exist_ok=True)
    img_path = target / img_name
    meta_path = target / meta_name
    img_path.write_bytes(png_bytes)
    meta_path.write_text(json.dumps(meta, indent=2))
    return str(img_path)


def _write_r2(png_bytes: bytes, meta: dict, img_name: str, meta_name: str) -> str:
    try:
        import boto3
    except ImportError as e:
        raise RuntimeError("boto3 required for r2 backend — `uv sync` to install") from e

    required = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"missing env vars for r2 backend: {missing}")

    client = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )
    bucket = os.environ["R2_BUCKET_NAME"]
    client.put_object(Bucket=bucket, Key=img_name, Body=png_bytes, ContentType="image/png")
    client.put_object(
        Bucket=bucket,
        Key=meta_name,
        Body=json.dumps(meta, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    return f"r2://{bucket}/{img_name}"


def record_failure(
    image: Any,
    *,
    source: str | None = None,
    page: int | None = None,
    reason: str = "no_decode",
    attempts: list[str] | None = None,
    extra: dict | None = None,
) -> str | None:
    """Record a failed-scan image. Returns the location written, or None if off/errored.

    Never raises — storage failures are logged but must not crash the scanner.
    """
    backend = _backend()
    if backend == "off":
        return None

    try:
        png_bytes = _to_png_bytes(image)
        stem, img_name, meta_name = _filenames(png_bytes)

        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        elif isinstance(image, Image.Image):
            w, h = image.size
        else:
            h = w = None

        meta = {
            "stem": stem,
            "captured_at": datetime.now().isoformat(timespec="seconds"),
            "run_id": os.environ.get("RUN_ID"),
            "reason": reason,
            "source": source,
            "page": page,
            "width": w,
            "height": h,
            "attempts": attempts or [],
            "sha256_8": stem.rsplit("_", 1)[1],
            "backend": backend,
        }
        if extra:
            meta.update(extra)

        if backend == "local":
            location = _write_local(png_bytes, meta, img_name, meta_name)
        elif backend == "r2":
            location = _write_r2(png_bytes, meta, img_name, meta_name)
        else:
            logger.warning("unknown FAILED_SCAN_BACKEND=%s — skipping capture", backend)
            return None

        logger.info("captured failed scan | backend=%s location=%s source=%s reason=%s",
                    backend, location, source, reason)
        return location
    except Exception:
        logger.exception("failed_scan_store: capture failed (non-fatal)")
        return None
