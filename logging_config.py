"""Centralized logging setup for USPS IMB Scanner.

Call setup_logging() once at entrypoint. Writes to both stdout and a
timestamped file in logs/ tagged with the run UUID so pytest artifacts
and the parent run share an ID.
"""
from __future__ import annotations

import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_NAME = "usps_imb_scanner"

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | [run_id=%(run_id)s] | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _RunIdFilter(logging.Filter):
    def __init__(self, run_id: str):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self.run_id
        return True


def _resolve_run_id() -> str:
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        run_id = str(uuid.uuid4())
        os.environ["RUN_ID"] = run_id
    return run_id


def _resolve_level(debug: bool) -> int:
    if debug:
        return logging.DEBUG
    env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    return getattr(logging, env_level, logging.INFO)


def setup_logging(debug: bool = False, log_dir: str | Path = "logs") -> dict:
    """Configure root logging once per process. Subsequent calls are no-ops
    (important for Streamlit, which reruns the script on every interaction).
    Returns {run_id, log_file}.
    """
    root = logging.getLogger()
    if getattr(root, "_usps_imb_configured", False):
        return {
            "run_id": os.environ.get("RUN_ID", ""),
            "log_file": getattr(root, "_usps_imb_log_file", ""),
        }

    run_id = _resolve_run_id()
    run_id_short = run_id[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{PROJECT_NAME}_{timestamp}_{run_id_short}.log"

    level = _resolve_level(debug)

    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    run_filter = _RunIdFilter(run_id)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(run_filter)
    root.addHandler(stream)

    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setFormatter(formatter)
    file_h.addFilter(run_filter)
    root.addHandler(file_h)

    root._usps_imb_configured = True
    root._usps_imb_log_file = str(log_file)

    logging.getLogger(__name__).info(
        "logging initialized | level=%s | file=%s", logging.getLevelName(level), log_file
    )
    return {"run_id": run_id, "log_file": str(log_file)}
