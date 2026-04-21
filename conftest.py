"""Pytest session setup and reporters.

- Shares RUN_ID with the spawning process so artifacts line up across
  log files, junit XML, markdown summaries, and history.jsonl.
- Appends one line per run to test-results/history.jsonl.
- Writes a markdown summary per run to test-results/run_{ts}_{sha}.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

import pytest


RESULTS_DIR = Path("test-results")
HISTORY_PATH = RESULTS_DIR / "history.jsonl"

_SESSION_START: float | None = None


def _session_run_id() -> str:
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        run_id = str(uuid.uuid4())
        os.environ["RUN_ID"] = run_id
    return run_id


def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


@pytest.fixture(scope="session")
def run_id() -> str:
    return _session_run_id()


@pytest.fixture(scope="session")
def run_id_short(run_id: str) -> str:
    return run_id[:8]


def pytest_configure(config):
    global _SESSION_START
    _SESSION_START = time.time()
    _session_run_id()


def _stat_counts(stats: dict) -> dict:
    return {
        "passed": len(stats.get("passed", [])),
        "failed": len(stats.get("failed", [])),
        "skipped": len(stats.get("skipped", [])),
        "errors": len(stats.get("error", [])),
    }


def _append_history(entry: dict) -> None:
    if not RESULTS_DIR.exists():
        return
    with HISTORY_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _write_markdown(md_path: Path, entry: dict, stats: dict) -> None:
    lines: list[str] = []
    lines.append(f"# Test Run — {entry['timestamp']}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Run ID | `{entry['run_id']}` |")
    lines.append(f"| Commit | `{entry['commit_sha'][:8] or 'n/a'}` |")
    lines.append(f"| Duration | {entry['duration_s']}s |")
    lines.append(f"| Passed | {entry['passed']} |")
    lines.append(f"| Failed | {entry['failed']} |")
    lines.append(f"| Skipped | {entry['skipped']} |")
    lines.append(f"| Errors | {entry['errors']} |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Test | Status | Duration |")
    lines.append("|---|---|---|")
    for status, icon in [("passed", "PASS"), ("failed", "FAIL"),
                         ("skipped", "SKIP"), ("error", "ERR")]:
        for r in stats.get(status, []):
            lines.append(f"| `{r.nodeid}` | {icon} | {getattr(r, 'duration', 0):.2f}s |")
    lines.append("")

    failures = stats.get("failed", []) + stats.get("error", [])
    if failures:
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            lines.append(f"### `{r.nodeid}`")
            lines.append("")
            lines.append("```")
            lines.append(str(getattr(r, "longrepr", "")).strip() or "(no detail)")
            lines.append("```")
            lines.append("")

    md_path.write_text("\n".join(lines))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Emit history.jsonl row + markdown summary after the session wraps."""
    stats = terminalreporter.stats
    counts = _stat_counts(stats)
    duration = time.time() - (_SESSION_START or time.time())

    run_id = os.environ.get("RUN_ID", "")
    run_id_short = run_id[:8] if run_id else ""
    now = datetime.now()

    entry = {
        "timestamp": now.isoformat(timespec="seconds"),
        "run_id": run_id,
        "commit_sha": _git_sha(),
        "duration_s": round(duration, 2),
        **counts,
    }
    _append_history(entry)

    md_path = RESULTS_DIR / f"run_{now.strftime('%Y%m%d_%H%M%S')}_{run_id_short}.md"
    try:
        _write_markdown(md_path, entry, stats)
    except Exception as e:
        terminalreporter.write_line(f"[reporter] markdown write failed: {e}", yellow=True)
