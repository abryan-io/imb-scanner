"""The run reporter writes one history row per session, not one per xdist worker."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import conftest


@pytest.fixture
def results_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(conftest, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(conftest, "HISTORY_PATH", tmp_path / "history.jsonl")
    return tmp_path


def _reporter():
    return SimpleNamespace(stats={}, write_line=lambda *a, **k: None)


def test_xdist_worker_writes_nothing(results_dir):
    worker_config = SimpleNamespace(workerinput={"workerid": "gw0"})

    conftest.pytest_terminal_summary(_reporter(), 0, worker_config)

    assert list(results_dir.iterdir()) == []


def test_controller_appends_one_history_row(results_dir):
    conftest.pytest_terminal_summary(_reporter(), 0, SimpleNamespace())

    assert len((results_dir / "history.jsonl").read_text().splitlines()) == 1
