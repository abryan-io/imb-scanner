"""Pytest session setup.

Reads RUN_ID from env so test artifacts (junit xml, log files, history.jsonl)
share the same identifier as the run that spawned them. If not set (e.g.
running pytest directly), generates one.
"""
import os
import uuid
import pytest


def _session_run_id() -> str:
    run_id = os.environ.get("RUN_ID")
    if not run_id:
        run_id = str(uuid.uuid4())
        os.environ["RUN_ID"] = run_id
    return run_id


@pytest.fixture(scope="session")
def run_id() -> str:
    return _session_run_id()


@pytest.fixture(scope="session")
def run_id_short(run_id: str) -> str:
    return run_id[:8]


def pytest_configure(config):
    _session_run_id()
