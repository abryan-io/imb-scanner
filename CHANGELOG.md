# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- `.gitignore` and `.env.example` establishing project conventions.
- Scaffolding directories: `logs/`, `test-results/`, `tests/fixtures/`, `scratch/`, `data/`.
- `conftest.py` with session-scoped `run_id` / `run_id_short` fixtures sourced from the `RUN_ID` env var so pytest artifacts share an identifier with the parent run.
- `logging_config.py` providing `setup_logging()` with run-id-tagged records, dual sink (stdout + `logs/*.log`), idempotent for Streamlit reruns.
- `pyproject.toml` and `uv.lock` for uv-managed dependencies; adds `boto3` (R2) and `python-dotenv`.
- `tests/test_pdf_detection.py` — pytest conversion of the legacy test_suite.py, parametrized over the fixture corpus.
- `failed_scan_store.py` with pluggable local / Cloudflare R2 backends for capturing images the scanner can't decode.
- CHANGELOG.md + expanded README covering uv, tests, failed-scan capture, and project layout.

### Changed
- `cli_app.py` and `app.py` now initialize logging and use module loggers; kept the formatted CLI result block as `print` since it's user-facing.
- README references `app.py` instead of the removed `zapp.py`.

### Removed
- `requirements.txt` — superseded by `pyproject.toml` + `uv.lock`.
- `test_suite.py` — superseded by `tests/test_pdf_detection.py`.
- Legacy `zapp.py`, `zzapp.py`, `zz_app.py`, `zzintelligent_mail_barcode.py`, `zz_intelligent_mail_barcode.py` — moved into `scratch/` (gitignored); history preserves them if ever needed.
- `venv/`, `__pycache__/`, `desktop.ini` untracked via `git rm --cached`. Files remain on disk; past commits still contain them (non-destructive cleanup).

### Notes
- Backup branch `pre-retrofit-backup` was created before any changes; safe to delete once the retrofit is verified.
- `.git/` stays at ~220MB because past commits contain `venv/`. A filter-repo pass would shrink it but would rewrite history — deferred.

## [0.1.0] — prior to 2026-04-21

Initial releases: PDF upload, MID lookup, robust multi-strategy IMB detector reaching 17/17 on the test corpus.
