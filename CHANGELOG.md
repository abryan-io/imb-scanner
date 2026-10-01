# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Scanner robustness overhaul in `cli_app.py`. Decode rate on the 17-page corpus under synthetic distortions, before → after: 1° rotation 2 → 17, 4° rotation 0 → 17, 90°/180° rotation 0 → 17, 2× scale 6 → 17, noise σ=20 6 → 17, JPEG q30 13 → 17, perspective 5% 4 → 17, uneven shading 11 → 16. Changes:
  - Scale normalization (long side capped at 3300px) and gated Gaussian denoise (Immerkær noise estimate on flat pixels).
  - Deskew from the dominant near-vertical edge angle; 90° retry; 180° handled by decoding each read in both directions.
  - FADT classification relative to a line fitted through the bar midpoints, using the ink run that crosses the tracker band; per-bar confidence scores.
  - Two new segmentation fallbacks: bar-shaped connected components along a common (tilted) line, and a regular pitch grid fitted to bar centers.
  - Background-flattened Otsu binarization for uneven lighting.
  - CRC-guided error correction of near-miss reads: single-bit flips limited to the 8 lowest-confidence bar measurements, at most 2 invalid codewords; results carry `corrected_bars`.
- `scan_image_robust(_all)` take an optional `diagnostics` dict that receives the best near-miss FADT on failure; the Streamlit app shows it and stores it in failed-scan sidecars.
- `tests/test_scanner_units.py` (synthetic barcodes), `tests/test_robustness.py` (`-m robustness`), and PNG fixtures in `tests/test_pdf_detection.py`.
- HEIC/HEIF uploads via `pillow-heif`.
- `failed_scan_store.record_failure()` now embeds the last 50 lines of the current run's log file in each sidecar as `log_tail`. Gives you the diagnostic trail leading up to a failure — useful when pulling captures down from R2 where the source log file isn't available. Covered by 4 new tests in `tests/test_failed_scan_store.py`.
- Pytest reporter in `conftest.py`: appends a JSON row per run to `test-results/history.jsonl` (timestamp, run_id, commit_sha, duration, pass/fail counts) and writes a per-run markdown summary.
- `tools/label_corpus.py`: interactive CLI to walk captured `data/failed_scans/` sidecars and label each image with expected IMb fields. Writes the label block back into the same JSON so capture metadata is preserved.
- `tests/test_label_corpus.py`: 23 unit tests covering the labeler validation (MID/serial length pairing, routing lengths, non-digit rejection).
- `.gitignore` and `.env.example` establishing project conventions.
- Scaffolding directories: `logs/`, `test-results/`, `tests/fixtures/`, `scratch/`, `data/`.
- `conftest.py` with session-scoped `run_id` / `run_id_short` fixtures sourced from the `RUN_ID` env var so pytest artifacts share an identifier with the parent run.
- `logging_config.py` providing `setup_logging()` with run-id-tagged records, dual sink (stdout + `logs/*.log`), idempotent for Streamlit reruns.
- `pyproject.toml` and `uv.lock` for uv-managed dependencies; adds `boto3` (R2) and `python-dotenv`.
- `tests/test_pdf_detection.py` — pytest conversion of the legacy test_suite.py, parametrized over the fixture corpus.
- `failed_scan_store.py` with pluggable local / Cloudflare R2 backends for capturing images the scanner can't decode.
- CHANGELOG.md + expanded README covering uv, tests, failed-scan capture, and project layout.

### Changed
- Streamlit uploads and camera captures honor EXIF orientation.
- Morphological detector's vertical opening kernel is now `H // 400` (was `H // 100`, which erased all but full-height bars at 300 DPI).
- Unified the duplicated sliding-window decode paths into `_decode_binary()`.
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
