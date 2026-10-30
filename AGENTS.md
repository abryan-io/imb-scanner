# usps-imb-scanner

Project-specific notes only. Shared rules live in the global AGENTS.md.

## How to run

- `just setup`: install dependencies and the pre-commit hooks
- `just test`: lint, then tests except `@pytest.mark.robustness` (deselected by `addopts`)
- `just test-all`: every test, including robustness, in parallel (`-n auto`)
- `just lint`: fix lint and format
- `just run`: the Streamlit app (`app.py`) with its secrets from Bitwarden;
  `uv run python cli_app.py <image>` decodes one image
- `just update-template`: pull template changes (`copier update`)

## Where data lives

`data/` (gitignored): failed scans under `data/failed_scans/` with the `local` backend, or in
Cloudflare R2 with the `r2` backend. `MID_Lkp.xlsx` is tracked.

## Known quirks

- `just run` exports secrets named `NAME__USPS_IMB_SCANNER` from the Bitwarden Secrets Manager
  project `dev` (`just bws-run`). Direct `uv run` falls back to `.env` via python-dotenv (see
  `.env.example`), which never overrides variables already set.
- Flat layout, no `src/`: the template's package skeleton and smoke test are excluded.
- Every pytest run appends a row to the tracked `test-results/history.jsonl`.
- `stid_table.py` is hand-aligned and excluded from `ruff format`.
