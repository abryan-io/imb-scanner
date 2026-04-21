# USPS IMB Scanner

Streamlit app that detects and decodes USPS Intelligent Mail Barcodes (IMB) from photos and PDFs using classical computer vision — no AI, no API calls, fully deterministic.

## How it works

1. **Sobel-X edge density** finds the barcode band (the row with the densest vertical edges)
2. **Otsu binarization + 4x upscale** produces a clean binary crop
3. **Column projection** locates 65 bar centers with outlier filtering for text artifacts
4. **Largest-gap clustering** classifies each bar as F/A/D/T by measuring top/bottom pixel extent
5. **pyimb decode** converts FADT → codewords → tracking + routing (CRC-11 validated)

The robust scanner tries multiple detection strategies (multi-candidate Sobel peaks, CLAHE-enhanced, morphological, inverted, horizontal strip scan) and four binarization variants per region.

## Setup

```bash
# Install dependencies (uv creates/syncs .venv from pyproject.toml + uv.lock)
uv sync

# Run the Streamlit app
uv run streamlit run app.py

# Or run the CLI decoder on a single image
uv run python cli_app.py <image_path>
```

Then open http://localhost:8501 in your browser.

## Configuration

Copy `.env.example` to `.env` and fill in values:

| Variable | Purpose |
|---|---|
| `FAILED_SCAN_BACKEND` | `local` (default), `r2`, or `off` |
| `FAILED_SCAN_LOCAL_PATH` | Where to stash failed-scan images when backend is `local` |
| `CLOUDFLARE_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_ENDPOINT_URL` | R2 bucket credentials |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## Tests

```bash
uv run pytest tests/ -v
```

Corpus lives in `tests/fixtures/` — 17 PDFs + 6 PNGs. Filenames encode expected IMB values (`BarcodeID_STID_MID_Serial.{pdf,png}`) so each test reads the truth from the filename.

Test artifacts land in `test-results/` (junit XML per run, plus `history.jsonl`). Logs land in `logs/` with the same `run_id_short` so they line up.

## Failed-scan capture

When the scanner can't decode an image, it writes the image + metadata to `FAILED_SCAN_BACKEND`:

- **`local`**: `./data/failed_scans/` (gitignored)
- **`r2`**: Cloudflare R2 bucket (for Streamlit Cloud deploys — filesystem is ephemeral)
- **`off`**: disabled

Pull R2 failures to your machine for labeling:

```bash
# One-time rclone config: rclone config  (S3 provider, endpoint = your R2 endpoint URL)
rclone sync r2:usps-imb-scanner ./data/failed_scans --progress
```

Labeled corpus feeds into future regression tests under `tests/fixtures/regression_corpus/`.

## Layout

```
USPS_IMB_Scanner/
├── app.py                      Streamlit UI
├── cli_app.py                  Core detection pipeline + CLI entrypoint
├── intelligent_mail_barcode.py FADT decoder (pyimb port, CRC-11)
├── stid_table.py               Service Type ID lookup
├── logging_config.py           setup_logging() — run_id, dual sink
├── failed_scan_store.py        Pluggable local/R2 capture of failed scans
├── MID_Lkp.xlsx                Mailer ID → company lookup
├── pyproject.toml / uv.lock    Dependency management
├── conftest.py                 Pytest session setup (RUN_ID fixture)
├── tests/                      pytest test suite + fixtures
├── logs/                       Run logs (gitignored, .gitkeep tracked)
├── test-results/               JUnit XML + history.jsonl
├── data/failed_scans/          Captured failures (local backend, gitignored)
└── scratch/                    Ad-hoc work (gitignored)
```

## IMB Field Reference

| Field | Length | Notes |
|-------|--------|-------|
| Barcode ID | 2 digits | Barcode type indicator |
| STID | 3 digits | Mail class + ancillary service |
| MID | 6 or 9 digits | Mailer identifier (USPS-assigned) |
| Serial Number | 9 or 6 digits | Piece identifier (complements MID to 15 digits) |
| Routing Code | 0, 5, 9, or 11 digits | ZIP / ZIP+4 / ZIP+4+DPC |

**MID length rule:** if position 5 is digit 0–8 → 9-digit MID; if digit 9 → 6-digit MID.

## Tips for best results

- Straight-on shot, good lighting
- IMB in focus and not obscured
- Higher resolution = more pixel data for the decoder
- If auto-detect fails, try pre-cropping your image around just the barcode
