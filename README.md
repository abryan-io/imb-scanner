# USPS IMB Scanner

Streamlit app that detects and decodes USPS Intelligent Mail Barcodes (IMB) from photos and PDFs using classical computer vision — no AI, no API calls, fully deterministic.

## How it works

1. **Normalize** — images larger than 3300px on the long side are downscaled; noisy images (estimated σ > 5) get a 5×5 Gaussian blur
2. **Orient** — the dominant near-vertical edge angle is measured and the image deskewed; if nothing decodes upright, the image is retried rotated 90°. Upside-down barcodes are handled by decoding every read in both directions
3. **Locate** — Sobel-X row energy (multi-candidate, plus CLAHE / morphological / inverted / strip-scan fallbacks) finds the barcode band
4. **Binarize** — five variants per region: Otsu, CLAHE+Otsu, adaptive Gaussian, background-flattened Otsu (uneven lighting), inverted Otsu
5. **Segment** into 65 bars, trying in order: column projection; bar-shaped connected components along a common tilted line (ignores nearby text); a regular pitch grid fitted to bar centers (recovers merged/split bars)
6. **Classify** F/A/D/T relative to a line fitted through the bars' tracker band, so skew and perspective don't shift the thresholds. Each bar gets a confidence score
7. **Decode** with pyimb (CRC-11). If nothing decodes cleanly, near-miss reads (≤ 2 invalid codewords) are repaired by flipping only the lowest-confidence bars, and accepted only if the CRC passes. Repaired decodes carry `corrected_bars`

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

Corpus lives in `tests/fixtures/` — 17 PDFs + 6 PNGs. PDF filenames encode the expected values (`BarcodeID_STID_MID_Serial.pdf`); PNG filenames are the full IMB number (20-digit tracking + routing). Each test reads the truth from the filename.

`tests/test_scanner_units.py` covers skew, 90°/180° rotation, scale, noise, and error correction on synthetically rendered barcodes (fast).

The robustness suite (`tests/test_robustness.py`) re-scans the corpus under seeded distortions — rotation, scale, blur, JPEG, noise, shading, perspective, and a phone-photo combination — and asserts a minimum decode rate per distortion. It takes ~10 minutes serially, so it is excluded by default:

```bash
uv run pytest -m robustness -n auto
```

Test artifacts land in `test-results/` (junit XML per run, plus `history.jsonl`). Logs land in `logs/` with the same `run_id_short` so they line up.

## Failed-scan capture

When the scanner can't decode an image, it writes the image + metadata to `FAILED_SCAN_BACKEND`:

- **`local`**: `./data/failed_scans/` (gitignored)
- **`r2`**: Cloudflare R2 bucket (for Streamlit Cloud deploys — filesystem is ephemeral)
- **`off`**: disabled

### Labeling workflow

1. **Pull from R2** (only needed when running from Streamlit Cloud):
   ```bash
   # One-time rclone config: rclone config  (S3 provider, endpoint = R2 endpoint URL)
   rclone sync r2:usps-imb-scanner ./data/failed_scans --progress
   ```
2. **Label interactively**:
   ```bash
   uv run python tools/label_corpus.py
   uv run python tools/label_corpus.py --open    # also open each image in default viewer
   ```
   The tool prompts for `barcode_id / stid / mid / serial / routing / notes` per image, validates lengths per the IMb spec, and writes a `labeled` block back into the sidecar JSON so the original capture metadata is preserved. Press Enter on the first field to skip an image; Ctrl-C to stop and resume later.
3. **Promote to regression corpus** once labeled: move the `failed_*.png` + `failed_*.json` pair into `tests/fixtures/regression_corpus/`. A future regression test can iterate that directory and assert the scanner reproduces the labeled decode.

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
├── conftest.py                 Pytest session setup + reporters
├── tools/
│   └── label_corpus.py         Interactive labeler for captured failures
├── tests/                      pytest test suite + fixtures
├── logs/                       Run logs (gitignored, .gitkeep tracked)
├── test-results/               JUnit XML + markdown summary + history.jsonl
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
