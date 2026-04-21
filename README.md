# USPS IMB Scanner

Streamlit app that detects and decodes USPS Intelligent Mail Barcodes (IMB) from photos using classical computer vision — no AI, no API calls, fully deterministic.

## How it works

1. **Sobel-X edge density** finds the barcode band (the row with the densest vertical edges)
2. **Otsu binarization + 4x upscale** produces a clean binary crop
3. **Column projection** locates 65 bar centers with outlier filtering for text artifacts
4. **Largest-gap clustering** classifies each bar as F/A/D/T by measuring top/bottom pixel extent
5. **pyimb decode** converts FADT → codewords → tracking + routing (CRC-11 validated)

## Setup

```bash
# Install dependencies (uv syncs .venv from pyproject.toml + uv.lock)
uv sync

# Run the Streamlit app
uv run streamlit run app.py

# Or run the CLI decoder on a single image
uv run python cli_app.py <image_path>
```

Then open http://localhost:8501 in your browser.

## Requirements

- Python 3.9+
- See requirements.txt

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
