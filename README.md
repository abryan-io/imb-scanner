# USPS IMB Scanner

Streamlit app that detects and decodes USPS Intelligent Mail Barcodes (IMB) from photos using classical computer vision — no AI, no API calls, fully deterministic.

## How it works

1. **OpenCV** locates the IMB band via morphological analysis (vertical bar clustering)
2. **zxingcpp** decodes the 4-state barcode (FADT → binary → 20-digit number)
3. **Pure Python** parses the 20-digit tracking code into: Barcode ID, STID, MID, Serial Number, Routing Code

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
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
