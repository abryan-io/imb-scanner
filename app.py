"""
USPS Intelligent Mail Barcode (IMb) Decoder
============================================
Full pipeline:
    1. Load image
    2. Grayscale
    3. Radon-based deskew  ← must run BEFORE Otsu
    4. Otsu binarization   ← operates on aligned image
    5. Barcode region detection & crop
    6. Bar segmentation → FADT string (programmatic, not vision-model)
    7. IMb decode (FADT → bit tracks → codewords → digits → payload)

Usage:
    python app.py <image_path>
    python app.py envelope.jpg

Dependencies:
    pip install opencv-python-headless scikit-image numpy
"""

import sys
import math
import numpy as np
import cv2
from skimage.transform import radon


# =============================================================================
# STAGE 1 — IMAGE LOADING & GRAYSCALE
# =============================================================================

def load_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =============================================================================
# STAGE 2 — RADON DESKEW  (must precede Otsu)
# =============================================================================

def radon_deskew(gray: np.ndarray,
                 angle_range: float = 10.0,
                 angle_steps: int = 361,
                 min_correction_deg: float = 0.3) -> tuple[np.ndarray, float]:
    """
    Find and correct image skew using Radon transform variance maximization.

    The Radon sinogram has maximum column variance when projection angle
    matches the dominant feature orientation — for IMb that's vertical bars,
    so the peak angle tells us how much the image is rotated off-axis.

    Args:
        gray:               Input grayscale image (pre-Otsu, raw)
        angle_range:        ± degrees to sweep (default ±10°)
        angle_steps:        Resolution of the sweep
        min_correction_deg: Skip rotation below this threshold (noise floor)

    Returns:
        (deskewed_gray, correction_angle_deg)
    """
    # Invert so bars are bright on dark — Radon accumulates bright energy
    inverted = 255 - gray

    theta = np.linspace(-angle_range, angle_range, angle_steps)
    sinogram = radon(inverted, theta=theta, circle=False)

    # Column with max variance = sharpest projection = best alignment
    col_variance = sinogram.var(axis=0)
    best_angle = float(theta[np.argmax(col_variance)])

    if abs(best_angle) <= min_correction_deg:
        return gray, 0.0

    h, w = gray.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    deskewed = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed, best_angle


# =============================================================================
# STAGE 3 — OTSU BINARIZATION  (on the aligned image)
# =============================================================================

def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's method on a skew-corrected grayscale image.
    Returns an inverted binary image (bars = 255 / white, background = 0).
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


# =============================================================================
# STAGE 4 — BARCODE REGION DETECTION & CROP
# =============================================================================

def find_barcode_region(binary: np.ndarray,
                        gray: np.ndarray,
                        min_aspect: float = 3.0,
                        max_aspect: float = 14.0,
                        min_area: int = 1500,
                        padding: int = 12) -> np.ndarray:
    """
    Locate the IMb bounding box via contour aspect-ratio filtering.
    IMb is a wide, short horizontal band (typically 3–12× wider than tall).

    Falls back to full image if no region is found.
    Returns a cropped grayscale region ready for bar analysis.
    """
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    best_score = 0
    barcode_box = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        area = w * h
        if min_aspect < aspect < max_aspect and area > min_area:
            score = area * aspect
            if score > best_score:
                best_score = score
                barcode_box = (x, y, w, h)

    if barcode_box is None:
        print("[WARN] No barcode region found — using full image.")
        return gray

    x, y, w, h = barcode_box
    img_h, img_w = gray.shape
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    return gray[y1:y2, x1:x2]


# =============================================================================
# STAGE 5 — UPSCALE & FINAL BINARIZATION
# =============================================================================

def prepare_for_classification(cropped_gray: np.ndarray,
                                scale: int = 4) -> np.ndarray:
    """
    Upscale with nearest-neighbor (preserves crisp bar edges) then re-binarize
    so bar segmentation works on a clean high-res image.
    """
    h, w = cropped_gray.shape
    upscaled = cv2.resize(
        cropped_gray, (w * scale, h * scale),
        interpolation=cv2.INTER_NEAREST
    )
    _, final_binary = cv2.threshold(
        upscaled, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return final_binary


# =============================================================================
# STAGE 6 — PROGRAMMATIC FADT CLASSIFICATION
# =============================================================================

def classify_bars(binary: np.ndarray,
                  expected_bars: int = 65) -> str:
    """
    Segment individual bars and classify each as F / A / D / T based on
    which vertical zones contain ink.

    IMb bar anatomy (three equal horizontal zones):
        Zone T (tracker/middle):  center third — ALL bars have ink here
        Zone A (ascender/top):    top third     — F and A bars extend here
        Zone D (descender/bot):   bottom third  — F and D bars extend here

    Classification:
        F = ink in A-zone AND D-zone  (full bar)
        A = ink in A-zone, NOT D-zone
        D = ink in D-zone, NOT A-zone
        T = neither A nor D zone

    Args:
        binary:        Inverted binary image (bars = 255)
        expected_bars: Expected bar count (65 for IMb)

    Returns:
        65-character FADT string, or raises ValueError if count is wrong.
    """
    h, w = binary.shape

    # Horizontal projection to find bar columns
    col_sum = binary.sum(axis=0)

    # Smooth to merge adjacent on-pixels within a bar
    kernel = np.ones(3, dtype=np.float32) / 3.0
    col_smooth = np.convolve(col_sum.astype(np.float32), kernel, mode='same')

    # Threshold: a column is "in a bar" if its ink density exceeds 5% of max
    ink_threshold = col_smooth.max() * 0.05
    in_bar = col_smooth > ink_threshold

    # Find contiguous bar runs
    bar_runs = []
    i = 0
    while i < w:
        if in_bar[i]:
            j = i
            while j < w and in_bar[j]:
                j += 1
            bar_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    if len(bar_runs) != expected_bars:
        raise ValueError(
            f"Expected {expected_bars} bars, found {len(bar_runs)}. "
            f"Adjust image preprocessing or check crop."
        )

    # Define three vertical zones
    zone_boundary_top    = h // 3        # 0 → top      (ascender zone)
    zone_boundary_bottom = 2 * h // 3   # bottom → end (descender zone)

    fadt = []
    for (col_start, col_end) in bar_runs:
        bar_col = binary[:, col_start:col_end + 1]

        top_ink    = bar_col[:zone_boundary_top, :].sum()
        bottom_ink = bar_col[zone_boundary_bottom:, :].sum()

        # Ink threshold per zone: 10% of total bar height worth of pixels
        zone_threshold = bar_col.sum() * 0.10

        has_ascender  = top_ink    > zone_threshold
        has_descender = bottom_ink > zone_threshold

        if has_ascender and has_descender:
            fadt.append("F")
        elif has_ascender:
            fadt.append("A")
        elif has_descender:
            fadt.append("D")
        else:
            fadt.append("T")

    return "".join(fadt)


# =============================================================================
# STAGE 7 — IMb DECODE  (FADT → payload digits)
# =============================================================================
# Reference: USPS Publication 197

CHAR_TO_BITS = {"F": 3, "A": 2, "D": 1, "T": 0}

# Descender bit positions (0-indexed, left-to-right across 65 bars)
# From USPS Pub 197 Table 3
DESCENDER_POSITIONS = [
     4, 0, 2, 6, 3,  5, 1,  9, 8,  7,
    19,11, 15,13,10, 16,12, 14,20, 18,
    17,21,22,23,24, 25,26,27,28,29,
    30,31,32,33,34, 35,36,37,38,39,
    40,41,42,43,44, 45,46,47,48,49,
    50,51,52,53,54, 55,56,57,58,59,
    60,61,62,63,64
]

# Ascender bit positions (0-indexed)
# From USPS Pub 197 Table 4
ASCENDER_POSITIONS = [
     0, 1, 2, 3, 4,  5, 6, 7, 8,  9,
    10,11,12,13,14, 15,16,17,18, 19,
    20,21,22,23,24, 25,26,27,28, 29,
    30,31,32,33,34, 35,36,37,38, 39,
    40,41,42,43,44, 45,46,47,48, 49,
    50,51,52,53,54, 55,56,57,58, 59,
    60,61,62,63,64
]

# Codeword slot radix values (mixed-radix table, USPS Pub 197 Table 5)
# 13 codewords total: index 0 has special handling
CODEWORD_BASES = [
    2, 8, 8, 8, 8, 8, 8,     # upper track codewords (7)
    2, 8, 8, 8, 8, 8          # lower track codewords (6) — first is 2-val
]

def fadt_to_bits(fadt: str) -> tuple[int, int]:
    """
    Map FADT characters at each bar position to two bit tracks.

    Descender bit → lower track (33 bits, LSB first per position table)
    Ascender bit  → upper track (32 bits, LSB first per position table)
    """
    bars = [CHAR_TO_BITS[c] for c in fadt]

    lower = 0
    for bit_pos, bar_idx in enumerate(DESCENDER_POSITIONS):
        descender_bit = bars[bar_idx] & 1   # bit 0 = descender
        lower |= (descender_bit << bit_pos)

    upper = 0
    for bit_pos, bar_idx in enumerate(ASCENDER_POSITIONS):
        ascender_bit = (bars[bar_idx] >> 1) & 1  # bit 1 = ascender
        upper |= (ascender_bit << bit_pos)

    return upper, lower


def bits_to_codewords(upper: int, lower: int) -> list[int]:
    """
    Slice the two bit tracks into 13 codewords using mixed-radix layout.

    Upper track (32 bits) → 7 codewords
    Lower track (33 bits) → 6 codewords
    Layout per USPS Pub 197 Table 5.
    """
    # Upper: [2-bit][5-bit][5-bit][5-bit][5-bit][5-bit][5-bit]
    cw = []
    cw.append(upper & 0x3)            # 2 bits
    val = upper >> 2
    for _ in range(6):
        cw.append(val & 0x1F)         # 5 bits each
        val >>= 5

    # Lower: [4-bit][5-bit][5-bit][5-bit][5-bit][5-bit][4-bit (unused padding)]
    cw.append(lower & 0xF)            # 4 bits
    val = lower >> 4
    for _ in range(5):
        cw.append(val & 0x1F)
        val >>= 5

    return cw[:13]


def codewords_to_digit_string(cws: list[int]) -> str:
    """
    Reconstruct the large integer from 13 codewords using mixed-radix arithmetic,
    then extract 20 raw decimal digits.

    Mixed radix bases (Table 5, USPS Pub 197):
        Upper codewords: [3, 32, 32, 32, 32, 32, 32]
        Lower codewords: [3, 32, 32, 32, 32, 32]
    Combined value covers 0..636,413,622,384,679,119 (just under 2^60)
    """
    bases = [3, 32, 32, 32, 32, 32, 32,   # upper 7
             3, 32, 32, 32, 32, 32]        # lower 6

    value = 0
    for i in range(12, -1, -1):
        value = value * bases[i] + cws[i]

    # Extract 20 decimal digits
    digits = []
    for _ in range(20):
        value, rem = divmod(value, 10)
        digits.append(str(rem))

    # Digits were extracted LSB first
    return "".join(reversed(digits))


def parse_payload(digit_str: str) -> dict:
    """
    Parse 20-digit string into IMb payload fields per USPS Pub 197.

    Field layout depends on whether MID is 6-digit or 9-digit:
        6-digit MID: MID value < 1,000,000
        9-digit MID: 6-digit value would encode as ≥ certain threshold
                     (determined by first two digits of the raw string)

    Standard layout:
        [0:2]   Barcode ID     (2 digits)
        [2:5]   STID           (3 digits)
        [5:11]  MID-6 / start of MID-9 field
        [5:14]  MID-9 (if applicable)
        Remaining → Serial + Routing ZIP
    """
    if len(digit_str) < 20:
        digit_str = digit_str.zfill(20)

    barcode_id = digit_str[0:2]
    stid       = digit_str[2:5]

    # Determine MID length from raw digit range
    mid_probe = int(digit_str[5:11])
    if mid_probe < 1_000_000:
        mid_6  = digit_str[5:11]
        mid_9  = None
        serial = digit_str[11:20]   # 9 digits
    else:
        mid_6  = None
        mid_9  = digit_str[5:14]
        serial = digit_str[14:20]   # 6 digits

    # Routing ZIP (last 0, 5, 9, or 11 digits — encoded within serial field)
    routing = _extract_routing(digit_str)

    return {
        "barcode_id": barcode_id,
        "stid":       stid,
        "mid_6":      mid_6,
        "mid_9":      mid_9,
        "serial":     serial,
        "routing":    routing,
    }


def _extract_routing(digit_str: str) -> str | None:
    """
    Routing ZIP is encoded in the last portion of the digit string.
    USPS encodes routing length via an offset system — simplified heuristic:
    if the trailing digits are all zeros, no routing is present.
    """
    trailing = digit_str[11:]
    if trailing == "0" * len(trailing):
        return None
    # Routing lengths: 11 (delivery point), 9 (ZIP+4), 5 (ZIP), 0 (none)
    for length in [11, 9, 5]:
        candidate = trailing[-length:] if len(trailing) >= length else None
        if candidate and candidate != "0" * length:
            return candidate
    return None


def decode_fadt(fadt: str) -> dict:
    """Full decode pipeline: FADT string → parsed payload dict."""
    fadt = fadt.strip().upper()
    if len(fadt) != 65:
        raise ValueError(f"FADT must be 65 characters, got {len(fadt)}")
    if not all(c in "FADT" for c in fadt):
        bad = set(fadt) - set("FADT")
        raise ValueError(f"Invalid FADT characters: {bad}")

    upper, lower    = fadt_to_bits(fadt)
    codewords       = bits_to_codewords(upper, lower)
    digit_str       = codewords_to_digit_string(codewords)
    payload         = parse_payload(digit_str)
    payload["fadt"] = fadt
    payload["raw_digits"] = digit_str
    return payload


# =============================================================================
# TOP-LEVEL PIPELINE
# =============================================================================

def process_image(image_path: str, debug: bool = False) -> dict:
    """
    Run the full IMb pipeline on a mailpiece photo.

    Args:
        image_path: Path to input image (JPEG, PNG, etc.)
        debug:      If True, saves intermediate images alongside input

    Returns:
        Parsed payload dict with all IMb fields
    """
    print(f"\n[IMb Pipeline] Input: {image_path}")

    # Stage 1 — load
    gray = load_gray(image_path)
    print(f"  Loaded: {gray.shape[1]}w × {gray.shape[0]}h px")

    # Stage 2 — Radon deskew (MUST come before Otsu)
    gray_aligned, angle = radon_deskew(gray)
    print(f"  Radon correction: {angle:+.2f}°" if angle else "  Radon: no correction needed")
    if debug:
        cv2.imwrite("debug_01_radon_aligned.png", gray_aligned)

    # Stage 3 — Otsu binarization (on aligned image)
    binary = otsu_binarize(gray_aligned)
    if debug:
        cv2.imwrite("debug_02_otsu_binary.png", binary)

    # Stage 4 — Barcode crop
    cropped = find_barcode_region(binary, gray_aligned)
    print(f"  Barcode region: {cropped.shape[1]}w × {cropped.shape[0]}h px")
    if debug:
        cv2.imwrite("debug_03_cropped.png", cropped)

    # Stage 5 — Upscale + re-binarize for bar segmentation
    bar_binary = prepare_for_classification(cropped)
    print(f"  Upscaled binary: {bar_binary.shape[1]}w × {bar_binary.shape[0]}h px")
    if debug:
        cv2.imwrite("debug_04_bar_binary.png", bar_binary)

    # Stage 6 — FADT classification (programmatic, not vision-model)
    fadt = classify_bars(bar_binary)
    print(f"  FADT ({len(fadt)} chars): {fadt}")

    # Stage 7 — Decode
    result = decode_fadt(fadt)
    return result


def print_result(r: dict) -> None:
    print()
    print("=" * 40)
    print("  IMb Decode Result")
    print("=" * 40)
    print(f"  FADT string  : {r['fadt']}")
    print(f"  Raw digits   : {r['raw_digits']}")
    print(f"  Barcode ID   : {r['barcode_id']}")
    print(f"  STID         : {r['stid']}")
    if r['mid_6']:
        print(f"  MID (6-digit): {r['mid_6']}")
    if r['mid_9']:
        print(f"  MID (9-digit): {r['mid_9']}")
    print(f"  Serial       : {r['serial']}")
    print(f"  Routing ZIP  : {r['routing'] or 'none'}")
    print("=" * 40)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_path> [--debug]")
        print("  --debug   Save intermediate images for each pipeline stage")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_mode = "--debug" in sys.argv

    try:
        result = process_image(image_path, debug=debug_mode)
        print_result(result)
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
