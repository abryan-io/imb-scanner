"""
USPS Intelligent Mail Barcode (IMb) Decoder
============================================
Full pipeline:
    1. Load image & grayscale
    2. Sobel-X edge density → locate barcode band
    3. Crop barcode region
    4. Otsu binarization + upscale
    5. Bar segmentation with outlier filtering → 65 bar runs
    6. Largest-gap clustering → FADT string
    7. Decode via pyimb (FADT → tracking + routing)

Usage:
    python app.py <image_path>
    python app.py envelope.jpg --debug

Dependencies:
    pip install opencv-python-headless numpy
"""

import sys
import numpy as np
import cv2
import intelligent_mail_barcode as imb


# =============================================================================
# STAGE 1 — IMAGE LOADING & GRAYSCALE
# =============================================================================

def load_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =============================================================================
# STAGE 2 — SOBEL-X BARCODE REGION DETECTION
# =============================================================================

def detect_barcode_region(gray: np.ndarray):
    """
    Locate the IMb barcode band using vertical edge density (Sobel-X).

    IMb is a dense forest of vertical bars — the row with peak vertical-edge
    energy corresponds to the barcode. We expand up/down while energy stays
    above 40% of peak, then find horizontal extent.

    Returns (x, y, w, h) or None if no suitable region found.
    """
    H, W = gray.shape

    edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    # Row energy profile
    row_energy = edge.sum(axis=1)
    ks = max(3, H // 60)
    row_energy = np.convolve(row_energy, np.ones(ks) / ks, mode='same')

    peak_row = int(np.argmax(row_energy))
    threshold = row_energy[peak_row] * 0.40

    # Expand band vertically
    top = peak_row
    while top > 0 and row_energy[top - 1] > threshold:
        top -= 1
    bot = peak_row
    while bot < H - 1 and row_energy[bot + 1] > threshold:
        bot += 1

    pad_v = max(4, (bot - top) // 2)
    top = max(0, top - pad_v)
    bot = min(H - 1, bot + pad_v)

    if bot - top < 6:
        return None

    # Horizontal extent
    col_energy = edge[top:bot, :].sum(axis=0)
    active = np.where(col_energy > col_energy.max() * 0.15)[0]
    if len(active) == 0:
        return None

    xl, xr = int(active[0]), int(active[-1])
    bw = xr - xl

    if bw < 80 or bw / max(bot - top, 1) < 3.0:
        return None

    return xl, top, bw, bot - top


# =============================================================================
# STAGE 3 — BAR SEGMENTATION
# =============================================================================

def find_bar_runs(binary: np.ndarray):
    """
    Find contiguous vertical bar runs from column projection of a binary image.
    Returns list of (start_col, end_col) tuples.
    """
    _, uw = binary.shape
    col_sum = binary.sum(axis=0)
    kernel = np.ones(3, dtype=np.float32) / 3.0
    col_smooth = np.convolve(col_sum.astype(np.float32), kernel, mode='same')

    ink_thr = col_smooth.max() * 0.05
    in_bar = col_smooth > ink_thr

    bar_runs = []
    i = 0
    while i < uw:
        if in_bar[i]:
            j = i
            while j < uw and in_bar[j]:
                j += 1
            bar_runs.append((i, j - 1))
            i = j
        else:
            i += 1

    return bar_runs


def filter_to_65_bars(bar_runs, image_width):
    """
    Filter bar runs to exactly 65 by removing width/spacing outliers.

    Handles text artifacts (rotated stamps, printed text) that appear near
    the barcode and get picked up as extra bars. These are typically much
    wider than real bars or separated by large gaps.
    """
    if len(bar_runs) == 65:
        return bar_runs

    if len(bar_runs) > 65:
        widths = [e - s + 1 for s, e in bar_runs]
        median_w = float(np.median(widths))

        # Remove bars much wider than median (text artifacts)
        filtered = [(s, e) for s, e in bar_runs if (e - s + 1) <= median_w * 2.5]

        # If still too many, trim from edges based on inter-bar gap size
        if len(filtered) > 65:
            centers = [(s + e) / 2 for s, e in filtered]
            keep = list(range(len(filtered)))
            while len(keep) > 65:
                left_gap = centers[keep[1]] - centers[keep[0]]
                right_gap = centers[keep[-1]] - centers[keep[-2]]
                if left_gap > right_gap:
                    keep.pop(0)
                else:
                    keep.pop()
            filtered = [filtered[i] for i in keep]

        bar_runs = filtered

    # Merge close bars if still over 65
    if len(bar_runs) > 65:
        pitch = (bar_runs[-1][1] - bar_runs[0][0]) / 65
        merged = [bar_runs[0]]
        for run in bar_runs[1:]:
            prev_center = (merged[-1][0] + merged[-1][1]) / 2
            curr_center = (run[0] + run[1]) / 2
            if curr_center - prev_center < pitch * 0.55:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        bar_runs = merged

    return bar_runs


# =============================================================================
# STAGE 4 — FADT CLASSIFICATION (largest-gap clustering)
# =============================================================================

def classify_bars_fadt(binary: np.ndarray, bar_runs: list) -> str:
    """
    Classify each bar as F/A/D/T by measuring top and bottom pixel rows,
    then splitting into two clusters via the largest gap in sorted values.

    This is more robust than fixed-zone-thirds because it adapts to the
    actual bar geometry in each image.
    """
    uh, _ = binary.shape

    measurements = []
    for (s, e) in bar_runs:
        col_strip = binary[:, s:e + 1].max(axis=1)
        rows = np.where(col_strip > 127)[0]
        if len(rows):
            measurements.append((int(rows[0]), int(rows[-1])))
        else:
            measurements.append((uh // 4, 3 * uh // 4))

    tops = [m[0] for m in measurements]
    bottoms = [m[1] for m in measurements]

    def largest_gap_threshold(vals):
        s = sorted(vals)
        gaps = [s[k + 1] - s[k] for k in range(len(s) - 1)]
        if not gaps or max(gaps) < 2:
            return float(np.mean(s))
        sp = gaps.index(max(gaps))
        return (s[sp] + s[sp + 1]) / 2.0

    top_thr = largest_gap_threshold(tops)
    bot_thr = largest_gap_threshold(bottoms)

    fadt = []
    for (bt, bb) in measurements:
        has_asc = bt < top_thr
        has_desc = bb > bot_thr
        if has_asc and has_desc:
            fadt.append('F')
        elif has_asc:
            fadt.append('A')
        elif has_desc:
            fadt.append('D')
        else:
            fadt.append('T')

    return ''.join(fadt)


# =============================================================================
# FULL PIPELINE
# =============================================================================

def process_image(image_path: str, debug: bool = False) -> dict:
    """
    Run the full IMb pipeline on a mailpiece photo.

    Returns dict with tracking, routing, barcode_id, service_type,
    mailer_id, serial, crc_ok, fadt.
    """
    print(f"\n[IMb Pipeline] Input: {image_path}")

    # Stage 1 — load
    gray = load_gray(image_path)
    H, W = gray.shape
    print(f"  Loaded: {W}w x {H}h px")

    # Stage 2 — detect barcode region via Sobel-X
    region = detect_barcode_region(gray)
    if region is None:
        raise ValueError("No barcode region detected in image.")

    x, y, w, h = region
    print(f"  Barcode region: x={x} y={y} {w}w x {h}h px")

    # Crop with padding
    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    crop = gray[y1:y2, x1:x2]

    if debug:
        cv2.imwrite("debug_01_crop.png", crop)

    # Stage 3 — binarize and upscale
    _, binary = cv2.threshold(crop, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    scale = 4
    ch, cw = binary.shape
    up = cv2.resize(binary, (cw * scale, ch * scale),
                    interpolation=cv2.INTER_NEAREST)

    if debug:
        cv2.imwrite("debug_02_binary_upscaled.png", up)

    # Stage 4 — find bars
    bar_runs = find_bar_runs(up)
    raw_count = len(bar_runs)
    print(f"  Raw bar count: {raw_count}")

    bar_runs = filter_to_65_bars(bar_runs, up.shape[1])

    if len(bar_runs) != 65:
        raise ValueError(
            f"Expected 65 bars, found {len(bar_runs)} "
            f"(raw: {raw_count}). Image may be blurry or partially cropped."
        )

    # Stage 5 — FADT classification
    fadt = classify_bars_fadt(up, bar_runs)
    print(f"  FADT ({len(fadt)} chars): {fadt}")

    # Stage 6 — decode via pyimb
    result = imb.decode(fadt)
    if result is None:
        raise ValueError(
            f"FADT decode failed — CRC mismatch or invalid codewords. "
            f"FADT: {fadt}"
        )

    print(f"  CRC OK: {result['crc_ok']}")
    result['fadt'] = fadt
    return result


def print_result(r: dict) -> None:
    tracking = r['tracking']
    routing = r.get('routing', '')

    print()
    print("=" * 50)
    print("  IMb Decode Result")
    print("=" * 50)
    print(f"  FADT string   : {r['fadt']}")
    print(f"  Tracking code : {tracking}")
    print(f"  Routing code  : {routing or 'none'}")
    print(f"  Full number   : {tracking}{routing}")
    print(f"  Barcode ID    : {r['barcode_id']}")
    print(f"  Service Type  : {r['service_type']}")
    print(f"  Mailer ID     : {r['mailer_id']}")
    print(f"  Serial        : {r['serial']}")
    print(f"  CRC OK        : {r['crc_ok']}")
    print("=" * 50)


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
