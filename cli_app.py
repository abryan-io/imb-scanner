"""
USPS Intelligent Mail Barcode (IMb) Decoder
============================================
Full pipeline:
    1. Load image & grayscale
    2. Multi-strategy barcode region detection
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

import logging
import sys
import numpy as np
import cv2
import intelligent_mail_barcode as imb

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE 1 — IMAGE LOADING & GRAYSCALE
# =============================================================================

def load_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =============================================================================
# STAGE 2 — BARCODE REGION DETECTION (multi-strategy)
# =============================================================================

def detect_barcode_region(gray: np.ndarray):
    """
    Locate the IMb barcode band using vertical edge density (Sobel-X).
    Returns (x, y, w, h) or None. This is the legacy single-region API.
    """
    candidates = detect_barcode_candidates(gray)
    if candidates:
        return candidates[0]
    return None


def _expand_peak_to_region(row_energy, edge, peak_row, H, W, threshold_frac=0.40):
    """Expand a peak row into a candidate region. Returns (x, y, w, h) or None."""
    threshold = row_energy[peak_row] * threshold_frac

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

    col_energy = edge[top:bot, :].sum(axis=0)
    col_max = col_energy.max()
    if col_max == 0:
        return None
    active = np.where(col_energy > col_max * 0.15)[0]
    if len(active) == 0:
        return None

    xl, xr = int(active[0]), int(active[-1])
    bw = xr - xl

    if bw < 80 or bw / max(bot - top, 1) < 3.0:
        return None

    return (xl, top, bw, bot - top)


def detect_barcode_candidates(gray: np.ndarray, max_candidates: int = 10):
    """
    Find multiple candidate barcode regions ranked by IMB likelihood.

    Instead of just the global Sobel-X peak, finds all significant local peaks
    in the row-energy profile and filters by IMB-plausible dimensions.

    Returns list of (x, y, w, h) tuples, best candidates first.
    """
    H, W = gray.shape
    edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    row_energy = edge.sum(axis=1)
    ks = max(3, H // 60)
    row_energy_smooth = np.convolve(row_energy, np.ones(ks) / ks, mode='same')

    # Find local maxima in row energy profile
    peaks = _find_energy_peaks(row_energy_smooth, min_distance=max(10, H // 40))

    candidates = []
    seen_rows = set()

    for peak_row in peaks:
        # Skip if too close to an already-found region
        if any(abs(peak_row - s) < 20 for s in seen_rows):
            continue

        region = _expand_peak_to_region(
            row_energy_smooth, edge, peak_row, H, W, threshold_frac=0.40
        )
        if region is None:
            continue

        x, y, w, h = region
        seen_rows.add(y + h // 2)

        # Score: prefer IMB-like aspect ratios (very wide, very short)
        # IMB at 300 DPI: ~750-1100px wide, ~15-50px tall
        aspect = w / max(h, 1)
        # Ideal height range at 300 DPI: 15-80px (with padding)
        height_score = 1.0
        if h > 150:
            height_score = max(0.1, 150 / h)
        elif h < 8:
            height_score = 0.1

        # Ideal aspect ratio > 10 for IMB
        aspect_score = min(aspect / 10.0, 2.0)

        # Energy density in the region
        region_energy = row_energy_smooth[y:y+h].mean()
        energy_score = region_energy / (row_energy_smooth.max() + 1e-9)

        score = aspect_score * height_score * energy_score
        candidates.append((score, (x, y, w, h)))

    # Sort by score descending
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [c[1] for c in candidates[:max_candidates]]


def _find_energy_peaks(energy, min_distance=20):
    """Find local maxima in a 1D energy profile."""
    peaks = []
    n = len(energy)
    if n < 3:
        return peaks

    global_max = energy.max()
    if global_max == 0:
        return peaks

    # Threshold: only consider peaks above 15% of global max
    threshold = global_max * 0.15

    for i in range(1, n - 1):
        if energy[i] < threshold:
            continue
        # Check if local maximum within min_distance window
        window_start = max(0, i - min_distance)
        window_end = min(n, i + min_distance + 1)
        if energy[i] == energy[window_start:window_end].max():
            peaks.append(i)

    # Sort by energy value descending
    peaks.sort(key=lambda p: energy[p], reverse=True)
    return peaks


def detect_barcode_candidates_morphological(gray: np.ndarray):
    """
    Fallback barcode detection using morphological operations.

    Targets thin vertical bar structures and groups them into
    barcode-like clusters.

    Returns list of (x, y, w, h) tuples.
    """
    H, W = gray.shape

    # Adaptive threshold to handle varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 5
    )

    # Morphological: enhance vertical bars, suppress horizontal structures
    # Vertical kernel to keep bars, horizontal kernel to remove text
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, H // 100)))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)

    # Dilate horizontally to connect nearby bars into a barcode band
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, W // 80), 1))
    connected = cv2.dilate(binary, horiz_kernel, iterations=1)

    # Find contours of the connected bands
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # IMB should be very wide relative to height
        if aspect > 5 and w > 200 and h > 5 and h < max(200, H // 4):
            score = aspect * (w / W)  # Prefer wider, higher-aspect regions
            candidates.append((score, (x, y, w, h)))

    candidates.sort(key=lambda c: c[0], reverse=True)
    return [c[1] for c in candidates[:10]]


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
# ROBUST DECODE — try a single region with multiple binarization strategies
# =============================================================================

def try_decode_region(gray: np.ndarray, region: tuple, collect_all: bool = False):
    """
    Attempt to decode IMB(s) from a specific region of a grayscale image.

    Tries multiple binarization strategies (Otsu, adaptive, CLAHE+Otsu).

    If collect_all=False, returns the first CRC-valid decode dict or None.
    If collect_all=True, returns a list of all unique CRC-valid decodes.
    """
    H, W = gray.shape
    x, y, w, h = region

    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    crop = gray[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 20:
        return [] if collect_all else None

    # Generate multiple binary images to try
    binaries = []

    # Strategy 1: Otsu (original approach)
    _, bin_otsu = cv2.threshold(crop, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_otsu)

    # Strategy 2: CLAHE enhanced + Otsu
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(crop)
    _, bin_clahe = cv2.threshold(enhanced, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_clahe)

    # Strategy 3: Adaptive Gaussian threshold
    block_size = max(11, (min(crop.shape) // 4) | 1)  # ensure odd
    bin_adapt = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )
    binaries.append(bin_adapt)

    # Strategy 4: Inverted image + Otsu (for dark backgrounds)
    crop_inv = 255 - crop
    _, bin_inv = cv2.threshold(crop_inv, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_inv)

    if not collect_all:
        for binary in binaries:
            result = _try_binary_decode(binary)
            if result is not None:
                return result
        return None
    else:
        all_results = []
        seen_tracking = set()
        for binary in binaries:
            results = _try_binary_decode_all(binary)
            for r in results:
                key = r['tracking']
                if key not in seen_tracking:
                    seen_tracking.add(key)
                    all_results.append(r)
        return all_results


def _try_binary_decode(binary: np.ndarray) -> dict:
    """Try to decode from a single binary image. Returns result dict or None."""
    scale = 4
    ch, cw = binary.shape
    up = cv2.resize(binary, (cw * scale, ch * scale),
                    interpolation=cv2.INTER_NEAREST)

    bar_runs = find_bar_runs(up)

    # If we have way more than 65 bars, try sliding windows of consecutive bars
    if len(bar_runs) > 85:
        result = _try_sliding_bar_windows(up, bar_runs)
        if result is not None:
            return result

    bar_runs_filtered = filter_to_65_bars(bar_runs, up.shape[1])

    if len(bar_runs_filtered) != 65:
        return None

    fadt = classify_bars_fadt(up, bar_runs_filtered)
    result = imb.decode(fadt)
    if result is not None and result.get('crc_ok'):
        result['fadt'] = fadt
        return result

    return None


def _try_binary_decode_all(binary: np.ndarray) -> list:
    """Try to decode ALL valid IMBs from a binary image. Returns list of results."""
    scale = 4
    ch, cw = binary.shape
    up = cv2.resize(binary, (cw * scale, ch * scale),
                    interpolation=cv2.INTER_NEAREST)

    bar_runs = find_bar_runs(up)
    results = []
    seen_tracking = set()

    # If many bars, try sliding windows to find multiple barcodes
    if len(bar_runs) > 75:
        for start_idx in range(0, max(1, len(bar_runs) - 64)):
            window = bar_runs[start_idx:start_idx + 65]
            if len(window) != 65:
                continue
            # Check spacing consistency
            pitches = [(window[i+1][0] - window[i][0]) for i in range(64)]
            med_pitch = np.median(pitches)
            if med_pitch < 1:
                continue
            deviations = [abs(p - med_pitch) / med_pitch for p in pitches]
            if max(deviations) > 0.8:
                continue

            fadt = classify_bars_fadt(up, window)
            result = imb.decode(fadt)
            if result is not None and result.get('crc_ok'):
                key = result['tracking']
                if key not in seen_tracking:
                    seen_tracking.add(key)
                    result['fadt'] = fadt
                    results.append(result)

    # Also try standard filter approach
    bar_runs_filtered = filter_to_65_bars(bar_runs, up.shape[1])
    if len(bar_runs_filtered) == 65:
        fadt = classify_bars_fadt(up, bar_runs_filtered)
        result = imb.decode(fadt)
        if result is not None and result.get('crc_ok'):
            key = result['tracking']
            if key not in seen_tracking:
                seen_tracking.add(key)
                result['fadt'] = fadt
                results.append(result)

    return results


def _try_sliding_bar_windows(up: np.ndarray, bar_runs: list) -> dict:
    """
    When we have many more than 65 bars (e.g. wide region with text + barcode),
    try sliding windows of 65 consecutive bars looking for consistent pitch.
    """
    n = len(bar_runs)
    if n < 65:
        return None

    best_result = None
    best_score = -1

    for start in range(0, n - 64):
        window = bar_runs[start:start + 65]

        # Check pitch consistency — real IMB bars have very regular spacing
        pitches = [(window[i+1][0] - window[i][0]) for i in range(64)]
        med_pitch = np.median(pitches)
        if med_pitch < 1:
            continue
        deviations = [abs(p - med_pitch) / med_pitch for p in pitches]
        max_dev = max(deviations)

        # Skip windows with highly irregular spacing
        if max_dev > 0.6:
            continue

        # Spacing consistency score
        score = 1.0 / (1.0 + np.mean(deviations))

        fadt = classify_bars_fadt(up, window)
        result = imb.decode(fadt)
        if result is not None and result.get('crc_ok'):
            if score > best_score:
                best_score = score
                result['fadt'] = fadt
                best_result = result

    return best_result


# =============================================================================
# ROBUST SCAN — multi-strategy IMB detection
# =============================================================================

def scan_image_robust(gray: np.ndarray) -> dict:
    """
    Robust IMB scanning that tries multiple detection strategies.
    Returns the first decoded result dict or None.
    """
    results = scan_image_robust_all(gray)
    return results[0] if results else None


def scan_image_robust_all(gray: np.ndarray) -> list:
    """
    Robust IMB scanning that returns ALL valid IMB decodes found.

    Strategies are tried in order of speed. Fast strategies (Sobel-X)
    run first; expensive strategies (strip scan) only run if needed.

    Returns list of decoded result dicts (may contain multiple barcodes).
    """
    H, W = gray.shape
    all_results = []
    seen_tracking = set()

    def _collect(results_or_result):
        if results_or_result is None:
            return
        items = results_or_result if isinstance(results_or_result, list) else [results_or_result]
        for r in items:
            key = r['tracking']
            if key not in seen_tracking:
                seen_tracking.add(key)
                all_results.append(r)

    # Strategy 1: Multi-candidate Sobel-X on original image (fast)
    candidates = detect_barcode_candidates(gray)
    for region in candidates:
        _collect(try_decode_region(gray, region, collect_all=True))

    # Strategy 2: CLAHE-enhanced image (fast)
    if not all_results:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        candidates = detect_barcode_candidates(enhanced)
        for region in candidates:
            _collect(try_decode_region(enhanced, region, collect_all=True))

    # Strategy 3: Morphological detection (moderate speed)
    if not all_results:
        morph_candidates = detect_barcode_candidates_morphological(gray)
        for region in morph_candidates:
            _collect(try_decode_region(gray, region, collect_all=True))

    # Strategy 4: Inverted image (for dark backgrounds)
    if not all_results and gray.mean() < 160:
        inverted = 255 - gray
        candidates = detect_barcode_candidates(inverted)
        for region in candidates:
            _collect(try_decode_region(inverted, region, collect_all=True))

    # Strategy 5: Horizontal strip scan (slow, last resort)
    if not all_results:
        result = _strip_scan(gray)
        _collect(result)

    return all_results


def _strip_scan(gray: np.ndarray) -> dict:
    """
    Scan horizontal strips across the image looking for IMB.

    This catches barcodes that don't produce a dominant Sobel-X peak
    because surrounding content (tables, images) has higher edge energy.
    """
    H, W = gray.shape
    # IMB is ~15-50px tall at 300 DPI; scan with overlapping strips
    strip_heights = [60, 100, 40]
    step = 20

    for strip_h in strip_heights:
        for y in range(0, H - strip_h, step):
            strip = gray[y:y + strip_h, :]
            # Quick check: does this strip have enough vertical edge energy?
            edge = np.abs(cv2.Sobel(strip, cv2.CV_32F, 1, 0, ksize=3))
            col_energy = edge.sum(axis=0)
            # Need a reasonable spread of energy across columns
            active_cols = np.sum(col_energy > col_energy.max() * 0.1)
            if active_cols < W * 0.2:
                continue

            # Try to decode this strip directly
            result = _try_strip_decode(strip)
            if result is not None:
                return result

    return None


def _try_strip_decode(strip_gray: np.ndarray) -> dict:
    """Try to decode an IMB from a narrow strip of the image."""
    # Try Otsu and adaptive binarization
    _, bin_otsu = cv2.threshold(strip_gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    result = _try_binary_decode(bin_otsu)
    if result is not None:
        return result

    # Try adaptive threshold
    block_size = max(11, (min(strip_gray.shape) // 3) | 1)
    bin_adapt = cv2.adaptiveThreshold(
        strip_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )
    result = _try_binary_decode(bin_adapt)
    if result is not None:
        return result

    return None


# =============================================================================
# FULL PIPELINE
# =============================================================================

def process_image(image_path: str, debug: bool = False) -> dict:
    """
    Run the full IMb pipeline on a mailpiece photo.

    Returns dict with tracking, routing, barcode_id, service_type,
    mailer_id, serial, crc_ok, fadt.
    """
    logger.info("pipeline start | input=%s", image_path)

    # Stage 1 — load
    gray = load_gray(image_path)
    H, W = gray.shape
    logger.info("image loaded | width=%d height=%d", W, H)

    # Try robust scan first
    result = scan_image_robust(gray)
    if result is not None:
        logger.info("decode success | fadt_len=%d crc_ok=%s tracking=%s",
                    len(result['fadt']), result['crc_ok'], result.get('tracking'))
        return result

    logger.warning("no IMB decoded | input=%s", image_path)
    raise ValueError("No IMB barcode detected or decoded from this image.")


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
    from logging_config import setup_logging

    if len(sys.argv) < 2:
        print("Usage: python cli_app.py <image_path> [--debug]")
        print("  --debug   Enable DEBUG-level logging")
        sys.exit(1)

    image_path = sys.argv[1]
    debug_mode = "--debug" in sys.argv
    setup_logging(debug=debug_mode)

    try:
        result = process_image(image_path, debug=debug_mode)
        print_result(result)
    except (ValueError, FileNotFoundError) as e:
        logger.exception("pipeline failed | input=%s", image_path)
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
