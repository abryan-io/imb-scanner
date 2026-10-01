"""
USPS Intelligent Mail Barcode (IMb) Decoder
============================================
Full pipeline:
    1. Load image & grayscale
    2. Normalize scale; deskew; try 0° and 90° orientations
    3. Multi-strategy barcode region detection
    4. Crop + multiple binarizations + upscale
    5. Bar segmentation with outlier filtering → 65 bar runs
    6. Tilt-tolerant FADT classification with per-bar confidence
    7. Decode via pyimb (FADT → tracking + routing), both reading directions
    8. If nothing decodes: CRC-guided correction of low-confidence bars

Usage:
    python cli_app.py <image_path>
    python cli_app.py envelope.jpg --debug

Dependencies:
    pip install opencv-python-headless numpy
"""

import itertools
import logging
import sys
from dataclasses import dataclass

import numpy as np
import cv2
import intelligent_mail_barcode as imb

logger = logging.getLogger(__name__)

# Larger inputs (e.g. 12MP phone photos) are downscaled so the pixel constants
# below, tuned for ~300 DPI renders, stay meaningful.
WORKING_MAX_SIDE = 3300
SKEW_MIN_DEG = 0.3
NOISE_SIGMA_DENOISE = 5.0

# Error correction: only reads with at most this many invalid codewords are
# repaired, using single-bit flips drawn from the weakest bar measurements.
EC_MAX_BAD_CODEWORDS = 2
EC_WEAK_BITS = 8
EC_MAX_NEAR_MISSES = 40


# =============================================================================
# STAGE 1 — IMAGE LOADING & GRAYSCALE
# =============================================================================

def load_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# =============================================================================
# STAGE 2 — SCALE & ORIENTATION NORMALIZATION
# =============================================================================

def normalize_scale(gray: np.ndarray) -> np.ndarray:
    long_side = max(gray.shape)
    if long_side <= WORKING_MAX_SIDE:
        return gray
    f = WORKING_MAX_SIDE / long_side
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def estimate_noise_sigma(gray: np.ndarray) -> float:
    """
    Gaussian noise sigma estimate (Immerkær 1996), measured only on the
    flattest 90% of pixels so text and bar edges don't read as noise.
    """
    if min(gray.shape) < 8:
        return 0.0
    step = 2 if min(gray.shape) > 1500 else 1
    f = gray[::step, ::step].astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    resp = np.abs(cv2.filter2D(f, -1, k))
    grad = np.abs(cv2.Sobel(f, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(f, cv2.CV_32F, 0, 1))
    flat = grad <= np.percentile(grad, 90)
    return float(resp[flat].mean() * np.sqrt(np.pi / 2) / 6)


def denoise_if_noisy(gray: np.ndarray) -> np.ndarray:
    sigma = estimate_noise_sigma(gray)
    if sigma <= NOISE_SIGMA_DENOISE:
        return gray
    logger.debug("denoise | sigma=%.1f", sigma)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate image rotation in degrees from the direction of strong
    near-vertical edges (barcode bars, text stems). Returns the angle to pass
    to rotate_image() to make those edges vertical; 0.0 if undetermined.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    agx = np.abs(gx)
    mask = (agx > 200) & (agx > 2.5 * np.abs(gy))
    if mask.sum() < 500:
        return 0.0
    angles = np.degrees(np.arctan(gy[mask] / gx[mask]))
    est = float(np.median(angles))
    for _ in range(2):
        near = angles[np.abs(angles - est) < 5]
        if len(near) < 500:
            break
        est = float(np.median(near))
    return est


def rotate_image(gray: np.ndarray, angle: float) -> np.ndarray:
    """Rotate by `angle` degrees (counter-clockwise), expanding the canvas."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _deskewed_views(gray: np.ndarray):
    angle = estimate_skew_angle(gray)
    if abs(angle) >= SKEW_MIN_DEG:
        logger.debug("deskew | angle=%.2f", angle)
        yield rotate_image(gray, angle)
    yield gray


# =============================================================================
# STAGE 3 — BARCODE REGION DETECTION (multi-strategy)
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
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, H // 400)))
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
# STAGE 4 — BAR SEGMENTATION
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


def _regular_windows(bar_runs, max_dev=0.8):
    """Yield (start, end) indices of 65-bar windows with consistent pitch."""
    if len(bar_runs) < 65:
        return
    starts = np.array([s for s, _ in bar_runs], dtype=np.float64)
    windows = np.lib.stride_tricks.sliding_window_view(np.diff(starts), 64)
    med = np.median(windows, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(windows - med[:, None]).max(axis=1) / med
    for i in np.flatnonzero((med >= 1) & (dev <= max_dev)):
        yield int(i), int(i) + 65


def _estimate_pitch(binary: np.ndarray, bar_runs: list):
    """
    Bar pitch in px from the autocorrelation of the column ink profile,
    restricted to bar-width runs so dark blobs don't dominate it.
    """
    widths = np.array([e - s + 1 for s, e in bar_runs])
    max_w = np.median(widths) * 2.5
    col_ink = binary.sum(axis=0).astype(np.float64)
    profile = np.zeros_like(col_ink)
    for (s, e), w in zip(bar_runs, widths):
        if w <= max_w:
            profile[s:e + 1] = col_ink[s:e + 1]
    profile -= profile.mean()
    n = len(profile)
    hi = n // 64
    if hi < 9:
        return None
    spec = np.fft.rfft(profile, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:hi + 2]
    lags = np.arange(8, hi + 1)
    local_max = (ac[lags] > ac[lags - 1]) & (ac[lags] >= ac[lags + 1])
    if not local_max.any():
        return None
    lag = int(lags[local_max][np.argmax(ac[lags][local_max])])
    denom = ac[lag - 1] - 2 * ac[lag] + ac[lag + 1]
    return lag + (0.5 * (ac[lag - 1] - ac[lag + 1]) / denom if denom else 0.0)


def _pitch_grid_windows(binary: np.ndarray, bar_runs: list, max_windows: int = 3):
    """
    Yield synthetic 65-bar runs laid on a regular grid fitted to the observed
    bar centers. Recovers barcodes whose column projection merges or splits
    bars (JPEG ringing, blur, broken bars), where run counting fails.
    """
    if len(bar_runs) < 20:
        return
    pitch = _estimate_pitch(binary, bar_runs)
    if pitch is None:
        return
    centers = np.array([(s + e) / 2 for s, e in bar_runs], dtype=np.float64)

    # Dominant phase: the grid offset that the most bar centers agree with
    phases = np.mod(centers, pitch)
    cand = np.linspace(0, pitch, 32, endpoint=False)
    d = np.abs(phases[None, :] - cand[:, None])
    d = np.minimum(d, pitch - d)
    offset, step = cand[np.argmax((d < pitch / 8).sum(axis=1))], pitch

    for _ in range(3):
        k = np.round((centers - offset) / step)
        inliers = np.abs(centers - (offset + step * k)) < step / 4
        if inliers.sum() < 20:
            return
        step, offset = np.polyfit(k[inliers], centers[inliers], 1)

    ks = np.unique(k[inliers]).astype(int)
    windows = sorted(
        ((np.count_nonzero((ks >= k0) & (ks < k0 + 65)), k0)
         for k0 in range(ks.min(), max(ks.min(), ks.max() - 64) + 1)),
        reverse=True,
    )
    half_w = max(1.0, 0.2 * step)
    width = binary.shape[1]
    for coverage, k0 in windows[:max_windows]:
        if coverage < 45:
            break
        xs = offset + step * np.arange(k0, k0 + 65)
        if xs[0] - half_w < 0 or xs[-1] + half_w >= width:
            continue
        yield [(int(round(x - half_w)), int(round(x + half_w))) for x in xs]


def _component_bar_windows(binary: np.ndarray, scale: int, max_lines: int = 3):
    """
    Yield (xs, runs_per_bar) for 65-bar windows of bar-shaped connected
    components that a common, possibly tilted, line passes through (the
    tracker band). Unlike column projection, this ignores text above or below
    a tilted barcode and dark blobs sharing its columns. Coordinates are
    multiplied by `scale` to match the upscaled binary.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    left, top, w, h, area = stats[1:].T
    bar_like = (h >= 3) & (h >= 1.2 * w) & (area >= 4)
    left, top, w, h = left[bar_like], top[bar_like], w[bar_like], h[bar_like]
    if len(left) < 65:
        return
    cx = left + w / 2.0
    bottom = top + h - 1
    remaining = np.ones(len(left), dtype=bool)

    for _ in range(max_lines):
        idx = np.flatnonzero(remaining)
        if len(idx) < 65:
            return
        best_count, best_slope, best_icpt = 0, 0.0, 0.0
        for slope in np.linspace(-0.12, 0.12, 97):
            lo = top[idx] - slope * cx[idx]
            hi = bottom[idx] - slope * cx[idx]
            events = np.concatenate([lo, hi])
            delta = np.r_[np.ones(len(lo)), -np.ones(len(hi))]
            order = np.lexsort((-delta, events))
            coverage = np.cumsum(delta[order])
            k = int(np.argmax(coverage))
            if coverage[k] > best_count:
                best_count, best_slope, best_icpt = coverage[k], slope, events[order][k]
        if best_count < 65:
            return

        crosses = ((top[idx] - best_slope * cx[idx] <= best_icpt)
                   & (bottom[idx] - best_slope * cx[idx] >= best_icpt))
        on_line = idx[crosses]
        remaining[on_line] = False
        on_line = on_line[np.argsort(cx[on_line])]
        gaps = np.diff(cx[on_line])
        for seg in np.split(on_line, np.flatnonzero(gaps > 2.5 * np.median(gaps)) + 1):
            for i in range(len(seg) - 64):
                win = seg[i:i + 65]
                xs = (cx[win] * scale).astype(np.float64)
                runs = [[(int(top[j]) * scale, int(bottom[j]) * scale + scale - 1)]
                        for j in win]
                yield xs, runs


# =============================================================================
# STAGE 5 — FADT CLASSIFICATION (tilt-tolerant, with confidence)
# =============================================================================

@dataclass
class BarRead:
    """Ascender/descender presence for 65 bars plus per-bit confidence.

    Confidence is the distance of a bar's measured extent from the class
    threshold, normalized by the separation between the two classes.
    """
    asc: np.ndarray
    desc: np.ndarray
    asc_conf: np.ndarray
    desc_conf: np.ndarray

    def fadt(self) -> str:
        return _bits_to_fadt(self.asc, self.desc)

    def flipped(self) -> "BarRead":
        """The same read as seen with the barcode rotated 180°."""
        return BarRead(self.desc[::-1].copy(), self.asc[::-1].copy(),
                       self.desc_conf[::-1].copy(), self.asc_conf[::-1].copy())


def _bits_to_fadt(asc, desc) -> str:
    return ''.join('TADF'[int(a) | int(d) << 1] for a, d in zip(asc, desc))


def _vertical_runs(binary: np.ndarray, bar_runs: list) -> list:
    """For each bar, the vertical ink runs (top, bottom) in its column strip."""
    uh = binary.shape[0]
    max_gap = max(2, int(uh * 0.05))
    bounds = np.array([(s, e + 1) for s, e in bar_runs]).ravel()
    ink = np.maximum.reduceat(binary > 127, bounds[bounds < binary.shape[1]], axis=1)
    out = []
    for i in range(len(bar_runs)):
        rows = np.flatnonzero(ink[:, 2 * i])
        if len(rows) == 0:
            out.append([(uh // 4, 3 * uh // 4)])
            continue
        breaks = np.flatnonzero(np.diff(rows) > max_gap)
        starts = np.r_[rows[0], rows[breaks + 1]]
        ends = np.r_[rows[breaks], rows[-1]]
        out.append(list(zip(starts.tolist(), ends.tolist())))
    return out


def _robust_line(xs: np.ndarray, ys: np.ndarray):
    slope, intercept = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + intercept)
    mad = np.median(np.abs(resid - np.median(resid)))
    inliers = np.abs(resid) <= max(3 * mad, 1.0)
    if inliers.sum() >= 10:
        slope, intercept = np.polyfit(xs[inliers], ys[inliers], 1)
    return slope, intercept


def _run_at(runs: list, y: float):
    """The run containing row y, else the nearest one."""
    return min(runs, key=lambda r: (max(r[0] - y, y - r[1], 0), -(r[1] - r[0])))


def _split_threshold(vals: np.ndarray):
    """Largest-gap split into two clusters. Returns (threshold, separation)."""
    s = np.sort(vals)
    gaps = np.diff(s)
    if len(gaps) == 0 or gaps.max() < 2:
        return float(s.mean()), 1.0
    k = int(np.argmax(gaps))
    sep = float(np.median(s[k + 1:]) - np.median(s[:k + 1]))
    return (s[k] + s[k + 1]) / 2.0, max(sep, 1.0)


def _classify_runs(xs: np.ndarray, runs_per_bar: list) -> BarRead:
    """
    Classify bars from their vertical ink runs. A line is fitted through the
    bar midpoints (the tracker band), so tops/bottoms are measured relative to
    the band rather than the image rows — this absorbs skew and perspective.
    The run crossing the band is used, ignoring stray text above/below.
    """
    longest = [max(r, key=lambda t: t[1] - t[0]) for r in runs_per_bar]
    mids = np.array([(t + b) / 2 for t, b in longest], dtype=np.float64)
    slope, intercept = _robust_line(xs, mids)
    center = slope * xs + intercept

    chosen = [_run_at(runs, c) for runs, c in zip(runs_per_bar, center)]
    trend = slope * xs
    tops = np.array([t for t, _ in chosen], dtype=np.float64) - trend
    bottoms = np.array([b for _, b in chosen], dtype=np.float64) - trend

    top_thr, top_sep = _split_threshold(tops)
    bot_thr, bot_sep = _split_threshold(bottoms)
    return BarRead(
        asc=tops < top_thr,
        desc=bottoms > bot_thr,
        asc_conf=np.abs(tops - top_thr) / top_sep,
        desc_conf=np.abs(bottoms - bot_thr) / bot_sep,
    )


def measure_bars(binary: np.ndarray, bar_runs: list) -> BarRead:
    xs = np.array([(s + e) / 2 for s, e in bar_runs], dtype=np.float64)
    return _classify_runs(xs, _vertical_runs(binary, bar_runs))


def classify_bars_fadt(binary: np.ndarray, bar_runs: list) -> str:
    """Classify each bar as F/A/D/T. See _classify_runs."""
    return measure_bars(binary, bar_runs).fadt()


# =============================================================================
# STAGE 6 — DECODE, ORIENTATION & ERROR CORRECTION
# =============================================================================

def _codeword_valid(code: int) -> bool:
    return code in imb.inverted or (code ^ 0x1fff) in imb.inverted


def _bad_codewords(fadt: str) -> list:
    return [i for i, c in enumerate(imb.unbar(fadt)) if not _codeword_valid(c)]


def _decode_fadt(fadt: str):
    result = imb.decode(fadt)
    if result is not None and result.get('crc_ok'):
        result['fadt'] = fadt
        return result
    return None


def _decode_read(read: BarRead, near_misses):
    """Decode in both reading directions; remember repairable failures."""
    oriented = (read, read.flipped())
    for r in oriented:
        result = _decode_fadt(r.fadt())
        if result is not None:
            return result
    if near_misses is not None:
        bad = min(len(_bad_codewords(r.fadt())) for r in oriented)
        if 0 < bad <= EC_MAX_BAD_CODEWORDS:
            near_misses.append((bad, read))
    return None


def _error_correct(read: BarRead):
    """
    Repair up to EC_MAX_BAD_CODEWORDS invalid codewords with one bit flip
    each. A flip is only considered if it is among the EC_WEAK_BITS
    lowest-confidence measurements and makes its codeword valid; the
    CRC-11 must then pass.
    """
    for r in (read, read.flipped()):
        codes = imb.unbar(r.fadt())
        bad = [i for i, c in enumerate(codes) if not _codeword_valid(c)]
        if not bad or len(bad) > EC_MAX_BAD_CODEWORDS:
            continue
        weakest = np.argsort(np.concatenate([r.asc_conf, r.desc_conf]))[:EC_WEAK_BITS]
        options = []
        for cw in bad:
            opts = []
            for k in weakest:
                bar, is_desc = int(k) % 65, k >= 65
                idx, bit = (imb.tableD if is_desc else imb.tableA)[bar]
                if idx == cw and _codeword_valid(codes[cw] ^ (1 << bit)):
                    opts.append(int(k))
            options.append(opts)
        if not all(options):
            continue
        for combo in itertools.product(*options):
            asc, desc = r.asc.copy(), r.desc.copy()
            for k in combo:
                bits = desc if k >= 65 else asc
                bits[k % 65] = not bits[k % 65]
            result = _decode_fadt(_bits_to_fadt(asc, desc))
            if result is not None:
                result['corrected_bars'] = len(combo)
                return result
    return None


# =============================================================================
# ROBUST DECODE — try a single region with multiple binarization strategies
# =============================================================================

def try_decode_region(gray: np.ndarray, region: tuple, collect_all: bool = False,
                      near_misses: list = None):
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

    # Strategy 4: Background-flattened + Otsu (uneven lighting, shadows).
    # A horizontal closing wider than the bar pitch erases the bars, leaving
    # the illumination to divide out.
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, crop.shape[0] // 2) | 1, 1))
    background = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, bg_kernel)
    flat = cv2.divide(crop, background, scale=255)
    _, bin_flat = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_flat)

    # Strategy 5: Inverted image + Otsu (for dark backgrounds)
    crop_inv = 255 - crop
    _, bin_inv = cv2.threshold(crop_inv, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaries.append(bin_inv)

    all_results = []
    seen_tracking = set()
    for binary in binaries:
        for r in _decode_binary(binary, near_misses, first_only=not collect_all):
            if r['tracking'] not in seen_tracking:
                seen_tracking.add(r['tracking'])
                all_results.append(r)
        if all_results and not collect_all:
            return all_results[0]
    return all_results if collect_all else None


def _decode_binary(binary: np.ndarray, near_misses: list = None,
                   first_only: bool = False, fallbacks: bool = True) -> list:
    """
    Decode all valid IMBs from a binary crop. Returns list of results.
    `fallbacks` enables the slower component and pitch-grid segmenters.
    """
    scale = 4
    ch, cw = binary.shape
    up = cv2.resize(binary, (cw * scale, ch * scale),
                    interpolation=cv2.INTER_NEAREST)

    bar_runs = find_bar_runs(up)
    results = []
    seen_tracking = set()

    def _add(result):
        if result is not None and result['tracking'] not in seen_tracking:
            seen_tracking.add(result['tracking'])
            results.append(result)

    # Many bars (barcode + text, or several barcodes): slide a 65-bar window
    if len(bar_runs) > 75:
        xs = np.array([(s + e) / 2 for s, e in bar_runs], dtype=np.float64)
        vruns = _vertical_runs(up, bar_runs)
        for i, j in _regular_windows(bar_runs):
            _add(_decode_read(_classify_runs(xs[i:j], vruns[i:j]), near_misses))
            if first_only and results:
                return results

    bar_runs_filtered = filter_to_65_bars(bar_runs, up.shape[1])
    if len(bar_runs_filtered) == 65:
        _add(_decode_read(measure_bars(up, bar_runs_filtered), near_misses))

    if not fallbacks:
        return results

    if not results:
        for xs, runs in _component_bar_windows(binary, scale):
            _add(_decode_read(_classify_runs(xs, runs), near_misses))
            if first_only and results:
                return results

    if not results:
        for grid_runs in _pitch_grid_windows(up, bar_runs):
            _add(_decode_read(measure_bars(up, grid_runs), near_misses))
            if first_only and results:
                break

    return results


# =============================================================================
# ROBUST SCAN — multi-strategy IMB detection
# =============================================================================

def scan_image_robust(gray: np.ndarray, diagnostics: dict = None) -> dict:
    """
    Robust IMB scanning that tries multiple detection strategies.
    Returns the first decoded result dict or None.
    """
    results = scan_image_robust_all(gray, diagnostics)
    return results[0] if results else None


def scan_image_robust_all(gray: np.ndarray, diagnostics: dict = None) -> list:
    """
    Robust IMB scanning that returns ALL valid IMB decodes found.

    The image is scaled to a working size, then scanned upright (deskewed and
    as-is) and, failing that, rotated 90°. 180° is handled by decoding each
    read in both directions. If no clean decode is found in an orientation,
    the best near-miss reads are error-corrected before moving on.

    If `diagnostics` is given, it receives `near_miss_fadt` on failure.

    Returns list of decoded result dicts (may contain multiple barcodes).
    """
    gray = denoise_if_noisy(normalize_scale(gray))
    near_misses = []

    # The brute-force strip scan only runs upright; it is the slowest strategy
    # and the 90° pass exists for sideways photos, not hard-to-locate codes.
    for base, strip_scan in ((gray, True), (cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE), False)):
        for view in _deskewed_views(base):
            results = _scan_view(view, near_misses, strip_scan)
            if results:
                return results
        result = _correct_near_misses(near_misses)
        if result is not None:
            logger.info("decoded with error correction | corrected_bars=%d",
                        result['corrected_bars'])
            return [result]

    if diagnostics is not None and near_misses:
        diagnostics['near_miss_fadt'] = min(near_misses, key=lambda m: m[0])[1].fadt()
    return []


def _correct_near_misses(near_misses: list):
    seen = set()
    for _, read in sorted(near_misses, key=lambda m: m[0]):
        fadt = read.fadt()
        if fadt in seen:
            continue
        seen.add(fadt)
        if len(seen) > EC_MAX_NEAR_MISSES:
            break
        result = _error_correct(read)
        if result is not None:
            return result
    return None


def _scan_view(gray: np.ndarray, near_misses: list, strip_scan: bool = True) -> list:
    """
    Strategies are tried in order of speed. Fast strategies (Sobel-X)
    run first; expensive strategies (strip scan) only run if needed.
    """
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
        _collect(try_decode_region(gray, region, collect_all=True, near_misses=near_misses))

    # Strategy 2: CLAHE-enhanced image (fast)
    if not all_results:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        candidates = detect_barcode_candidates(enhanced)
        for region in candidates:
            _collect(try_decode_region(enhanced, region, collect_all=True,
                                       near_misses=near_misses))

    # Strategy 3: Morphological detection (moderate speed)
    if not all_results:
        morph_candidates = detect_barcode_candidates_morphological(gray)
        for region in morph_candidates:
            _collect(try_decode_region(gray, region, collect_all=True,
                                       near_misses=near_misses))

    # Strategy 4: Inverted image (for dark backgrounds)
    if not all_results and gray.mean() < 160:
        inverted = 255 - gray
        candidates = detect_barcode_candidates(inverted)
        for region in candidates:
            _collect(try_decode_region(inverted, region, collect_all=True,
                                       near_misses=near_misses))

    # Strategy 5: Horizontal strip scan (slow, last resort)
    if not all_results and strip_scan:
        _collect(_strip_scan(gray, near_misses))

    return all_results


def _strip_scan(gray: np.ndarray, near_misses: list = None) -> dict:
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
            result = _try_strip_decode(strip, near_misses)
            if result is not None:
                return result

    return None


def _try_strip_decode(strip_gray: np.ndarray, near_misses: list = None) -> dict:
    """Try to decode an IMB from a narrow strip of the image."""
    _, bin_otsu = cv2.threshold(strip_gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    block_size = max(11, (min(strip_gray.shape) // 3) | 1)
    bin_adapt = cv2.adaptiveThreshold(
        strip_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, 5
    )
    for binary in (bin_otsu, bin_adapt):
        results = _decode_binary(binary, near_misses, first_only=True, fallbacks=False)
        if results:
            return results[0]
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
    if r.get('corrected_bars'):
        print(f"  Corrected bars: {r['corrected_bars']}")
    print("=" * 50)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from logging_config import setup_logging
    from failed_scan_store import record_failure

    load_dotenv()

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
    except ValueError as e:
        logger.exception("pipeline failed | input=%s", image_path)
        try:
            gray = load_gray(image_path)
            record_failure(gray, source=image_path, reason="no_decode_cli",
                           attempts=["scan_image_robust"])
        except Exception:
            pass
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error("input not found | path=%s", image_path)
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
