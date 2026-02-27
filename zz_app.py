"""
USPS Intelligent Mail Barcode (IMB) Scanner
============================================
Deterministic pipeline — no AI, no API calls.

Pipeline:
  1. Sobel-X edge density → locate barcode band
  2. Crop, binarize, upscale
  3. Bar segmentation with outlier filtering → 65 bar runs
  4. Largest-gap clustering → FADT string (65 chars)
  5. Decode FADT via pyimb → tracking + routing
  6. Parse into USPS components (BarcodeID, STID, MID, Serial, Routing)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import intelligent_mail_barcode as imb
from stid_table import lookup as stid_lookup, describe as stid_describe

# ─── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="USPS IMB Scanner",
    page_icon="📮",
    layout="centered",
)

st.title("📮 USPS Intelligent Mail Barcode Scanner")
st.caption("Deterministic IMB detection · no AI · just pixels and math")


# ─── Image Processing ───────────────────────────────────────────────────────────

def to_gray(img_array: np.ndarray) -> np.ndarray:
    """RGB/BGR → grayscale."""
    if img_array.ndim == 3:
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    return img_array.copy()


def detect_barcode_region(gray: np.ndarray):
    """
    Locate the IMB barcode band via Sobel-X vertical edge density.
    Returns (x, y, w, h) or None.
    """
    H, W = gray.shape

    edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    row_energy = edge.sum(axis=1)
    ks = max(3, H // 60)
    row_energy_smooth = np.convolve(row_energy, np.ones(ks) / ks, mode='same')

    peak_row = int(np.argmax(row_energy_smooth))
    threshold = row_energy_smooth[peak_row] * 0.40

    top = peak_row
    while top > 0 and row_energy_smooth[top - 1] > threshold:
        top -= 1
    bot = peak_row
    while bot < H - 1 and row_energy_smooth[bot + 1] > threshold:
        bot += 1

    pad_v = max(4, (bot - top) // 2)
    top = max(0, top - pad_v)
    bot = min(H - 1, bot + pad_v)

    if bot - top < 6:
        return None

    col_energy = edge[top:bot, :].sum(axis=0)
    active = np.where(col_energy > col_energy.max() * 0.15)[0]
    if len(active) == 0:
        return None

    xl, xr = int(active[0]), int(active[-1])
    bw = xr - xl

    if bw < 80 or bw / max(bot - top, 1) < 3.0:
        return None

    return xl, top, bw, bot - top


def find_bar_runs(binary: np.ndarray):
    """Find contiguous bar runs from column projection."""
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
    """Filter bar runs to exactly 65 by removing width/spacing outliers."""
    if len(bar_runs) == 65:
        return bar_runs

    if len(bar_runs) > 65:
        widths = [e - s + 1 for s, e in bar_runs]
        median_w = float(np.median(widths))

        # Remove bars much wider than median (text artifacts)
        filtered = [(s, e) for s, e in bar_runs if (e - s + 1) <= median_w * 2.5]

        # Trim from edges based on inter-bar gap size
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


def classify_bars_fadt(binary: np.ndarray, bar_runs: list) -> str:
    """Classify bars as F/A/D/T using largest-gap clustering."""
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


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def scan_image(pil_img: Image.Image):
    """
    Full pipeline: image → decoded IMB.

    Returns (result_dict | None, annotated_img, crop_img | None,
             fadt_str | None, method_str, debug_info)
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = to_gray(img_rgb)
    H, W = gray.shape
    debug_info = {}

    # Detect barcode region
    region = detect_barcode_region(gray)
    annotated = img_rgb.copy()

    if region is None:
        return None, annotated, None, None, "No IMB region detected", debug_info

    x, y, w, h = region
    debug_info['region'] = {'x': x, 'y': y, 'w': w, 'h': h}

    pad = 4
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
    cv2.putText(annotated, "IMB region", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 1)

    crop_rgb = img_rgb[y1:y2, x1:x2]
    crop_gray = gray[y1:y2, x1:x2]

    # Binarize and upscale
    _, binary = cv2.threshold(crop_gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    scale = 4
    ch, cw = binary.shape
    up = cv2.resize(binary, (cw * scale, ch * scale),
                    interpolation=cv2.INTER_NEAREST)

    # Find and filter bars
    bar_runs = find_bar_runs(up)
    raw_count = len(bar_runs)
    debug_info['raw_bar_count'] = raw_count

    bar_runs = filter_to_65_bars(bar_runs, up.shape[1])
    debug_info['filtered_bar_count'] = len(bar_runs)

    if len(bar_runs) != 65:
        return (None, annotated, crop_rgb, None,
                f"Found {len(bar_runs)} bars (raw: {raw_count}), expected 65",
                debug_info)

    # FADT classification
    fadt = classify_bars_fadt(up, bar_runs)
    debug_info['fadt'] = fadt

    # Decode
    result = imb.decode(fadt)
    if result is None:
        return (None, annotated, crop_rgb, fadt,
                "FADT extracted but decode failed (CRC error)", debug_info)

    result['fadt'] = fadt
    method = "Sobel-X + FADT + pyimb"
    return result, annotated, crop_rgb, fadt, method, debug_info


# ─── Streamlit UI ────────────────────────────────────────────────────────────────

tab_upload, tab_camera = st.tabs(["Upload", "Camera"])

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a mailpiece photo",
        type=["jpg", "jpeg", "png", "webp", "heic"],
        help="Clear, well-lit photos work best.",
    )

with tab_camera:
    camera_img = st.camera_input("Take a photo of a mailpiece")

# Determine which image to process
pil_img = None
if uploaded:
    pil_img = Image.open(uploaded)
elif camera_img:
    pil_img = Image.open(camera_img)

if pil_img:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with st.spinner("Scanning for IMB..."):
        result, annotated, crop, fadt, method, debug_info = scan_image(pil_img)

    with col2:
        st.subheader("Detection")
        st.image(annotated, use_container_width=True, caption=f"Method: {method}")

    if crop is not None:
        st.subheader("Detected IMB Region")
        crop_large = cv2.resize(crop, None, fx=4, fy=4,
                                interpolation=cv2.INTER_CUBIC)
        st.image(crop_large, use_container_width=True)

    if fadt is not None:
        st.subheader("Extracted FADT String (65 bars)")
        st.code(fadt, language=None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Full (F)", fadt.count('F'))
        c2.metric("Ascender (A)", fadt.count('A'))
        c3.metric("Descender (D)", fadt.count('D'))
        c4.metric("Tracker (T)", fadt.count('T'))

    # Debug expander
    with st.expander("Debug: edge map & row energy"):
        gray_dbg = to_gray(np.array(pil_img.convert("RGB")))
        sobelx = cv2.Sobel(gray_dbg, cv2.CV_32F, 1, 0, ksize=3)
        edge_vis = np.abs(sobelx)
        edge_vis = (edge_vis / edge_vis.max() * 255).astype(np.uint8)
        st.image(edge_vis, caption="Sobel-X edge map", use_container_width=True)
        row_e = edge_vis.astype(float).sum(axis=1)
        row_e /= row_e.max()
        st.line_chart(row_e, height=150)
        st.caption("Peak should align with barcode row")
        if debug_info:
            st.json(debug_info)

    st.divider()

    if result:
        tracking = result['tracking']
        routing = result.get('routing', '')
        full_number = tracking + routing

        st.success(f"Decoded: `{full_number}`")
        st.caption(f"CRC: {'PASS' if result['crc_ok'] else 'FAIL'}")

        st.subheader("IMB Components")

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Barcode ID", result['barcode_id'])
        col_b.metric("Service Type (STID)", result['service_type'])

        mid = result['mailer_id']
        mid_len = len(mid)
        col_c.metric("MID Length", f"{mid_len}-digit")

        col_d, col_e = st.columns(2)
        col_d.metric("Mailer ID (MID)", mid)
        col_e.metric("Serial Number", result['serial'])

        # STID lookup
        stid_info = stid_lookup(result['service_type'])
        if stid_info:
            st.info(f"STID {result['service_type']}: {stid_describe(result['service_type'])}")

        # Routing code
        st.subheader("Routing Code")
        if routing:
            rlen = len(routing)
            if rlen == 11:
                rtype = "ZIP+4+DPC"
                rcols = st.columns(3)
                rcols[0].metric("ZIP Code", routing[:5])
                rcols[1].metric("ZIP+4", routing[5:9])
                rcols[2].metric("Delivery Point", routing[9:11])
            elif rlen == 9:
                rtype = "ZIP+4"
                rcols = st.columns(2)
                rcols[0].metric("ZIP Code", routing[:5])
                rcols[1].metric("ZIP+4", routing[5:9])
            elif rlen == 5:
                rtype = "ZIP Code"
                st.metric("ZIP Code", routing)
            else:
                rtype = f"Unknown ({rlen} digits)"
                st.write(f"Raw: {routing}")
            st.write(f"**Type:** {rtype}")
        else:
            st.write("No routing code")

        # Tracking code breakdown
        with st.expander("Tracking code breakdown"):
            mid_end = 5 + mid_len
            st.markdown(
                f"`{tracking[0:2]}` **Barcode ID** · "
                f"`{tracking[2:5]}` **STID** · "
                f"`{tracking[5:mid_end]}` **MID** · "
                f"`{tracking[mid_end:20]}` **Serial**"
            )
            if routing:
                st.markdown(f"`{routing}` **Routing**")

        # Full JSON
        with st.expander("Full decoded output (JSON)"):
            st.json(result)
    else:
        st.error("Could not decode an IMB from this image.")
        if fadt and len(fadt) == 65:
            st.warning(
                "65 bar states were extracted (see FADT above) but the CRC "
                "check failed. A few bars may be misclassified. "
                "Try a higher-resolution or tighter-cropped photo."
            )
        else:
            n = debug_info.get('raw_bar_count', 0)
            if n:
                st.warning(f"Found {n} bars instead of 65. "
                           "Barcode may be cut off, blurry, or at an angle.")
            st.markdown("""
            **Tips for better results:**
            - Make sure the IMB is in focus and well-lit
            - Try cropping the image tighter around the barcode
            - Avoid extreme angles — straight-on works best
            - Higher resolution photos give more pixel data to work with
            """)

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("IMB Field Reference")
    st.markdown("""
    **Barcode ID** (2 digits)
    Identifies barcode type and usage.

    **Service Type ID / STID** (3 digits)
    Defines mail class + ancillary service.
    e.g., `300` = First-Class, no ancillary.

    **Mailer ID / MID** (6 or 9 digits)
    Unique identifier assigned by USPS.
    - 6-digit MID: first digit is 9
    - 9-digit MID: first digit is 0-8

    **Serial Number** (6 or 9 digits)
    Mailer-assigned piece identifier.

    **Routing Code** (optional)
    | Length | Type |
    |--------|------|
    | 0 | No routing |
    | 5 | ZIP Code |
    | 9 | ZIP+4 |
    | 11 | ZIP+4+DPC |
    """)

    st.divider()
    st.header("Pipeline")
    st.markdown("""
    1. Sobel-X edge density → barcode row
    2. Crop + Otsu binarize + 4x upscale
    3. Column projection → 65 bar runs
    4. Largest-gap clustering → FADT
    5. pyimb decode (CRC-11 validated)

    No AI · No API · Fully offline
    """)
