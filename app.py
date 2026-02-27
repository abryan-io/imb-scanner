"""
USPS Intelligent Mail Barcode (IMB) Scanner
============================================
Fully deterministic pipeline — no AI, no API calls.

Pipeline:
  1. Detect IMB region via Sobel-X edge density (vertical edge energy)
  2. Attempt direct decode via zxing-cpp on the crop
  3. If that fails: extract FADT bar states from pixel heights,
     render a perfect synthetic barcode, then decode the synthetic image
  4. Parse decoded string into USPS components (BarcodeID, STID, MID, Serial, Routing)
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

try:
    import zxingcpp
    ZXING_AVAILABLE = True
except ImportError:
    ZXING_AVAILABLE = False

try:
    import zxing as pyzxing_lib
    PYZXING_AVAILABLE = True
except ImportError:
    PYZXING_AVAILABLE = False

try:
    import intelligent_mail_barcode as pyimb
    PYIMB_AVAILABLE = True
except ImportError:
    PYIMB_AVAILABLE = False

try:
    import stid_table as stid_db
    STID_DB_AVAILABLE = True
except ImportError:
    STID_DB_AVAILABLE = False

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="USPS IMB Scanner",
    page_icon="📮",
    layout="centered",
)

st.title("📮 USPS Intelligent Mail Barcode Scanner")
st.caption("Deterministic IMB detection · no AI · just pixels and math · v4")

if not ZXING_AVAILABLE and not PYZXING_AVAILABLE and not PYIMB_AVAILABLE:
    st.error("No decoder available. Install at minimum: pip install zxing-cpp")
    st.stop()

# ─── IMB String Parser ────────────────────────────────────────────────────────

def parse_imb(raw_text: str) -> dict:
    digits = ''.join(filter(str.isdigit, raw_text))
    if len(digits) < 20:
        return {"error": f"Expected 20+ digits, got {len(digits)}. Raw: '{raw_text}'"}

    tracking = digits[:20]
    routing  = digits[20:]

    mid_pivot = int(tracking[5])
    if mid_pivot <= 8:
        mid, serial, mid_len = tracking[5:14], tracking[14:20], 9
    else:
        mid, serial, mid_len = tracking[5:11], tracking[11:20], 6

    rlen = len(routing)
    if rlen == 0:
        route_info = {"type": "No routing code"}
    elif rlen == 5:
        route_info = {"type": "ZIP Code", "zip5": routing}
    elif rlen == 9:
        route_info = {"type": "ZIP+4", "zip5": routing[:5], "plus4": routing[5:]}
    elif rlen == 11:
        route_info = {"type": "ZIP+4+DPC", "zip5": routing[:5],
                      "plus4": routing[5:9], "dpc": routing[9:]}
    else:
        route_info = {"type": f"Unexpected length ({rlen})", "raw": routing}

    return {
        "raw_decoded":   raw_text,
        "tracking_code": tracking,
        "barcode_id":    tracking[0:2],
        "stid":          tracking[2:5],
        "mid":           mid,
        "mid_length":    mid_len,
        "serial_number": serial,
        "routing":       route_info,
    }

# ─── Image Helpers ────────────────────────────────────────────────────────────

def to_gray(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)


def sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


# ─── Region Detection ─────────────────────────────────────────────────────────

def detect_imb_region(gray: np.ndarray):
    H, W = gray.shape
    edge = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))

    row_e = edge.sum(axis=1)
    ks    = max(3, H // 60)
    row_e = np.convolve(row_e, np.ones(ks) / ks, mode='same')

    peak = int(np.argmax(row_e))
    thr  = row_e[peak] * 0.40
    top  = peak
    while top > 0 and row_e[top - 1] > thr:
        top -= 1
    bot = peak
    while bot < H - 1 and row_e[bot + 1] > thr:
        bot += 1

    pad_v = max(4, (bot - top) // 2)
    top   = max(0, top - pad_v)
    bot   = min(H - 1, bot + pad_v)
    if bot - top < 6:
        return None

    col_e  = edge[top:bot, :].sum(axis=0)
    active = np.where(col_e > col_e.max() * 0.15)[0]
    if len(active) == 0:
        return None

    xl, xr = int(active[0]), int(active[-1])
    bw = xr - xl
    if bw < 80 or bw / max(bot - top, 1) < 3.0:
        return None

    return xl, top, bw, bot - top


# ─── Direct zxing-cpp Decode ──────────────────────────────────────────────────

def _probe_zxing_formats() -> dict:
    """Introspect installed zxingcpp for available BarcodeFormat names."""
    fmt = zxingcpp.BarcodeFormat
    return {k: getattr(fmt, k) for k in dir(fmt) if not k.startswith('_')}

ZXING_FORMATS = _probe_zxing_formats()
IMB_ALIASES   = ['IMB', 'OneCode', 'USPSOneCode', 'ONECODE', 'OneD']
IMB_FORMAT    = next((ZXING_FORMATS[a] for a in IMB_ALIASES if a in ZXING_FORMATS), None)


def try_zxing(img: np.ndarray, label: str = ""):
    """
    Attempt zxing-cpp decode.
    Tries: targeted IMB format (if found), then common collection formats,
    then all-formats fallback.
    """
    fmts_to_try = []
    if IMB_FORMAT is not None:
        fmts_to_try.append((IMB_FORMAT, "IMB"))
    # AllLinear and AllIndustrial include postal barcodes in most builds
    for name in ["AllLinear", "AllIndustrial", "All"]:
        if name in ZXING_FORMATS:
            fmts_to_try.append((ZXING_FORMATS[name], name))
            break  # just need one collection fmt
    fmts_to_try.append((None, "any"))  # final fallback

    for fmt, tag in fmts_to_try:
        try:
            kw = {"formats": fmt} if fmt is not None else {}
            results = zxingcpp.read_barcodes(img, **kw)
            if results:
                return results[0].text, f"{label}[zxing-{tag}]"
        except Exception:
            pass
    return None, label


# ─── python-zxing (Java ZXing) Fallback Decoder ──────────────────────────────

def try_pyzxing(img: np.ndarray, label: str = ""):
    """
    Fallback decoder using python-zxing (wraps Java ZXing library).
    Java ZXing has full USPS IMB support. Requires: pip install python-zxing
    and Java (sudo apt install default-jre).
    """
    if not PYZXING_AVAILABLE:
        return None, label
    try:
        import tempfile, os
        from PIL import Image as PILImage
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            tmppath = f.name
        if img.ndim == 3:
            PILImage.fromarray(img).save(tmppath)
        else:
            PILImage.fromarray(img).save(tmppath)
        reader = pyzxing_lib.BarCodeReader()
        result = reader.decode(tmppath)
        os.unlink(tmppath)
        if result and hasattr(result, "raw") and result.raw:
            return result.raw, f"{label}[pyzxing]"
        if result and hasattr(result, "parsed") and result.parsed:
            return result.parsed, f"{label}[pyzxing-parsed]"
    except Exception:
        pass
    return None, label


# ─── pyimb Decoder (FADT string → tracking number) ──────────────────────────

def try_pyimb(fadt: str) -> tuple:
    """
    Decode a 65-char FADT string directly to tracking+routing number using pyimb.
    Returns (tracking+routing string | None, method_label).
    pyimb.decode() returns a dict — we combine tracking+routing into the
    format parse_imb() expects: 20-digit tracking + 0/5/9/11-digit routing.
    """
    if not PYIMB_AVAILABLE or not fadt or len(fadt) != 65:
        return None, "pyimb-unavailable"
    try:
        result = pyimb.decode(fadt)
        if result:
            combined = result['tracking'] + result['routing']
            return combined, f"pyimb[crc={'OK' if result['crc_ok'] else 'FAIL'}]"
    except Exception as e:
        return None, f"pyimb-error:{e}"
    return None, "pyimb-no-result"


# ─── FADT Extractor ───────────────────────────────────────────────────────────

def extract_fadt(crop_gray: np.ndarray):
    """
    Read 65 bar states (F/A/D/T) from pixel heights in a grayscale IMB crop.

    1. Binarize
    2. Column projection  locate 65 bar centers
    3. Measure each bars top and bottom pixel row
    4. Cluster tops into 2 groups (tracker-top vs ascender-top) via largest gap
    5. Same for bottoms
    6. Classify: ascender? descender? both? neither?

    Returns (fadt_string | None, debug_dict)
    """
    h, w   = crop_gray.shape
    debug  = {}
    best   = None

    binarizations = [
        (cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1], "otsu"),
        (cv2.adaptiveThreshold(crop_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 21, 8), "adaptive"),
    ]

    for binary, bname in binarizations:
        col_proj = np.convolve(binary.sum(axis=0).astype(float),
                               np.ones(2) / 2, mode='same')
        thr = col_proj.max() * 0.25
        in_bar = col_proj > thr

        centers = []
        i = 0
        while i < w:
            if in_bar[i]:
                s = i
                while i < w and in_bar[i]:
                    i += 1
                centers.append((s + i) // 2)
            else:
                i += 1

        # Attempt merge if slightly over 65
        if len(centers) > 65:
            pitch = w / 65
            merged = [centers[0]]
            for bc in centers[1:]:
                if bc - merged[-1] < pitch * 0.6:
                    merged[-1] = (merged[-1] + bc) // 2
                else:
                    merged.append(bc)
            centers = merged

        debug[f"{bname}_bars"] = len(centers)
        if len(centers) != 65:
            continue

        # Measure each bar
        measurements = []
        for bx in centers:
            x0 = max(0, bx - 2)
            x1 = min(w, bx + 3)
            col_strip = binary[:, x0:x1].max(axis=1)
            rows = np.where(col_strip > 127)[0]
            measurements.append(
                (int(rows[0]), int(rows[-1])) if len(rows) else (h // 4, 3 * h // 4)
            )

        tops    = [m[0] for m in measurements]
        bottoms = [m[1] for m in measurements]

        def largest_gap_split(vals):
            s    = sorted(vals)
            gaps = [s[k + 1] - s[k] for k in range(len(s) - 1)]
            if not gaps or max(gaps) < 2:
                mid = np.mean(s)
                return mid, mid
            sp   = gaps.index(max(gaps))
            return (s[sp] + s[sp + 1]) / 2.0, (s[sp] + s[sp + 1]) / 2.0

        top_mid, _  = largest_gap_split(tops)
        bot_mid, _  = largest_gap_split(bottoms)

        debug[f"{bname}_top_mid"] = round(top_mid, 1)
        debug[f"{bname}_bot_mid"] = round(bot_mid, 1)

        fadt = []
        for (bt, bb) in measurements:
            has_asc  = bt < top_mid
            has_desc = bb > bot_mid
            if has_asc and has_desc:
                fadt.append('F')
            elif has_asc:
                fadt.append('A')
            elif has_desc:
                fadt.append('D')
            else:
                fadt.append('T')

        fadt_str = ''.join(fadt)
        debug[f"{bname}_fadt"] = fadt_str
        if best is None:
            best = fadt_str  # keep first (otsu runs first, is more reliable)

    return best, debug


# ─── Synthetic Render ─────────────────────────────────────────────────────────

def render_synthetic_imb(fadt: str) -> np.ndarray:
    """
    Render a clean, pixel-perfect IMB image from a 65-char FADT string.
    Dimensions based on USPS B-3200 spec at ~200 DPI.
    """
    assert len(fadt) == 65

    BAR_W   = 4
    PITCH   = 9
    IMG_H   = 60
    QUIET   = 15
    TRACK_H = 12
    EXT_H   = 7       # ascender or descender extension above/below tracker

    mid   = IMG_H // 2
    t_top = mid - TRACK_H // 2
    t_bot = mid + TRACK_H // 2

    img_w = QUIET * 2 + 64 * PITCH + BAR_W
    img   = np.full((IMG_H, img_w), 255, dtype=np.uint8)

    for i, state in enumerate(fadt):
        x0 = QUIET + i * PITCH
        x1 = x0 + BAR_W
        if state == 'F':
            y0, y1 = t_top - EXT_H, t_bot + EXT_H
        elif state == 'A':
            y0, y1 = t_top - EXT_H, t_bot
        elif state == 'D':
            y0, y1 = t_top, t_bot + EXT_H
        else:
            y0, y1 = t_top, t_bot
        img[y0:y1, x0:x1] = 0

    return img


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def scan_image(pil_img: Image.Image, log=None):
    """
    log: optional callable(icon, message) for live progress updates.
         icon is one of "⏳", "✅", "❌", "🔍", "📊"
    """
    def emit(icon, msg):
        if log:
            log(icon, msg)

    img_rgb  = np.array(pil_img.convert("RGB"))
    H0, W0   = img_rgb.shape[:2]
    gray     = to_gray(img_rgb)
    fadt_dbg = {}
    fadt_str = None
    synthetic = None

    emit("🔍", f"Image loaded — {W0}×{H0}px")

    # Pass 1: full image zxing
    emit("⏳", "Pass 1 — trying zxing-cpp on full image (color / gray / sharpened)...")
    for candidate, lbl in [(img_rgb, "full-color"), (gray, "full-gray"),
                           (sharpen(gray), "full-sharp")]:
        text, method = try_zxing(candidate, lbl)
        if text:
            emit("✅", f"Pass 1 decoded via zxing-cpp [{lbl}]")
            return text, img_rgb.copy(), None, None, None, method, {}
        text, method = try_pyzxing(candidate, lbl)
        if text:
            emit("✅", f"Pass 1 decoded via python-zxing [{lbl}]")
            return text, img_rgb.copy(), None, None, None, method, {}
    emit("❌", "Pass 1 — no decode from full image")

    # Detect region
    emit("⏳", "Detecting IMB region via Sobel-X edge density...")
    region    = detect_imb_region(gray)
    annotated = img_rgb.copy()
    if region is None:
        emit("❌", "Region detection FAILED — no high-density vertical-edge band found. "
                   "Try a flatter, better-lit photo with the barcode filling more of the frame.")
        return None, annotated, None, None, None, "No IMB region detected", {}

    x, y, w, h = region
    pad = 10
    H, W = img_rgb.shape[:2]
    x1 = max(0, x - pad);      y1 = max(0, y - pad)
    x2 = min(W, x + w + pad);  y2 = min(H, y + h + pad)
    emit("✅", f"Region detected — bounding box ({x1},{y1})→({x2},{y2}), "
               f"crop size {x2-x1}×{y2-y1}px")

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
    cv2.putText(annotated, "IMB region", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)

    crop_rgb  = img_rgb[y1:y2, x1:x2]
    crop_gray = gray[y1:y2, x1:x2]

    # Pass 2: zxing-cpp on crop at multiple scales
    emit("⏳", "Pass 2 — trying zxing-cpp on detected crop at scales 1×–4×...")
    for scale in [2, 3, 4, 1]:
        for src, lbl in [(crop_rgb, "crop-color"), (crop_gray, "crop-gray"),
                          (sharpen(crop_gray), "crop-sharp")]:
            s = cv2.resize(src, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC) if scale != 1 else src
            text, method = try_zxing(s, f"{lbl}x{scale}")
            if text:
                emit("✅", f"Pass 2 decoded via zxing-cpp [{lbl} ×{scale}]")
                return text, annotated, crop_rgb, None, None, method, {}
            text, method = try_pyzxing(s, f"{lbl}x{scale}")
            if text:
                emit("✅", f"Pass 2 decoded via python-zxing [{lbl} ×{scale}]")
                return text, annotated, crop_rgb, None, None, method, {}
    emit("❌", "Pass 2 — zxing-cpp could not decode crop at any scale "
               "(expected — zxing-cpp lacks IMB support in this build)")

    # Pass 3: pixel FADT extraction -> pyimb direct decode -> synthetic fallback
    emit("⏳", "Pass 3 — extracting FADT bar states via pixel analysis...")
    for scale in [4, 6, 8, 3]:
        scaled = cv2.resize(crop_gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
        _, fadt_dbg = extract_fadt(scaled)
        fadt_dbg['pyimb_available'] = PYIMB_AVAILABLE

        otsu_bars = fadt_dbg.get('otsu_bars', 0)
        otsu_fadt = fadt_dbg.get('otsu_fadt', '')
        adap_fadt = fadt_dbg.get('adaptive_fadt', '')
        emit("📊", f"Scale ×{scale} — Otsu: {otsu_bars} bars detected, "
                   f"FADT length={len(otsu_fadt)} | "
                   f"Adaptive FADT length={len(adap_fadt)}")

        # Collect all valid FADT candidates from this scale (otsu + adaptive)
        candidates = []
        for key in ['otsu_fadt', 'adaptive_fadt']:
            val = fadt_dbg.get(key, '')
            if val and len(val) == 65:
                candidates.append((val, key))

        if not candidates:
            emit("❌", f"Scale ×{scale} — could not extract 65 bars "
                       f"(got {otsu_bars}). Bar spacing may be too tight — try closer/better-lit photo.")
            continue

        # Try pyimb on each candidate — otsu first
        for fadt_candidate, src_name in candidates:
            fadt_dbg['pyimb_fadt_input'] = fadt_candidate
            fadt_dbg['pyimb_source'] = src_name
            emit("⏳", f"Scale ×{scale} — running pyimb decode on {src_name} FADT...")
            text, method = try_pyimb(fadt_candidate)
            fadt_dbg['pyimb_result'] = text
            fadt_dbg['pyimb_method'] = method
            if text:
                fadt_str = fadt_candidate
                emit("✅", f"pyimb decoded successfully from {src_name} at ×{scale}! CRC OK.")
                return (text, annotated, crop_rgb, fadt_str, None,
                        f"FADT({src_name})+{method}", fadt_dbg)
            else:
                emit("❌", f"pyimb FAILED on {src_name} at ×{scale} — "
                           f"result: {method}. Bar states likely have misclassifications.")

        # Fallback: render synthetic and try image decoders
        fadt_str = fadt_dbg.get('otsu_fadt') or fadt_dbg.get('adaptive_fadt')
        if fadt_str and len(fadt_str) == 65:
            emit("⏳", f"Scale ×{scale} — rendering synthetic barcode from FADT, "
                       "trying zxing-cpp as last resort...")
            synthetic = render_synthetic_imb(fadt_str)
            for syn_scale in [1, 2, 3]:
                syn = (cv2.resize(synthetic, None, fx=syn_scale, fy=syn_scale,
                                  interpolation=cv2.INTER_NEAREST)
                       if syn_scale > 1 else synthetic)
                text, method = try_zxing(syn, f"synthetic-x{syn_scale}")
                if text:
                    emit("✅", f"Synthetic barcode decoded via zxing-cpp at ×{syn_scale}")
                    return (text, annotated, crop_rgb, fadt_str, synthetic,
                            f"FADT+{method}", fadt_dbg)
                text, method = try_pyzxing(syn, f"synthetic-x{syn_scale}")
                if text:
                    emit("✅", f"Synthetic barcode decoded via python-zxing at ×{syn_scale}")
                    return (text, annotated, crop_rgb, fadt_str, synthetic,
                            f"FADT+{method}", fadt_dbg)
            emit("❌", "Synthetic barcode also failed — all decode paths exhausted at this scale.")
            break  # FADT extracted but all decoders failed

    emit("❌", "All passes failed. See FADT debug output below for clues.")
    return (None, annotated, crop_rgb, fadt_str, synthetic,
            "Detected region, decode failed", fadt_dbg)


# ─── UI ───────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a mailpiece photo",
    type=["jpg", "jpeg", "png", "webp", "heic"],
)

if uploaded:
    pil_img = Image.open(uploaded)
    # Fix EXIF orientation — phone cameras embed rotation metadata that PIL
    # ignores by default, causing the image to arrive sideways or upside down
    try:
        from PIL.ExifTags import TAGS
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
                    break
    except Exception:
        pass  # No EXIF data or not a JPEG — continue normally

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    # Live progress log using st.status
    log_lines = []
    with st.status("🔍 Scanning barcode...", expanded=True) as status_box:
        step_container = st.container()

        def log(icon, msg):
            log_lines.append((icon, msg))
            step_container.markdown(f"{icon} {msg}")
            status_box.update(label=f"{icon} {msg}")

        decoded, annotated, crop, fadt, synthetic, method, fadt_dbg = scan_image(pil_img, log=log)

        if decoded:
            status_box.update(label="✅ Barcode decoded successfully!", state="complete")
        else:
            status_box.update(label="❌ Could not decode — see steps above for clues",
                              state="error")

    with col2:
        st.subheader("Detection")
        st.image(annotated, use_container_width=True, caption=f"Method: {method}")

    if crop is not None:
        st.subheader("Detected IMB Crop")
        st.image(cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                 use_container_width=True)

    if fadt is not None:
        st.subheader("Extracted FADT String (65 bars)")
        st.code(fadt, language=None)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Full (F)",      fadt.count('F'))
        c2.metric("Ascender (A)",  fadt.count('A'))
        c3.metric("Descender (D)", fadt.count('D'))
        c4.metric("Tracker (T)",   fadt.count('T'))

    if synthetic is not None:
        st.subheader("Synthetic Barcode (fed to decoder)")
        st.image(cv2.resize(synthetic, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST),
                 use_container_width=True)

    with st.expander("Debug: edge map & row energy"):
        gray_dbg = to_gray(np.array(pil_img.convert("RGB")))
        edge_vis = np.abs(cv2.Sobel(gray_dbg, cv2.CV_32F, 1, 0, ksize=3))
        edge_vis = (edge_vis / edge_vis.max() * 255).astype(np.uint8)
        st.image(edge_vis, caption="Sobel-X edge map", use_container_width=True)
        row_e = edge_vis.astype(float).sum(axis=1)
        st.line_chart(row_e / row_e.max(), height=120)
        st.caption("Peak should align with barcode row")
        st.markdown("**zxing-cpp available formats:**")
        st.code(", ".join(sorted(ZXING_FORMATS.keys())))
        if IMB_FORMAT is not None:
            st.success(f"IMB format found as: {[a for a in IMB_ALIASES if a in ZXING_FORMATS][0]}")
        else:
            st.warning("No IMB/OneCode format constant found in this zxing-cpp build.")
        if PYIMB_AVAILABLE:
            st.success("pyimb is available — pure Python FADT decode active (best path)")
        else:
            st.warning("pyimb not found. Copy intelligent_mail_barcode.py to this folder. "
                       "Get it from: https://github.com/samrushing/pyimb")
        if PYZXING_AVAILABLE:
            st.success("python-zxing (Java ZXing) is available")
        else:
            st.info("python-zxing not installed (optional). "
                    "sudo apt install default-jre && pip install python-zxing")
        if fadt_dbg:
            st.json(fadt_dbg)

    st.divider()

    if decoded:
        st.success(f"Decoded: `{decoded}`")
        parsed = parse_imb(decoded)

        if "error" in parsed:
            st.error(parsed["error"])
        else:
            st.subheader("IMB Components")
            ca, cb, cc = st.columns(3)
            ca.metric("Barcode ID",      parsed["barcode_id"])
            cb.metric("STID",            parsed["stid"])
            cc.metric("MID",             f"{parsed['mid_length']}-digit")
            cd, ce = st.columns(2)
            cd.metric("Mailer ID",       parsed["mid"])
            ce.metric("Serial Number",   parsed["serial_number"])

            # STID Lookup
            st.subheader("Service Type Details")
            if STID_DB_AVAILABLE:
                stid_info = stid_db.lookup(parsed["stid"])
                if stid_info:
                    sc1, sc2 = st.columns(2)
                    mail_label = stid_info["mail_class"]
                    if stid_info["mail_subclass"]:
                        mail_label += f" — {stid_info['mail_subclass']}"
                    sc1.metric("Mail Class",          mail_label)
                    sc2.metric("ACS Type",
                               stid_info["acs_type"] if stid_info["acs_type"] else "No ACS")
                    sc3, sc4, sc5 = st.columns(3)
                    sc3.metric("Address Correction",  stid_info["address_correction"])
                    sc4.metric("Service Level",       stid_info["service_level"])
                    sc5.metric("IV® MTR",
                               "✅ With IV MTR" if stid_info["iv_mtr"] else "❌ Without IV MTR")
                    if stid_info["note_texts"]:
                        with st.expander("📋 Applicable Notes & Footnotes"):
                            for flag, note in zip(stid_info["flags"], stid_info["note_texts"]):
                                st.markdown(f"**{flag}** — {note}")
                else:
                    st.warning(f"STID {parsed['stid']} not found in lookup table. "
                               "May be a promotional or Secure Destruction STID not in the Jan 2024 table.")
            else:
                st.info("stid_table.py not found — copy it to the project folder for STID lookups.")

            st.subheader("Routing")
            r = parsed["routing"]
            st.write(f"**Type:** {r.get('type')}")
            rc = st.columns(3)
            if "zip5"  in r: rc[0].metric("ZIP",       r["zip5"])
            if "plus4" in r: rc[1].metric("ZIP+4",     r["plus4"])
            if "dpc"   in r: rc[2].metric("Del Point", r["dpc"])

            with st.expander("Full JSON"):
                st.json(parsed)

            with st.expander("Tracking breakdown"):
                tc = parsed["tracking_code"]
                me = 5 + parsed["mid_length"]
                st.markdown(
                    f"`{tc[0:2]}` **BarcodeID** · "
                    f"`{tc[2:5]}` **STID** · "
                    f"`{tc[5:me]}` **MID** · "
                    f"`{tc[me:20]}` **Serial**"
                )
                if r.get("type") != "No routing code":
                    st.markdown(f"`{parsed['raw_decoded'][20:]}` **Routing**")
    else:
        st.error("Could not decode an IMB from this image.")
        if fadt and len(fadt) == 65:
            st.warning(
                "65 bar states were extracted (see FADT above) but zxing-cpp "
                "rejected the result. A few bars may be misclassified. "
                "Try a higher-resolution or tighter-cropped photo."
            )
        else:
            n = fadt_dbg.get("otsu_bars") or fadt_dbg.get("adaptive_bars", 0)
            if n:
                st.warning(f"Found {n} bars instead of 65. "
                           "Barcode may be cut off, blurry, or at an angle.")

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("IMB Reference")
    st.markdown("""
**Barcode ID** (2 digits) — type indicator

**STID** (3 digits) — mail class + service
e.g. `300` = First-Class

**MID** (6 or 9 digits) — mailer identifier
- Starts 0–8 → 9-digit MID
- Starts 9 → 6-digit MID

**Serial** — piece ID

**Routing**
| Len | Type |
|-----|------|
| 0 | None |
| 5 | ZIP |
| 9 | ZIP+4 |
| 11 | ZIP+4+DPC |
    """)
    st.divider()
    st.markdown("""
**Pipeline**
1. Sobel-X edge density → barcode row
2. zxing-cpp direct decode
3. Pixel FADT extraction →
   synthetic render → zxing-cpp

No AI · No API · Fully offline
    """)
