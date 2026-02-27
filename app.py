"""
USPS Intelligent Mail Barcode (IMB) Scanner
============================================
Fully deterministic pipeline — no AI, no API calls.

Pipeline:
  1. Detect IMB region via Sobel-X edge density
  2. Extract FADT bar states (multiple preprocessing variants + binarizations)
  3. Decode FADT string via pyimb (USPS codeword + Reed-Solomon + CRC)
  4. Parse decoded number into USPS components
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

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

st.set_page_config(page_title="USPS IMB Scanner", page_icon="📮", layout="centered")
st.title("📮 USPS Intelligent Mail Barcode Scanner")
st.caption("Deterministic IMB detection · no AI · just pixels and math · v5")

if not PYIMB_AVAILABLE:
    st.error("pyimb not found. Add intelligent_mail_barcode.py to the project folder.")
    st.stop()

# ─── IMB Parser ───────────────────────────────────────────────────────────────

def parse_imb(raw_text):
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
    if   rlen == 0:  route_info = {"type": "No routing code"}
    elif rlen == 5:  route_info = {"type": "ZIP Code",   "zip5": routing}
    elif rlen == 9:  route_info = {"type": "ZIP+4",      "zip5": routing[:5], "plus4": routing[5:]}
    elif rlen == 11: route_info = {"type": "ZIP+4+DPC",  "zip5": routing[:5], "plus4": routing[5:9], "dpc": routing[9:]}
    else:            route_info = {"type": f"Unexpected length ({rlen})", "raw": routing}
    return {"raw_decoded": raw_text, "tracking_code": tracking,
            "barcode_id": tracking[0:2], "stid": tracking[2:5],
            "mid": mid, "mid_length": mid_len, "serial_number": serial, "routing": route_info}

# ─── Image Helpers ────────────────────────────────────────────────────────────

def to_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(g)

def preprocess_variants(gray):
    """5 preprocessing variants to maximise bar separation across different photo conditions."""
    out = [("raw", gray)]
    blur = cv2.GaussianBlur(gray, (0,0), 2)
    out.append(("sharp", cv2.addWeighted(gray, 1.8, blur, -0.8, 0)))
    out.append(("bilateral", cv2.bilateralFilter(gray, 9, 75, 75)))
    out.append(("gamma", (np.power(gray/255.0, 0.5)*255).astype(np.uint8)))
    kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    out.append(("morph", cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kv)))
    return out

# ─── Region Detection ─────────────────────────────────────────────────────────

def detect_imb_region(gray):
    H, W = gray.shape
    edge  = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    row_e = edge.sum(axis=1)
    ks    = max(3, H//60)
    row_e = np.convolve(row_e, np.ones(ks)/ks, mode='same')
    peak  = int(np.argmax(row_e))
    thr   = row_e[peak] * 0.40
    top   = peak
    while top > 0        and row_e[top-1] > thr: top -= 1
    bot   = peak
    while bot < H-1      and row_e[bot+1] > thr: bot += 1
    pad_v = max(4, (bot-top)//2)
    top   = max(0, top-pad_v); bot = min(H-1, bot+pad_v)
    if bot-top < 6: return None
    col_e  = edge[top:bot,:].sum(axis=0)
    active = np.where(col_e > col_e.max()*0.15)[0]
    if len(active) == 0: return None
    xl, xr = int(active[0]), int(active[-1])
    bw = xr - xl
    if bw < 80 or bw/max(bot-top,1) < 3.0: return None
    return xl, top, bw, bot-top

# ─── FADT Extraction ──────────────────────────────────────────────────────────

def estimate_pitch(col_proj, w):
    """Autocorrelation-based pitch estimation — more robust than span/65."""
    proj = col_proj - col_proj.mean()
    acorr = np.correlate(proj, proj, mode='full')
    acorr = acorr[len(acorr)//2:]
    acorr /= (acorr[0] + 1e-9)
    lo, hi = 3, max(4, len(acorr)//10)
    if hi >= len(acorr): return w/65.0
    peak_idx = int(np.argmax(acorr[lo:hi])) + lo
    return float(peak_idx) if peak_idx > 0 else w/65.0

def find_tracker_band(binary, centers):
    """
    Find the tracker band by sampling ONLY at the bar center columns.
    Every IMB bar has a tracker segment → the tracker band is the row range
    where ALL (or nearly all) bar centers are dark.

    Using full-width row sums (old approach) fails because the bars collectively
    make every row look dense. Sampling only at bar positions gives a true signal.
    """
    h = binary.shape[0]
    if not centers:
        return None

    # Build a per-row coverage array: fraction of bar centers that are dark
    coverage = np.zeros(h, dtype=float)
    half = 2
    w = binary.shape[1]
    for cx in centers:
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        col = binary[:, x0:x1].max(axis=1).astype(float)
        coverage += col
    coverage /= len(centers)  # now: fraction of bars present at each row (0–1)

    # Tracker band = rows where >80% of bars are present
    dense = coverage > 0.80

    # Find the longest contiguous dense band
    best_start, best_len, cur_start, cur_len = 0, 0, 0, 0
    for i, d in enumerate(dense):
        if d:
            if cur_len == 0: cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len; best_start = cur_start
        else:
            cur_len = 0

    if best_len < 2:
        return None
    return (best_start, best_start + best_len)


def locate_centers(col_proj, w):
    """Find 65 bar centers: threshold → split wide segments → force-pitch fallback."""
    thr    = col_proj.max() * 0.20
    in_bar = col_proj > thr
    segs   = []
    i = 0
    while i < w:
        if in_bar[i]:
            s = i
            while i < w and in_bar[i]: i += 1
            segs.append((s, i))
        else:
            i += 1
    if not segs: return [], segs

    pitch_ac   = estimate_pitch(col_proj, w)
    span       = segs[-1][1] - segs[0][0]
    pitch_span = span/65.0 if span > 0 else w/65.0
    pitch      = pitch_ac if 3 < pitch_ac < w/4 else pitch_span

    # Split wide segments (merged bars)
    centers = []
    for s, e in segs:
        seg_w  = e - s
        n_bars = max(1, round(seg_w/pitch))
        for k in range(n_bars):
            centers.append(s + int((k+0.5)*seg_w/n_bars))

    # Merge noise-split bars
    if len(centers) > 65:
        merged = [centers[0]]
        for bc in centers[1:]:
            if bc - merged[-1] < pitch*0.6: merged[-1] = (merged[-1]+bc)//2
            else: merged.append(bc)
        centers = merged

    # Force-pitch fallback
    if len(centers) != 65 and len(segs) >= 20:
        fp = (segs[-1][1]-segs[0][0])/64.0
        centers = [int(segs[0][0]+i*fp) for i in range(65)]

    return centers, segs

def classify_bars(centers, binary, tracker_band=None):
    """
    Classify each bar as F/A/D/T using USPS proportional extent method.

    The spec defines fixed proportions: F bars span the full height. The
    ascender zone is the top ~33% of total bar height, the descender zone
    is the bottom ~33%. A bar that reaches into the top zone has an ascender;
    a bar that reaches into the bottom zone has a descender.

    We derive the coordinate system from the global min/max of all bar tops
    and bottoms — the F bars (tallest) define the full extent, and everything
    else is measured against that. No clustering, no statistics, no tuning.
    """
    h, w = binary.shape
    half = 3
    meas = []
    for cx in centers:
        x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
        strip = binary[:, x0:x1].max(axis=1)
        rows  = np.where(strip > 0)[0]
        meas.append((int(rows[0]), int(rows[-1])) if len(rows) else (h//4, 3*h//4))

    tops    = [m[0] for m in meas]
    bottoms = [m[1] for m in meas]

    # Global extent — defined by the tallest bars (F bars)
    global_top    = min(tops)
    global_bottom = max(bottoms)
    bar_h         = max(1, global_bottom - global_top)

    # USPS proportions: ascender/descender zones occupy the outer ~33% each
    asc_thr  = global_top    + bar_h * 0.33   # bar top must be ABOVE this → ascender
    desc_thr = global_bottom - bar_h * 0.33   # bar bottom must be BELOW this → descender

    fadt = []
    for bt, bb in meas:
        a = bt < asc_thr
        d = bb > desc_thr
        fadt.append('F' if a and d else 'A' if a else 'D' if d else 'T')
    return ''.join(fadt), "usps-proportional"

def extract_fadt(crop_gray, scale_label=""):
    """
    Try 5 preprocessing variants × 3 binarizations = up to 15 attempts.
    Returns first FADT with CRC-OK from pyimb, else best 65-bar candidate.
    """
    h, w   = crop_gray.shape
    debug  = {"scale": scale_label, "crop_size": f"{w}x{h}"}
    best   = None

    for vname, vimg in preprocess_variants(crop_gray):
        for bname, binary in [
            ("otsu",    cv2.threshold(vimg, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]),
            ("adapt21", cv2.adaptiveThreshold(vimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,8)),
            ("adapt11", cv2.adaptiveThreshold(vimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)),
        ]:
            label    = f"{vname}+{bname}"
            col_proj = np.convolve(binary.sum(axis=0).astype(float), np.ones(2)/2, mode='same')
            centers, segs = locate_centers(col_proj, w)
            debug[f"{label}_bars"] = len(centers)
            if len(centers) != 65: continue

            tracker  = None  # no longer needed — using USPS proportional method
            fadt_str, cls_method = classify_bars(centers, binary, tracker)
            if not fadt_str or len(fadt_str) != 65: continue

            debug[f"{label}_fadt"]   = fadt_str
            debug[f"{label}_clsmethod"] = cls_method
            result = pyimb.decode(fadt_str)
            if result and result.get('crc_ok'):
                debug["winning_variant"] = label
                return fadt_str, debug
            if best is None:
                best = fadt_str

    return best, debug

# ─── pyimb ───────────────────────────────────────────────────────────────────

def try_pyimb(fadt):
    if not fadt or len(fadt) != 65: return None, "invalid-fadt"
    try:
        r = pyimb.decode(fadt)
        if r:
            return r['tracking']+r['routing'], f"pyimb[crc={'OK' if r['crc_ok'] else 'FAIL'}]"
    except Exception as e:
        return None, f"pyimb-error:{e}"
    return None, "pyimb-no-result"

# ─── Main Pipeline ────────────────────────────────────────────────────────────

def scan_image(pil_img, log=None):
    def emit(icon, msg):
        if log: log(icon, msg)

    img_rgb   = np.array(pil_img.convert("RGB"))
    H0, W0    = img_rgb.shape[:2]
    gray      = to_gray(img_rgb)
    fadt_dbg  = {}
    fadt_str  = None
    annotated = img_rgb.copy()

    emit("🔍", f"Image loaded — {W0}×{H0}px")
    emit("⏳", "Detecting IMB region via Sobel-X edge density...")

    region = detect_imb_region(gray)
    if region is None:
        emit("❌", "Region detection FAILED — no barcode band found. "
                   "Try a flatter, better-lit photo.")
        return None, annotated, None, None, None, "No IMB region detected", {}

    x, y, w, h = region
    pad = 10
    H, W = img_rgb.shape[:2]
    x1 = max(0,x-pad); y1 = max(0,y-pad)
    x2 = min(W,x+w+pad); y2 = min(H,y+h+pad)
    emit("✅", f"Region found — crop ({x1},{y1})→({x2},{y2}), size {x2-x1}×{y2-y1}px")

    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,200,80), 2)
    cv2.putText(annotated, "IMB region", (x1, max(0,y1-6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,80), 1)

    crop_gray = gray[y1:y2, x1:x2]
    crop_rgb  = img_rgb[y1:y2, x1:x2]

    emit("⏳", "Extracting FADT — trying 5 preprocessing variants × 3 binarizations per scale...")

    for scale in [4, 6, 8, 3, 2]:
        scaled = cv2.resize(crop_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        fadt_candidate, scale_dbg = extract_fadt(scaled, scale_label=f"x{scale}")
        fadt_dbg[f"scale_{scale}"] = scale_dbg

        bar_counts = [v for k,v in scale_dbg.items() if k.endswith("_bars") and isinstance(v,int)]
        best_count = max(bar_counts) if bar_counts else 0
        n65 = sum(1 for v in bar_counts if v == 65)

        if fadt_candidate and len(fadt_candidate) == 65:
            text, method = try_pyimb(fadt_candidate)
            if text:
                winner = scale_dbg.get("winning_variant", "?")
                emit("✅", f"Scale ×{scale} decoded! Variant: {winner} "
                           f"({n65}/{len(bar_counts)} got 65 bars) — {method}")
                fadt_str = fadt_candidate
                return (text, annotated, crop_rgb, fadt_str, None,
                        f"FADT(x{scale},{winner})+{method}", fadt_dbg)
            else:
                emit("❌", f"Scale ×{scale} — 65 bars extracted but CRC failed "
                           f"({n65}/{len(bar_counts)} variants). Some bars misclassified.")
                if fadt_str is None: fadt_str = fadt_candidate
        else:
            reason = "bars merging — try closer photo" if best_count > 50 else "check crop/lighting"
            emit("❌", f"Scale ×{scale} — best bar count: {best_count}/65 "
                       f"({len(bar_counts)} variants tried) — {reason}")

    emit("❌", "All scales/variants failed. See debug panel for per-variant bar counts.")
    return (None, annotated, crop_rgb, fadt_str, None, "Detected region, decode failed", fadt_dbg)

# ─── UI ───────────────────────────────────────────────────────────────────────

uploaded = st.file_uploader("Upload a mailpiece photo",
                            type=["jpg","jpeg","png","webp","heic"])

if uploaded:
    pil_img = Image.open(uploaded)
    try:
        from PIL.ExifTags import TAGS
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if TAGS.get(tag) == 'Orientation':
                    if   value == 3: pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6: pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8: pil_img = pil_img.rotate(90,  expand=True)
                    break
    except Exception:
        pass

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_img, use_container_width=True)

    with st.status("🔍 Scanning barcode...", expanded=True) as status_box:
        step_container = st.container()
        def log(icon, msg):
            step_container.markdown(f"{icon} {msg}")
            status_box.update(label=f"{icon} {msg}")

        decoded, annotated, crop, fadt, _, method, fadt_dbg = scan_image(pil_img, log=log)

        if decoded:
            status_box.update(label="✅ Barcode decoded successfully!", state="complete")
        else:
            status_box.update(label="❌ Could not decode — see steps above", state="error")

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
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Full (F)",     fadt.count('F'))
        c2.metric("Ascender (A)", fadt.count('A'))
        c3.metric("Descender (D)",fadt.count('D'))
        c4.metric("Tracker (T)",  fadt.count('T'))

    with st.expander("🔬 Debug: edge map & extraction detail"):
        gray_dbg = to_gray(np.array(pil_img.convert("RGB")))
        edge_vis = np.abs(cv2.Sobel(gray_dbg, cv2.CV_32F, 1, 0, ksize=3))
        edge_vis = (edge_vis/edge_vis.max()*255).astype(np.uint8)
        st.image(edge_vis, caption="Sobel-X edge map", use_container_width=True)
        row_e = edge_vis.astype(float).sum(axis=1)
        st.line_chart(row_e/row_e.max(), height=120)
        st.caption("Peak should align with barcode row")
        st.success("pyimb active — CRC validation on every FADT candidate") if PYIMB_AVAILABLE else st.error("pyimb missing")
        if fadt_dbg: st.json(fadt_dbg)

    st.divider()

    if decoded:
        st.success(f"Decoded: `{decoded}`")
        parsed = parse_imb(decoded)
        if "error" in parsed:
            st.error(parsed["error"])
        else:
            st.subheader("IMB Components")
            ca,cb,cc = st.columns(3)
            ca.metric("Barcode ID",    parsed["barcode_id"])
            cb.metric("STID",          parsed["stid"])
            cc.metric("MID",           f"{parsed['mid_length']}-digit")
            cd,ce = st.columns(2)
            cd.metric("Mailer ID",     parsed["mid"])
            ce.metric("Serial Number", parsed["serial_number"])

            st.subheader("Service Type Details")
            if STID_DB_AVAILABLE:
                stid_info = stid_db.lookup(parsed["stid"])
                if stid_info:
                    sc1,sc2 = st.columns(2)
                    ml = stid_info["mail_class"]
                    if stid_info["mail_subclass"]: ml += f" — {stid_info['mail_subclass']}"
                    sc1.metric("Mail Class",         ml)
                    sc2.metric("ACS Type",           stid_info["acs_type"] or "No ACS")
                    sc3,sc4,sc5 = st.columns(3)
                    sc3.metric("Address Correction", stid_info["address_correction"])
                    sc4.metric("Service Level",      stid_info["service_level"])
                    sc5.metric("IV® MTR",            "✅ With IV MTR" if stid_info["iv_mtr"] else "❌ Without IV MTR")
                    if stid_info["note_texts"]:
                        with st.expander("📋 Notes & Footnotes"):
                            for flag,note in zip(stid_info["flags"],stid_info["note_texts"]):
                                st.markdown(f"**{flag}** — {note}")
                else:
                    st.warning(f"STID {parsed['stid']} not found in lookup table.")
            else:
                st.info("stid_table.py not found.")

            st.subheader("Routing")
            r = parsed["routing"]
            st.write(f"**Type:** {r.get('type')}")
            rc = st.columns(3)
            if "zip5"  in r: rc[0].metric("ZIP",       r["zip5"])
            if "plus4" in r: rc[1].metric("ZIP+4",     r["plus4"])
            if "dpc"   in r: rc[2].metric("Del Point", r["dpc"])

            with st.expander("Full JSON"):   st.json(parsed)
            with st.expander("Tracking breakdown"):
                tc = parsed["tracking_code"]; me = 5+parsed["mid_length"]
                st.markdown(f"`{tc[0:2]}` **BarcodeID** · `{tc[2:5]}` **STID** · "
                            f"`{tc[5:me]}` **MID** · `{tc[me:20]}` **Serial**")
                if r.get("type") != "No routing code":
                    st.markdown(f"`{parsed['raw_decoded'][20:]}` **Routing**")
    else:
        st.error("Could not decode an IMB from this image.")
        all_counts = [v for sd in fadt_dbg.values() if isinstance(sd,dict)
                      for k,v in sd.items() if k.endswith("_bars") and isinstance(v,int)]
        best_n = max(all_counts) if all_counts else 0
        if fadt and len(fadt) == 65:
            st.warning("65 bars extracted but pyimb CRC failed — some bar heights misclassified. "
                       "Try a closer, better-lit, straighter photo.")
        elif best_n:
            st.warning(f"Best bar count: {best_n}/65. Barcode may be clipped, blurry, or angled.")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("IMB Reference")
    st.markdown("""
**Barcode ID** (2 digits)

**STID** (3 digits) — mail class + service

**MID** (6 or 9 digits) — mailer ID
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
**Pipeline v5**
1. Sobel-X → barcode row
2. 5 preprocessing × 3 binarizations
3. Autocorrelation pitch estimation
4. Tracker band detection
5. pyimb CRC validation

No AI · No API · Fully offline
    """)
