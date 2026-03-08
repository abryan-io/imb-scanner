"""
USPS Intelligent Mail Barcode (IMB) Scanner — Streamlit UI
===========================================================
Deterministic pipeline — no AI, no API calls.

Run: streamlit run zapp.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import intelligent_mail_barcode as imb
from stid_table import lookup as stid_lookup, describe as stid_describe
from cli_app import (detect_barcode_region, find_bar_runs, filter_to_65_bars,
                     classify_bars_fadt, scan_image_robust)

# ─── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="USPS IMB Scanner",
    page_icon="📮",
    layout="centered",
)

st.title("📮 USPS Intelligent Mail Barcode Scanner")
st.caption("Deterministic IMB detection · no AI · just pixels and math")


# ─── Main Pipeline ──────────────────────────────────────────────────────────────

def scan_image(pil_img: Image.Image):
    """
    Full pipeline: image → decoded IMB.

    Returns (result_dict | None, annotated_img, crop_img | None,
             fadt_str | None, method_str, debug_info)
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    H, W = gray.shape
    debug_info = {}
    annotated = img_rgb.copy()

    # Use robust multi-strategy scanner
    result = scan_image_robust(gray)

    if result is None:
        # Show best candidate region for debugging
        region = detect_barcode_region(gray)
        if region is not None:
            x, y, w, h = region
            debug_info['region'] = {'x': x, 'y': y, 'w': w, 'h': h}
            pad = 4
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(annotated, "candidate (failed)", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1)
            crop_rgb = img_rgb[y1:y2, x1:x2]
            return None, annotated, crop_rgb, None, "No IMB decoded", debug_info
        return None, annotated, None, None, "No IMB region detected", debug_info

    fadt = result.get('fadt', '')
    debug_info['fadt'] = fadt
    method = "Multi-strategy robust scanner"
    return result, annotated, None, fadt, method, debug_info


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
        gray_dbg = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
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
