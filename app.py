"""
USPS Intelligent Mail Barcode (IMB) Scanner — Streamlit UI
===========================================================
Deterministic pipeline — no AI, no API calls.

Run: streamlit run app.py
"""

import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import openpyxl
import intelligent_mail_barcode as imb
from stid_table import lookup as stid_lookup, describe as stid_describe
from cli_app import (detect_barcode_region, find_bar_runs, filter_to_65_bars,
                     classify_bars_fadt, scan_image_robust)

# ─── MID Lookup Table ────────────────────────────────────────────────────────────

MID_LKP_PATH = os.path.join(os.path.dirname(__file__), "MID_Lkp.xlsx")

@st.cache_data
def load_mid_table():
    """Load MID lookup table from Excel. Returns dict keyed by MID string."""
    if not os.path.exists(MID_LKP_PATH):
        return {}
    wb = openpyxl.load_workbook(MID_LKP_PATH, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    wb.close()
    table = {}
    for row in rows:
        if len(row) >= 4 and row[0] is not None:
            mid_key = str(int(row[0])) if isinstance(row[0], (int, float)) else str(row[0]).strip()
            table[mid_key] = {
                'BID': str(int(row[1])) if isinstance(row[1], (int, float)) else str(row[1] or ''),
                'Company': str(row[2] or ''),
                'Address': str(row[3] or ''),
            }
    return table

mid_table = load_mid_table()


# ─── PDF Conversion ─────────────────────────────────────────────────────────────

def pdf_to_pil_images(pdf_bytes, dpi=300):
    """Convert PDF bytes to list of (page_num, PIL Image) tuples."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num + 1, img))  # 1-indexed
    doc.close()
    return images


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


# ─── Result Display (shared by image & PDF paths) ───────────────────────────────

def display_decode_result(result):
    """Display a successfully decoded IMB result with MID lookup."""
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

    # MID lookup
    mid_info = mid_table.get(mid)
    if mid_info:
        st.subheader("Mailer Info (MID Lookup)")
        m1, m2 = st.columns(2)
        m1.metric("Company", mid_info['Company'])
        m2.metric("BID", mid_info['BID'])
        st.write(f"**Address:** {mid_info['Address']}")
    else:
        st.caption("MID not found in lookup table")

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


def display_scan_failure(fadt, debug_info):
    """Display failure info when no IMB could be decoded."""
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


def display_image_scan(pil_img):
    """Run scan and display full results for a single image."""
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
        display_decode_result(result)
    else:
        display_scan_failure(fadt, debug_info)


# ─── Streamlit UI ────────────────────────────────────────────────────────────────

tab_upload, tab_camera = st.tabs(["Upload File", "Camera"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a mailpiece image or PDF",
        type=["jpg", "jpeg", "png", "webp", "heic", "pdf"],
        help="Accepts images and PDFs. Clear, well-lit photos work best.",
        key="file_uploader",
    )

with tab_camera:
    camera_img = st.camera_input("Take a photo of a mailpiece")

# ─── Process Upload or Camera ────────────────────────────────────────────────────

pil_img = None
uploaded_pdf = None

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        uploaded_pdf = uploaded_file
    else:
        pil_img = Image.open(uploaded_file)
elif camera_img:
    pil_img = Image.open(camera_img)

if pil_img:
    display_image_scan(pil_img)

# ─── Process PDF Upload ─────────────────────────────────────────────────────────

if uploaded_pdf:
    pdf_bytes = uploaded_pdf.read()

    with st.spinner("Converting PDF pages to images..."):
        try:
            page_images = pdf_to_pil_images(pdf_bytes)
        except Exception as e:
            st.error(f"Failed to open PDF: {e}")
            page_images = []

    if page_images:
        total_pages = len(page_images)
        st.write(f"**{total_pages} page{'s' if total_pages != 1 else ''}** detected. Scanning for IMB barcodes...")

        found_any = False
        progress = st.progress(0)

        for idx, (page_num, page_img) in enumerate(page_images):
            progress.progress((idx + 1) / total_pages)

            img_rgb = np.array(page_img.convert("RGB"))
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            result = scan_image_robust(gray)

            if result is not None:
                found_any = True
                st.subheader(f"Page {page_num}")

                # Show page thumbnail alongside results
                thumb_col, result_col = st.columns([1, 2])
                with thumb_col:
                    st.image(page_img, caption=f"Page {page_num}", use_container_width=True)
                with result_col:
                    display_decode_result(result)

                    fadt = result.get('fadt', '')
                    if fadt:
                        with st.expander("FADT String"):
                            st.code(fadt, language=None)

                st.divider()

        progress.empty()

        if not found_any:
            st.warning("No IMB barcode found on any page of this PDF.")

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
