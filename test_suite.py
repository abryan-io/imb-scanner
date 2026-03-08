#!/usr/bin/env python3
"""
Test suite for IMB detection pipeline.

Parses expected values from PDF filenames in the format:
    BarcodeID_STID_MID_SerialNumber.pdf
    e.g., 00_050_107516_047036913.pdf

Runs the detector against each PDF and reports pass/fail.
"""

import os
import sys
import glob
import time
import fitz  # pymupdf
import numpy as np
import cv2
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from cli_app import (
    detect_barcode_region, find_bar_runs, filter_to_65_bars,
    classify_bars_fadt, scan_image_robust, scan_image_robust_all
)
import intelligent_mail_barcode as imb


def parse_expected_from_filename(filename: str) -> dict:
    """
    Parse expected IMB fields from filename.
    Format: BarcodeID_STID_MID_SerialNumber.pdf
    e.g., 00_050_107516_047036913.pdf
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split('_')
    if len(parts) != 4:
        return None

    barcode_id = parts[0]
    stid = parts[1]
    mid = parts[2]
    serial = parts[3]

    return {
        'barcode_id': barcode_id,
        'service_type': stid,
        'mailer_id': mid,
        'serial': serial,
        'tracking': barcode_id + stid + mid + serial,
    }


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF pages to PIL Images using pymupdf."""
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at specified DPI
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num, img))
    doc.close()
    return images


def scan_image_for_imb(pil_img: Image.Image) -> dict:
    """
    Run the robust pipeline on a PIL image.
    Returns result dict or None.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return scan_image_robust(gray)


def scan_image_for_all_imb(pil_img: Image.Image) -> list:
    """
    Run the robust pipeline on a PIL image.
    Returns list of all decoded IMB result dicts.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return scan_image_robust_all(gray)


def run_test_suite(pdf_dir: str = None, scan_func=None):
    """
    Run all PDF test files through the pipeline.
    Returns (results_list, summary_dict).
    """
    if pdf_dir is None:
        pdf_dir = os.path.dirname(__file__)
    if scan_func is None:
        scan_func = scan_image_for_imb

    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))

    results = []
    passed = 0
    failed = 0
    errors = 0

    for pdf_path in pdf_files:
        expected = parse_expected_from_filename(pdf_path)
        if expected is None:
            continue

        filename = os.path.basename(pdf_path)
        start = time.time()

        try:
            images = pdf_to_images(pdf_path, dpi=300)
            all_decodes = []

            # Sort pages: prefer landscape/envelope-shaped pages first
            def page_priority(item):
                _, img = item
                w, h = img.size
                # Landscape pages (likely envelopes) first
                return 0 if w > h else 1

            sorted_images = sorted(images, key=page_priority)

            # Collect decodes from pages, with early exit
            found_match = False
            for page_num, pil_img in sorted_images:
                page_results = scan_image_for_all_imb(pil_img)
                for r in page_results:
                    if r.get('crc_ok'):
                        if not any(d['tracking'] == r['tracking'] for d in all_decodes):
                            all_decodes.append(r)
                            # Early exit if we found the expected barcode
                            if (r['barcode_id'] == expected['barcode_id'] and
                                r['service_type'] == expected['service_type'] and
                                r['mailer_id'] == expected['mailer_id'] and
                                r['serial'] == expected['serial']):
                                found_match = True
                if found_match:
                    break

            elapsed = time.time() - start

            if not all_decodes:
                status = "FAIL"
                detail = "No IMB detected/decoded"
                failed += 1
            else:
                # Check if ANY decode matches the expected values
                matched_result = None
                for result in all_decodes:
                    if (result['barcode_id'] == expected['barcode_id'] and
                        result['service_type'] == expected['service_type'] and
                        result['mailer_id'] == expected['mailer_id'] and
                        result['serial'] == expected['serial']):
                        matched_result = result
                        break

                if matched_result is not None:
                    status = "PASS"
                    n_found = len(all_decodes)
                    detail = f"tracking={matched_result['tracking']}"
                    if n_found > 1:
                        detail += f" ({n_found} barcodes found)"
                    passed += 1
                else:
                    status = "FAIL"
                    # Show the best decode for debugging
                    result = all_decodes[0]
                    mismatches = []
                    if result['barcode_id'] != expected['barcode_id']:
                        mismatches.append(f"bid={result['barcode_id']}!={expected['barcode_id']}")
                    if result['service_type'] != expected['service_type']:
                        mismatches.append(f"stid={result['service_type']}!={expected['service_type']}")
                    if result['mailer_id'] != expected['mailer_id']:
                        mismatches.append(f"mid={result['mailer_id']}!={expected['mailer_id']}")
                    if result['serial'] != expected['serial']:
                        mismatches.append(f"ser={result['serial']}!={expected['serial']}")
                    detail = f"Found {len(all_decodes)} barcode(s), none match. Best: {', '.join(mismatches)}"
                    failed += 1

        except Exception as e:
            elapsed = time.time() - start
            status = "ERROR"
            detail = str(e)
            errors += 1

        results.append({
            'filename': filename,
            'status': status,
            'detail': detail,
            'elapsed': elapsed,
            'expected': expected,
        })

        status_icon = {"PASS": "✓", "FAIL": "✗", "ERROR": "!"}[status]
        print(f"  [{status_icon}] {status:5s} {filename:45s} ({elapsed:.2f}s) {detail}")

    total = passed + failed + errors
    summary = {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'pass_rate': f"{passed}/{total} ({100*passed/max(total,1):.0f}%)",
    }

    return results, summary


if __name__ == "__main__":
    print("=" * 80)
    print("  IMB Detection Test Suite — Baseline")
    print("=" * 80)
    print()

    results, summary = run_test_suite()

    print()
    print("-" * 80)
    print(f"  Results: {summary['pass_rate']}")
    print(f"  Passed: {summary['passed']}  Failed: {summary['failed']}  Errors: {summary['errors']}")
    print("-" * 80)
