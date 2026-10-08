#!/usr/bin/env python3
"""List decoded Mailer IDs that aren't in MID_Lkp.xlsx yet.

Reads the records written by unknown_mid_log (local or R2, per
FAILED_SCAN_BACKEND in .env) and prints tab-separated rows, most-seen
first, ready to paste into a spreadsheet. MIDs added to MID_Lkp.xlsx since
they were logged are hidden.

Usage:
    uv run python tools/unknown_mids.py              # backend from .env
    uv run python tools/unknown_mids.py --backend r2 # what the Streamlit app logged
    uv run python tools/unknown_mids.py --all        # include MIDs now in the lookup
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

COLUMNS = ["mid", "pieces", "first_seen", "last_seen", "service_types", "barcode_ids", "sources"]


def format_rows(records: list[dict], known_mids: set[str], include_known: bool) -> list[str]:
    rows = ["\t".join(COLUMNS)]
    for r in records:
        if r["mid"] in known_mids and not include_known:
            continue
        rows.append(
            "\t".join(
                [
                    r["mid"],
                    str(len(r["pieces"])),
                    r["first_seen"],
                    r.get("last_seen", r["first_seen"]),
                    ",".join(r["service_types"]),
                    ",".join(r["barcode_ids"]),
                    "; ".join(r["sources"]),
                ]
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backend", choices=["local", "r2"], help="override FAILED_SCAN_BACKEND from .env"
    )
    parser.add_argument(
        "--all", action="store_true", help="include MIDs that are now in MID_Lkp.xlsx"
    )
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    if args.backend:
        os.environ["FAILED_SCAN_BACKEND"] = args.backend

    from mid_lookup import load_mid_table
    from unknown_mid_log import list_unknown_mids

    rows = format_rows(list_unknown_mids(), set(load_mid_table()), args.all)
    print("\n".join(rows))
    print(f"{len(rows) - 1} MID(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
