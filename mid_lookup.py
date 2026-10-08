"""Mailer ID → company lookup, loaded from MID_Lkp.xlsx."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl

MID_LKP_PATH = Path(__file__).parent / "MID_Lkp.xlsx"


def normalize_mid(value: Any) -> str:
    """
    Canonical MID string. Excel stores MIDs as numbers, dropping leading
    zeros (000128 → 128). 9-digit MIDs start with 9 and 6-digit MIDs never
    do, so any shorter value is a 6-digit MID and is zero-padded back.
    """
    s = str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
    return s.zfill(6) if s.isdigit() and len(s) < 6 else s


def load_mid_table(path: Path = MID_LKP_PATH) -> dict[str, dict[str, str]]:
    """Return {MID: {'BID', 'Company', 'Address'}}; empty if the file is missing."""
    if not Path(path).exists():
        return {}
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = list(wb.active.iter_rows(min_row=2, values_only=True))
    wb.close()
    table = {}
    for row in rows:
        if len(row) >= 4 and row[0] is not None:
            table[normalize_mid(row[0])] = {
                'BID': str(int(row[1])) if isinstance(row[1], (int, float)) else str(row[1] or ''),
                'Company': str(row[2] or ''),
                'Address': str(row[3] or ''),
            }
    return table
