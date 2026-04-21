#!/usr/bin/env python3
"""Interactively label captured failed-scan images.

Walks a directory of `failed_*.json` sidecars (default ./data/failed_scans/)
and prompts for the expected IMb decode per image. Labels are written
back into the sidecar JSON in a `labeled` block so the original capture
metadata is preserved.

Usage:
    uv run python tools/label_corpus.py
    uv run python tools/label_corpus.py --dir path/to/scans
    uv run python tools/label_corpus.py --include-labeled   # re-review
    uv run python tools/label_corpus.py --open              # open each image

After labeling, move the image + sidecar pair into
tests/fixtures/regression_corpus/ to fold it into the pytest corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    pass


def validate_labels(raw: dict[str, str]) -> dict[str, Any]:
    """Validate raw prompted strings. Returns normalized dict or raises.

    Rules (from IMb spec):
      - barcode_id : exactly 2 digits
      - stid       : exactly 3 digits
      - mid        : 6 or 9 digits
      - serial     : 9 digits if mid has 6, else 6 digits (mid + serial = 15)
      - routing    : 0, 5, 9, or 11 digits (optional — empty string allowed)
      - notes      : free text (optional)
    """
    def digits(key: str, value: str) -> str:
        v = (value or "").strip()
        if not v.isdigit():
            raise ValidationError(f"{key}: expected digits, got {v!r}")
        return v

    bid = digits("barcode_id", raw.get("barcode_id", ""))
    if len(bid) != 2:
        raise ValidationError(f"barcode_id must be 2 digits, got {len(bid)}")

    stid = digits("stid", raw.get("stid", ""))
    if len(stid) != 3:
        raise ValidationError(f"stid must be 3 digits, got {len(stid)}")

    mid = digits("mid", raw.get("mid", ""))
    if len(mid) not in (6, 9):
        raise ValidationError(f"mid must be 6 or 9 digits, got {len(mid)}")

    serial = digits("serial", raw.get("serial", ""))
    expected_serial = 9 if len(mid) == 6 else 6
    if len(serial) != expected_serial:
        raise ValidationError(
            f"serial must be {expected_serial} digits when mid is {len(mid)}, "
            f"got {len(serial)}"
        )

    routing_raw = (raw.get("routing") or "").strip()
    if routing_raw and not routing_raw.isdigit():
        raise ValidationError(f"routing: expected digits, got {routing_raw!r}")
    if len(routing_raw) not in (0, 5, 9, 11):
        raise ValidationError(
            f"routing must be 0, 5, 9, or 11 digits, got {len(routing_raw)}"
        )

    notes = (raw.get("notes") or "").strip()

    return {
        "barcode_id": bid,
        "service_type": stid,
        "mailer_id": mid,
        "serial": serial,
        "tracking": bid + stid + mid + serial,
        "routing": routing_raw,
        "notes": notes,
    }


def is_labeled(meta: dict) -> bool:
    return bool(meta.get("labeled"))


def iter_sidecars(root: Path):
    return sorted(root.glob("failed_*.json"))


def _open_externally(path: Path) -> None:
    if sys.platform == "linux":
        # WSL2 → explorer.exe; pure linux → xdg-open
        for cmd in (["explorer.exe", str(path)], ["xdg-open", str(path)]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except FileNotFoundError:
                continue
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]


def _prompt_labels_interactive(meta: dict, img_path: Path) -> dict | None:
    print(f"  source   : {meta.get('source')} page={meta.get('page')}")
    print(f"  size     : {meta.get('width')}x{meta.get('height')}")
    print(f"  captured : {meta.get('captured_at')}")
    print(f"  image    : {img_path.resolve()}")
    print()

    while True:
        raw = {
            "barcode_id": input("  Barcode ID [2 digits, blank to skip]: ").strip(),
        }
        if not raw["barcode_id"]:
            return None
        raw["stid"] = input("  STID       [3 digits]: ").strip()
        raw["mid"] = input("  MID        [6 or 9 digits]: ").strip()
        raw["serial"] = input("  Serial     [complementary to MID]: ").strip()
        raw["routing"] = input("  Routing    [0/5/9/11 digits, blank for none]: ").strip()
        raw["notes"] = input("  Notes      [optional]: ").strip()
        try:
            validated = validate_labels(raw)
        except ValidationError as e:
            print(f"  ! {e}  — try again")
            continue

        print()
        print(f"  tracking : {validated['tracking']}")
        if validated["routing"]:
            print(f"  routing  : {validated['routing']}")
        confirm = input("  Save? [Y/n/r=retry]: ").strip().lower()
        if confirm in ("", "y", "yes"):
            return validated
        if confirm in ("r", "retry"):
            continue
        return None


def label_one(sidecar_path: Path, open_image: bool = False) -> bool:
    """Returns True if a label was saved."""
    meta = json.loads(sidecar_path.read_text())
    img_path = sidecar_path.with_suffix(".png")
    if not img_path.exists():
        print(f"  ! image missing: {img_path} — skipping")
        return False

    if open_image:
        _open_externally(img_path)

    labels = _prompt_labels_interactive(meta, img_path)
    if labels is None:
        print("  skipped")
        return False

    meta["labeled"] = labels
    meta["labeled_at"] = datetime.now().isoformat(timespec="seconds")
    sidecar_path.write_text(json.dumps(meta, indent=2))
    print(f"  saved → {sidecar_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="data/failed_scans",
                        help="Directory of failed_*.json sidecars")
    parser.add_argument("--include-labeled", action="store_true",
                        help="Also prompt for already-labeled entries")
    parser.add_argument("--open", action="store_true",
                        help="Open each image in the OS default viewer")
    args = parser.parse_args(argv)

    root = Path(args.dir)
    if not root.exists():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    all_sidecars = iter_sidecars(root)
    if not all_sidecars:
        print(f"No failed_*.json sidecars in {root}")
        return 0

    targets = [
        s for s in all_sidecars
        if args.include_labeled or not is_labeled(json.loads(s.read_text()))
    ]
    if not targets:
        print(f"All {len(all_sidecars)} entries already labeled "
              f"(use --include-labeled to re-review).")
        return 0

    print(f"Labeling {len(targets)} entries from {root}")
    saved = 0
    for i, sidecar in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {sidecar.name}")
        try:
            if label_one(sidecar, open_image=args.open):
                saved += 1
        except (KeyboardInterrupt, EOFError):
            print("\n  interrupted — leaving remaining entries for next run")
            break

    print(f"\nDone. Saved {saved} label(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
