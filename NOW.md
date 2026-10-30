# NOW

## Where I left off

Branch `fix/2026-W44-audit`, committed: every 2026-W44 audit finding plus the code fixes from the
review (PDF page cap, upload errors, sidebar MID rule, labeler MID rule, xdist history rows,
devcontainer, wheel contents). Python 3.13, OpenCV 5, Pillow 13, Streamlit 1.66.1. Streamlit Cloud
on 3.11 is within the new floor. The 20-page PDF limit is confirmed. `audit/2026-W43` is deleted.

The devcontainer itself is untested (no Docker here).

## Next step

Merge `fix/2026-W44-audit` into master.

## Open questions

- The Stop hook's `just test` appends to the tracked `history.jsonl`, which changes the tree hash it
  caches on, so it re-runs the suite at every turn end. Keep history out of git, or out of the hash?
