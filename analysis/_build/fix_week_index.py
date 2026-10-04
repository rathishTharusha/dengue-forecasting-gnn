"""Repair data/corrected/rebuilt_index.csv without the raw report dump.

The first build left eight rows with a parsed date that had the right year and the wrong
month (``build_corrected_cases.snap_to_weekly_grid`` now prevents this). The raw dump is
not in the repository, but the rows are consecutive report weeks, so the correct index is
the weekly grid ``2013-06-15 + 7 * row``. This script checks that the grid agrees with
every currently correct row and differs only where a date was wrong, then rewrites the CSV.

Run:  python analysis/_build/fix_week_index.py            (dry run, prints the comparison)
      python analysis/_build/fix_week_index.py --write    (also writes the CSV and a report)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
INDEX = REPO / "data" / "corrected" / "rebuilt_index.csv"
REPORT = REPO / "data" / "corrected" / "week_index_fix.json"
ANCHOR = pd.Timestamp("2013-06-15")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    old = pd.read_csv(INDEX, parse_dates=["week_start"])
    grid = ANCHOR + pd.to_timedelta(7 * old["row"], unit="D")
    bad = old[old["week_start"] != grid]
    print(f"rows {len(old)}; already on the grid {len(old) - len(bad)}; different {len(bad)}")
    changes = [{"row": int(r.row), "year": int(r.year), "week_no": int(r.week_no),
                "old": r.week_start.date().isoformat(), "new": grid[i].date().isoformat(),
                "shift_days": int((grid[i] - r.week_start).days)} for i, r in bad.iterrows()]
    for c in changes:
        print(" ", c)
    # Every new date must still be in the report's own volume year (or the late-December start of week 1).
    new_year = grid.dt.year
    ok = (new_year == old["year"]) | ((grid.dt.month == 12) & (old["year"] == new_year + 1))
    if not ok.all():
        print("STOP: grid dates disagree with the volume year on rows", old.loc[~ok, "row"].tolist())
        return 1
    if (grid.diff().dropna().dt.days != 7).any() or grid.iloc[-1] != pd.Timestamp("2024-02-24"):
        print("STOP: the grid does not end on 2024-02-24")
        return 1
    if args.write:
        new = old.copy()
        new["week_start"] = grid.dt.date
        new.to_csv(INDEX, index=False)
        REPORT.write_text(json.dumps({"anchor": ANCHOR.date().isoformat(), "rows_changed": changes}, indent=2),
                          encoding="utf-8")
        print("wrote", INDEX.name, "and", REPORT.name)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
