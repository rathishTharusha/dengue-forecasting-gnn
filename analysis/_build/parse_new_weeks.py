"""Build the case series for the weeks after 2024-W10 from the downloaded weekly reports.

Every rule below was fixed and committed BEFORE this script was run on the new reports
(``docs/NEW_WEEKS_RULES.md``). Rules are applied automatically; every correction and every
missing week is logged in ``data/new_weeks/corrections.json``. No model is run on these weeks.

Outputs (``data/new_weeks/``): ``cases.npy`` (weeks, 25, 1) with NaN for missing weeks,
``index.csv`` (one row per expected report), ``corrections.json``, ``qc_report.json``.

    python analysis/_build/parse_new_weeks.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "_build"))
RAW = REPO / "data" / "raw" / "wer_new"
MANIFEST = REPO / "data" / "external" / "wer_new_manifest.csv"
OUT = REPO / "data" / "new_weeks"

# ---- frozen rules (docs/NEW_WEEKS_RULES.md) -------------------------------------------------
FIRST, LAST = (51, 11), (53, 33)            # expected reports, volume and number
FORMULA_CELL_THRESHOLD = 8                  # of 24 inner cells equal to the sum of the two before
MAX_GAP_FROM_PREVIOUS = (5, 9)              # days between consecutive printed week starts
TABLE_TITLE = re.compile(
    r"(\d{1,2})(?:st|nd|rd|th)\s*[^\w\s]+\s*(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]{3,9})\s+(\d{4})\s*\((\d+)", re.S)
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
# Amendment 1 (docs/NEW_WEEKS_RULES.md): the only label spellings accepted besides the exact
# column names. Nothing else is normalised.
LABEL_SPELLINGS = {"Nuwara Eliya": "NuwaraEliya", "Monaragala": "Moneragala"}


def expected_reports() -> list[tuple[int, int]]:
    out, (v, n) = [], FIRST
    while (v, n) <= LAST:
        out.append((v, n))
        n += 1
        if n > 52:
            v, n = v + 1, 1
    return out


def wer_columns() -> list[str]:
    from build_corrected_cases import WER_COLUMNS

    return list(WER_COLUMNS)


def _blank(cell) -> bool:
    return cell is None or not str(cell).strip()


def table_columns(header: list, b_row: list, a_row: list, columns: list[str]) -> tuple | str:
    """Amendment 1: the cells after the two label columns, checked against ``columns``.

    Only *trailing* blank cells are removed (PDF table extraction adds one); nothing
    inside the row is dropped or shifted. The header must then have exactly
    ``len(columns)`` labels, each equal to its expected column (allowing only
    ``LABEL_SPELLINGS``), checked before any count is read. A data row may have
    nothing but blank cells beyond that width. Returns ``(b_cells, a_cells)`` or an
    error string, which makes the week missing under R5.
    """
    n = len(columns)
    head = list(header[2:])
    while head and _blank(head[-1]):
        head.pop()
    if len(head) != n:
        return f"column count {len(head)} after removing trailing blanks"
    labels = [LABEL_SPELLINGS.get(str(c).strip(), str(c).strip()) for c in head]
    if labels != list(columns):
        bad = [(k, lab) for k, (lab, want) in enumerate(zip(labels, columns, strict=True)) if lab != want]
        return f"column labels differ from the expected schema at {bad[:3]}"
    rows = []
    for row in (b_row, a_row):
        cells = list(row[2:])
        if len(cells) < n or not all(_blank(c) for c in cells[n:]):
            return "data row has cells beyond the expected columns"
        rows.append(cells[:n])
    return tuple(rows)


def parse_pdf(path: Path, columns: list[str]) -> dict:
    import pymupdf

    doc = pymupdf.open(path)
    for page in doc:
        text = page.get_text()
        if "Table 1" not in text or "Dengue" not in text:
            continue
        m = TABLE_TITLE.search(text)
        tables = page.find_tables().tables
        if not tables or m is None:
            continue
        grid = tables[0].extract()
        header = next((r for r in grid if r and r[0] == "RDHS"), None)
        i = next((k for k, r in enumerate(grid) if r and r[0] and str(r[0]).startswith("Dengue")), None)
        if header is None or i is None or str(grid[i][1]) != "B" or str(grid[i + 1][1]) != "A":
            continue
        cols = table_columns(header, grid[i], grid[i + 1], columns)
        if isinstance(cols, str):
            return {"error": cols}
        end_day, month, year = int(m.group(2)), MONTHS[m.group(3)[:3].lower()], int(m.group(4))
        end = dt.date(year, month, end_day)
        try:
            b = [int(str(x).replace(",", "")) for x in cols[0]]
            a = [int(str(x).replace(",", "")) for x in cols[1]]
        except ValueError as exc:
            return {"error": f"non-integer cell: {exc}"}
        return {"A": a, "B": b, "end": end, "start": end - dt.timedelta(days=6), "week_label": int(m.group(5))}
    return {"error": "dengue table not found"}


def checks(a: list[int]) -> dict:
    districts, kalmunai, national = sum(a[:25]), a[25], a[26]
    formula = sum(1 for k in range(2, 26) if abs(a[k] - (a[k - 1] + a[k - 2])) <= 1)
    return {"sum_ok": districts + kalmunai == national, "district_sum": districts, "national": national,
            "formula_cells": formula, "formula_flag": formula >= FORMULA_CELL_THRESHOLD}


def _valid(v: list[int] | None) -> bool:
    """R1, R2 and (amendment 2) no negative entry."""
    if v is None or min(v) < 0:
        return False
    c = checks(v)
    return c["sum_ok"] and not c["formula_flag"]


def weekly_vector(cur: dict, prev: dict | None) -> tuple[list[int] | None, str, list[dict]]:
    """The week's 27-cell vector under R1-R3, R7 and amendment 2, its status, and log entries.

    ``cur`` and ``prev`` are parsed reports (``A``, ``B``, ``week_label`` = printed week);
    ``prev`` is the **immediately preceding** report, or None if it was not parsed.
    Row B alone is a weekly count only in printed week 1; otherwise the cumulative
    difference needs a predecessor whose printed week is exactly one less.
    """
    a, b, wk = cur["A"], cur["B"], cur["week_label"]
    log = []
    if wk == 1:
        b_week = list(b)
        if list(b[:25]) != list(a[:25]):
            log.append({"action": "qc", "reason": "printed week 1: row B differs from row A"})
    elif prev is not None and prev["week_label"] == wk - 1:
        b_week = [x - y for x, y in zip(b, prev["B"], strict=True)]
        if min(b_week) < 0:
            log.append({"action": "qc", "reason": "cumulative row decreases from the previous week"})
    else:
        b_week = None
    repeat = prev is not None and list(a[:25]) == list(prev["A"][:25])
    if repeat:
        log.append({"action": "qc", "reason": "row A repeats the previous report's row A (R7)"})
    c = checks(a)
    if c["sum_ok"] and not c["formula_flag"] and not repeat:
        if b_week is not None:
            n_bad = sum(1 for j in range(26) if a[j] != b_week[j])
            if n_bad:
                log.append({"action": "warning", "reason": "row A differs from the cumulative difference",
                            "n": n_bad})
        return list(a), "observed", log
    why = "repeats the previous report (R7)" if repeat else (
        "failed the district sum" if not c["sum_ok"] else "failed the formula-pattern test")
    if _valid(b_week):
        log.append({"action": "corrected", "reason": f"row A {why}", "replacement": "cumulative difference"})
        return b_week, "corrected", log
    log.append({"action": "missing", "reason": f"row A {why} and no valid replacement"})
    return None, "missing", log


def main() -> int:
    columns = wer_columns()
    files = {}
    for r in csv.DictReader(MANIFEST.open(encoding="utf-8")):
        files.setdefault((int(r["volume"]), int(r["number"])), []).append(RAW / r["file"])
    reports = expected_reports()
    cases = np.full((len(reports), 25, 1), np.nan, dtype=np.float32)
    rows, log = [], []
    prev = None          # the immediately preceding report, parsed, or None
    last_start = None    # the last printed start date seen, for the R4 gap log
    for k, key in enumerate(reports):
        row = {"row": k, "volume": key[0], "number": key[1], "status": "missing", "week_start": "",
               "week_end": "", "printed_week": "", "gap_days": "", "check": ""}
        paths = files.get(key, [])
        parsed = [parse_pdf(p, columns) for p in paths]
        ok = [p for p in parsed if "A" in p]
        if not ok:
            log.append({"report": key, "action": "missing", "reason": "no file" if not paths else
                        "; ".join(p.get("error", "?") for p in parsed)})
            rows.append(row)
            prev = None
            continue
        if len({tuple(p["A"]) for p in ok}) > 1:
            log.append({"report": key, "action": "missing", "reason": "two files disagree"})
            rows.append(row)
            prev = None
            continue
        p = ok[0]
        vec, status, entries = weekly_vector(p, prev)
        log.extend({"report": key, **e} for e in entries)
        gap = (p["start"] - last_start).days if last_start is not None else None
        regular = gap is None or MAX_GAP_FROM_PREVIOUS[0] <= gap <= MAX_GAP_FROM_PREVIOUS[1]
        if not regular:
            log.append({"report": key, "action": "warning", "reason": f"printed start date {p['start']} is {gap} days "
                        "after the previous week"})
        row.update(week_start=p["start"].isoformat(), week_end=p["end"].isoformat(), printed_week=p["week_label"],
                   gap_days="" if gap is None else gap, check="ok" if checks(p["A"])["sum_ok"] else "row A sum failed")
        if vec is not None:
            cases[k, :, 0] = np.array(vec[:25], dtype=np.float32)
            row["status"] = status
        rows.append(row)
        prev = p
        last_start = p["start"]
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "cases.npy", cases)
    with (OUT / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "corrections.json").write_text(json.dumps(log, indent=1, default=str), encoding="utf-8")
    counts = {s: sum(r["status"] == s for r in rows) for s in ("observed", "corrected", "missing")}
    gaps = sorted({int(r["gap_days"]) for r in rows if r["gap_days"] != ""})
    qc_flags = {}
    for e in log:
        if e["action"] == "qc":
            qc_flags[e["reason"]] = qc_flags.get(e["reason"], 0) + 1
    qc = {"expected_reports": len(reports), **counts, "gap_days_seen": gaps, "log_entries": len(log),
          "cross_week_qc": qc_flags}
    (OUT / "qc_report.json").write_text(json.dumps(qc, indent=1), encoding="utf-8")
    print(qc)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
