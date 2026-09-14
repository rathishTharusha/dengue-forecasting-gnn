"""Verify the weekly case data against the Epidemiology Unit's official PDFs.

Every case count in this project traces to ``datasets/output_Dengue Fever.csv``,
a table parsed from the Weekly Epidemiological Report (WER) PDFs. That table was
produced by someone else's parser. This script checks it the only way that means
anything: download the official PDFs and compare the printed numbers, district by
district.

For each report it reads **Table 1** ("Selected notifiable diseases reported by
Medical Officers of Health"), finds the **Dengue** block, and takes row **A**
(cases this week) and row **B** (cumulative for the year). It then compares row A
with the parsed table for the same report.

Run::

    python analysis/_build/verify_case_source.py --per-year 2
    python analysis/_build/verify_case_source.py --files vol_48_no_02-english_1.pdf

PDFs are cached in ``data/raw/wer/`` (git-ignored). Results go to
``analysis/results/source_verification/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RAW_CSV = Path(r"G:\My Drive\Dengue Forecasting GNN\datasets\output_Dengue Fever.csv")
PDF_DIR = REPO / "data" / "raw" / "wer"
OUT = REPO / "analysis" / "results" / "source_verification"
BASE_URL = "https://www.epid.gov.lk/storage/post/pdfs/"

#: Table 1 header spellings -> graph district keys.
HEADER_TO_GRAPH = {"Anuradhapur": "Anuradhapura", "Monaragala": "Moneragala",
                   "Kalmune": "Kalmunai", "NuwaraEliya": "NuwaraEliya", "Nuwara Eliya": "NuwaraEliya"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(name: str) -> Path | None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / name
    if path.exists() and path.read_bytes()[:4] == b"%PDF":
        return path
    req = urllib.request.Request(BASE_URL + name, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            data = r.read()
    except Exception as exc:  # the report is recorded as unreachable, not skipped silently
        print(f"  unreachable {name}: {exc}", flush=True)
        return None
    if data[:4] != b"%PDF":
        print(f"  not a PDF: {name}", flush=True)
        return None
    path.write_bytes(data)
    return path


def read_table1(pdf: Path) -> dict | None:
    """Row A and row B of the Dengue block, keyed by district, plus the title."""
    import pymupdf

    doc = pymupdf.open(pdf)
    for page in doc:
        text = page.get_text()
        if "Dengue" not in text or "RDHS" not in text:
            continue
        title = next((b[4].strip().replace("\n", " ") for b in page.get_text("blocks")
                      if "Table 1" in b[4]), "")
        for table in page.find_tables().tables:
            grid = table.extract()
            header = next((r for r in grid if r and str(r[0] or "").strip().startswith("RDHS")), None)
            start = next((i for i, r in enumerate(grid)
                          if r and r[0] and str(r[0]).strip().startswith("Dengue")), None)
            if header is None or start is None or start + 1 >= len(grid):
                continue
            # Layout, identical 2013-2024: each disease is two rows. The row carrying
            # the disease name is B (cumulative); the row directly below it is A
            # (this week). The row directly ABOVE belongs to the previous disease.
            b_row, a_row = grid[start], grid[start + 1]
            if str(b_row[1]).strip() != "B" or str(a_row[1]).strip() != "A":
                return None

            def parse(row, header=header):
                out = {}
                for name, value in zip(header[2:], row[2:], strict=False):
                    # Text extraction occasionally glues a stray digit to a header
                    # cell ("0Matara"); division names never contain digits.
                    clean = re.sub(r"\d", "", str(name or "")).strip()
                    name = HEADER_TO_GRAPH.get(clean, clean)
                    if not name:
                        continue           # 2023 reports carry an empty trailing column
                    try:
                        out[name] = int(str(value).replace(",", "").strip())
                    except (ValueError, TypeError):
                        out[name] = None
                return out

            return {"title": title, "A": parse(a_row), "B": parse(b_row)}
    return None


def dump_values(raw: pd.DataFrame, name: str) -> dict:
    sub = raw[raw["Source File"].astype(str).str.endswith(name)]
    label = sub["Location Name"].astype(str).str.replace("-", "").str.replace(" ", "")
    out = {}
    for lab, val in zip(label, pd.to_numeric(sub["Cases"], errors="coerce"), strict=True):
        out[lab] = val
    return out


def _canon(label: str) -> str:
    """One spelling per division for both the PDF header and the parsed table."""
    k = str(label).lower().replace(" ", "").replace("-", "")
    aliases = {"paha": "gampaha", "monaragala": "moneragala", "kalmune": "kalmunai",
               "kalmunei": "kalmunai", "srilanka": "srilanka", "94srilanka": "srilanka"}
    k = aliases.get(k, k)
    return k[:6]


def compare(parsed: dict, dump: dict) -> dict:
    """Match printed row A against the parsed table, division by division."""
    by_canon = {_canon(k): v for k, v in dump.items()}
    matched, mismatched, missing = 0, [], []
    for district, printed in parsed["A"].items():
        c = _canon(district)
        if c == "srilan":
            continue
        if printed is None or c not in by_canon or pd.isna(by_canon[c]):
            missing.append(district)
        elif int(by_canon[c]) == printed:
            matched += 1
        else:
            mismatched.append((district, printed, int(by_canon[c])))
    return {"matched": matched, "divisions": matched + len(mismatched) + len(missing),
            "mismatched": mismatched, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-year", type=int, default=2)
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--all", action="store_true", help="verify every report in the table")
    ap.add_argument("--seed", type=int, default=20260914)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    raw = pd.read_csv(RAW_CSV, encoding="utf-8", encoding_errors="replace")
    names = raw["Source File"].astype(str).map(lambda s: s.split("/")[-1]).unique()
    if args.files:
        chosen = list(args.files)
    elif args.all:
        chosen = sorted(names)
    else:
        df = pd.DataFrame({"file": names})
        v = df["file"].str.extract(r"(?i)vol[_ ]?(\d+)[_ ]?no[_ ]?(\d+)").astype(float)
        df["year"] = v[0] + 1973
        df = df.dropna()
        chosen = [f for y, g in df.groupby("year")
                  for f in g.sample(min(args.per_year, len(g)), random_state=args.seed + int(y))["file"]]

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in chosen:
        pdf = download(name)
        rec = {"file": name, "url": BASE_URL + name}
        if pdf is None:
            rec["status"] = "unreachable"
        else:
            rec["sha256"] = sha256(pdf)
            parsed = read_table1(pdf)
            if parsed is None or parsed["A"] is None:
                rec["status"] = "table not parsed"
            else:
                cmp = compare(parsed, dump_values(raw, name))
                rec.update(title=parsed["title"], **cmp)
                b = parsed.get("B") or {}
                rec["row_B_consistent"] = (sum(v for k, v in b.items() if k != "SRILANKA")
                                           == b.get("SRILANKA")) if b else None
                rec["status"] = ("exact" if not cmp["mismatched"] and not cmp["missing"]
                                 else "differs" if cmp["mismatched"] else "incomplete")
        results.append(rec)
        print(f"{rec['status']:16s} {name}  " + (f"{rec.get('matched')}/{rec.get('divisions')} match"
              + (f", differs: {rec['mismatched']}" if rec.get("mismatched") else "") if "matched" in rec else ""),
              flush=True)

    tag = "all" if args.all else ("files" if args.files else "sample")
    (OUT / f"wer_{tag}_verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = sum(r["status"] == "exact" for r in results)
    print(f"\n{ok}/{len(results)} reports match the official PDF exactly on every district")
    return 0


if __name__ == "__main__":
    sys.exit(main())
