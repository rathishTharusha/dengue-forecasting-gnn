"""Rebuild the weekly case series in true date order, from the dated source reports.

``docs/ARRAY_AUDIT.md`` shows the processed array's case column is out of date
order (2023 first), drops 49 reports that exist in the source, stores one week
twice, and sits on a different timeline from its climate columns. This script
builds two corrected case series. Neither modifies the original array.

``reordered``
    Exactly the rows the benchmark used, and nothing else: every array row that
    reproduces a published report, with the duplicate removed, sorted into date
    order. Comparing it with the original isolates the effect of **ordering**.

``rebuilt``
    Every weekly report in the source on a regular grid of report numbers, from
    the first (2013, no. 26) to the last (2024, no. 10). Reports missing from the
    source are filled by log-space interpolation and flagged, so evaluation can
    refuse to score against them. This is the corrected dataset going forward.

Both are **cases only**, shape ``(weeks, 25, 1)``. The climate channels are left
out on purpose: their dates are not yet established, and a covariate block that
silently reused the old row order would reintroduce the misalignment this fixes.

Time is keyed by the report's own **volume and number**, not the dump's parsed
date. Volume N is year 1973 + N, and reports are numbered consecutively within a
volume. The number is not always the epidemiological week -- Vol 48 No 02 covers
"26 Dec 2020 - 01 Jan 2021 (1st Week)" -- but the sequence is what orders the
series, and it is unbroken. The parsed date is wrong for a handful of reports.

The ``rebuilt`` series also applies **verified corrections** to published reports
(``data/external/report_corrections.json``). There is one: the week-395 spike is
a spreadsheet formula error in the published table, not a reporting backlog.
See :func:`extract_week1_correction`.

District labels follow the authors' ``Configs/disease_config.json``, including its
exclusion of the Kalmunai health division. Kalmunai lies inside Ampara, so the
benchmark's Ampara target is Ampara RDHS only. That is kept here for
comparability and flagged, not silently changed.

Run::

    python analysis/_build/build_corrected_cases.py \
        --raw "datasets/output_Dengue Fever.csv"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_seir_inputs as bsi  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
CAL = REPO / "data" / "external" / "calendar_index.csv"
OUT = REPO / "data" / "corrected"

#: The week-395 reporting spike, keyed by report rather than by row: its row
#: index is different in every dataset.
ARTIFACT_REPORT = (2021, 2)

#: Verified corrections to published reports, applied to ``rebuilt`` only.
CORRECTIONS = REPO / "data" / "external" / "report_corrections.json"

#: RDHS column order of Table 1 in the Weekly Epidemiological Report.
WER_COLUMNS = ["Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya", "Galle",
               "Hambantota", "Matara", "Jaffna", "Kilinochchi", "Mannar", "Vavuniya",
               "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee", "Kurunegala", "Puttalam",
               "Anuradhapura", "Polonnaruwa", "Badulla", "Moneragala", "Ratnapura", "Kegalle",
               "Kalmunai", "SRILANKA"]


def extract_week1_correction(pdf: Path) -> dict:
    """Recover the true weekly counts for WER Vol 48 No 02 from its own table.

    The published Dengue row **A** (cases this week) in this report is a
    spreadsheet formula error: 15 of its 24 inner cells equal the sum of the two
    cells before them (18, 18, 36, 54, 90, 144, 234, ...), its districts sum to
    7,165, and its national cell reads 35. No other report in the 552 shows the
    pattern in more than 3 cells. Row **B** (cumulative for the year) is
    consistent -- its districts plus Kalmunai sum exactly to its national total
    of 351 -- and because this is the first week of the year, cumulative equals
    weekly. So row B *is* the weekly count.

    Requires the PDF: https://www.epid.gov.lk/storage/post/pdfs/vol_48_no_02-english_1.pdf
    """
    import pymupdf

    page = pymupdf.open(pdf)[2]
    grid = page.find_tables().tables[0].extract()
    header = next(r for r in grid if r and r[0] == "RDHS")
    i = next(k for k, r in enumerate(grid) if r and r[0] and str(r[0]).startswith("Dengue"))
    if str(grid[i][1]) != "B" or str(grid[i + 1][1]) != "A":
        raise SystemExit("unexpected row layout in the Dengue block")
    labels = [str(c) for c in header[2:]]
    if len(labels) != len(WER_COLUMNS):
        raise SystemExit(f"unexpected column count {len(labels)}")
    b = [int(x) for x in grid[i][2:]]
    a = [int(x) for x in grid[i + 1][2:]]
    districts_b = sum(b[:25])
    if districts_b + b[25] != b[26]:
        raise SystemExit("row B does not sum to its national total; refusing to use it")
    formula_cells = sum(1 for k in range(2, 26) if abs(a[k] - (a[k - 1] + a[k - 2])) <= 1)
    return {
        "report": {"year": 2021, "volume": 48, "number": 2,
                   "covers": "26 Dec 2020 - 01 Jan 2021 (1st week)"},
        "source": "https://www.epid.gov.lk/storage/post/pdfs/vol_48_no_02-english_1.pdf",
        "published_row_A": dict(zip(WER_COLUMNS, a, strict=True)),
        "published_row_B": dict(zip(WER_COLUMNS, b, strict=True)),
        "replacement": dict(zip(WER_COLUMNS[:25], b[:25], strict=True)),
        "evidence": {"row_A_formula_cells": formula_cells,
                     "row_A_district_sum": sum(a[:25]), "row_A_national": a[26],
                     "row_B_district_sum_plus_kalmunai": districts_b + b[25],
                     "row_B_national": b[26]},
        "reason": "Row A is a spreadsheet formula error; row B is year-to-date, which "
                  "equals the weekly count in the first week of the year.",
    }


def load_reports(raw_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the dump into one row per (year, report number, district).

    Returns ``(district_table, national_table)``. The national table carries the
    Epidemiology Unit's own ``SRILANKA`` total and the Kalmunai division, used
    only to check the districts sum correctly.
    """
    names = bsi.district_names()
    raw = pd.read_csv(raw_csv, encoding="utf-8", encoding_errors="replace")
    vol = raw["Source File"].astype(str).str.extract(r"(?i)vol[_ ]?(\d+)[_ ]?no[_ ]?(\d+)")
    raw["year"] = vol[0].astype(int) + 1973
    raw["week_no"] = vol[1].astype(int)
    raw["parsed_date"] = pd.to_datetime(raw["TimeStampStart"], format="%d-%b-%Y %H:%M:%S",
                                        errors="coerce").dt.normalize()
    raw["cases"] = pd.to_numeric(raw["Cases"], errors="coerce")
    label = raw["Location Name"].astype(str).str.lower().str.replace("-", "").str.replace(" ", "")

    national = raw[label.isin(["srilanka", "94srilanka"])].groupby(["year", "week_no"])["cases"].max()
    kalmunai = raw[label.isin(["kalmune", "kalmunei", "kalmunai"])].groupby(["year", "week_no"])["cases"].max()

    raw["district"] = raw["Location Name"].map(lambda x: bsi._canonical(x, names))
    dist = raw.dropna(subset=["district"])
    conflicts = dist.groupby(["year", "week_no", "district"])["cases"].nunique()
    if (conflicts > 1).any():
        raise SystemExit(f"{int((conflicts > 1).sum())} report-district cells disagree across files")

    table = dist.groupby(["year", "week_no", "district"]).agg(
        cases=("cases", "first"), parsed_date=("parsed_date", "min"),
        source=("Source File", "first")).reset_index()
    nat = pd.DataFrame({"national": national, "kalmunai": kalmunai}).reset_index()
    return table, nat


def report_dates(table: pd.DataFrame) -> pd.Series:
    """One start date per report, repairing the dump's few wrong parsed dates.

    A parsed date is trusted only if its year agrees with the report's volume year
    (allowing a late-December start for week 1). Anything else is replaced by
    stepping seven days from the nearest trusted neighbour.
    """
    per = table.groupby(["year", "week_no"])["parsed_date"].min()
    ok = (per.dt.year == per.index.get_level_values("year")) | (
        (per.dt.month == 12) & (per.index.get_level_values("year") == per.dt.year + 1))
    good = per[ok]
    keys = list(per.index)
    fixed = per.copy()
    for i, k in enumerate(keys):
        if ok.loc[k]:
            continue
        for step in range(1, len(keys)):
            for j in (i - step, i + step):
                if 0 <= j < len(keys) and keys[j] in good.index:
                    fixed.loc[k] = good.loc[keys[j]] + pd.Timedelta(weeks=i - j)
                    break
            else:
                continue
            break
    return fixed


def build_rebuilt(table: pd.DataFrame, nat: pd.DataFrame,
                  corrections: list[dict] | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    names = bsi.district_names()
    corrected_keys = set()
    table = table.copy()
    for fix in corrections or []:
        key = (fix["report"]["year"], fix["report"]["number"])
        mask = (table.year == key[0]) & (table.week_no == key[1])
        for district, value in fix["replacement"].items():
            table.loc[mask & (table.district == district), "cases"] = value
        corrected_keys.add(key)
    last_no = table.groupby("year")["week_no"].max()
    first_year, last_year = int(table.year.min()), int(table.year.max())
    first_no = int(table.loc[table.year == first_year, "week_no"].min())
    grid = []
    for y in range(first_year, last_year + 1):
        # A year has 52 epidemiological weeks, or 53 when the source reports a week 53.
        n_weeks = max(52, int(last_no.get(y, 52)))
        start = first_no if y == first_year else 1
        stop = int(last_no[y]) if y == last_year else n_weeks
        grid.extend((y, n) for n in range(start, stop + 1))

    wide = table.pivot_table(index=["year", "week_no"], columns="district", values="cases",
                             aggfunc="first").reindex(pd.MultiIndex.from_tuples(grid, names=["year", "week_no"]))
    wide = wide.reindex(columns=names)
    observed = wide.notna().all(axis=1)

    # Fill whole missing reports in log space, per district, and flag them.
    filled = np.expm1(np.log1p(wide).interpolate(limit_direction="both"))
    dates = report_dates(table).reindex(wide.index)
    dates = dates.interpolate() if dates.isna().any() else dates
    if dates.isna().any():
        known = dates.dropna()
        pos = {k: i for i, k in enumerate(wide.index)}
        for k in dates[dates.isna()].index:
            j = min(known.index, key=lambda kk: abs(pos[kk] - pos[k]))
            dates.loc[k] = known.loc[j] + pd.Timedelta(weeks=pos[k] - pos[j])

    index = pd.DataFrame({
        "row": range(len(wide)),
        "year": [k[0] for k in wide.index],
        "week_no": [k[1] for k in wide.index],
        "week_start": pd.to_datetime(dates.to_numpy()).date,
        "status": [("corrected" if k in corrected_keys else "observed") if obs else "imputed"
                   for k, obs in zip(wide.index, observed.to_numpy(), strict=True)],
        # A corrected report is no longer an artifact; only an uncorrected one is.
        "is_artifact": [k == ARTIFACT_REPORT and k not in corrected_keys for k in wide.index],
    })

    # Integrity: districts + Kalmunai must equal the Epidemiology Unit's own total.
    check = wide.sum(axis=1).to_frame("districts").join(nat.set_index(["year", "week_no"]))
    check = check[observed & check.national.notna()]
    gap = (check.districts + check.kalmunai.fillna(0) - check.national).abs()
    index.attrs["national_check"] = {"reports_checked": int(len(check)),
                                     "exact": int((gap == 0).sum()),
                                     "max_abs_gap": float(gap.max())}
    return filled.to_numpy()[..., None].astype(np.float32), index


def build_reordered() -> tuple[np.ndarray, pd.DataFrame]:
    arr = np.nan_to_num(np.load(NPY, allow_pickle=True))[..., 5]
    cal = pd.read_csv(CAL)
    keep = cal[cal.abs_diff.eq(0)].copy()
    keep = keep.sort_values(["report_year", "report_number", "array_week"])
    keep = keep.drop_duplicates(["report_year", "report_number"], keep="first")
    rows = keep["array_week"].to_numpy()
    index = pd.DataFrame({
        "row": range(len(rows)),
        "original_row": rows,
        "year": keep["report_year"].to_numpy(),
        "week_no": keep["report_number"].to_numpy(),
        "week_start": keep["report_week_start"].to_numpy(),
        "status": "observed",
        "is_artifact": [(int(y), int(n)) == ARTIFACT_REPORT
                        for y, n in zip(keep.report_year, keep.report_number, strict=True)],
    })
    return arr[rows][..., None].astype(np.float32), index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--wer-pdf", type=Path, default=None,
                    help="WER Vol 48 No 02 PDF; regenerates data/external/report_corrections.json")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.wer_pdf:
        CORRECTIONS.write_text(json.dumps([extract_week1_correction(args.wer_pdf)], indent=2),
                               encoding="utf-8")
    corrections = json.loads(CORRECTIONS.read_text(encoding="utf-8")) if CORRECTIONS.exists() else []

    table, nat = load_reports(args.raw)
    rebuilt, rindex = build_rebuilt(table, nat, corrections)
    reordered, oindex = build_reordered()

    np.save(OUT / "rebuilt_cases.npy", rebuilt)
    rindex.to_csv(OUT / "rebuilt_index.csv", index=False)
    np.save(OUT / "reordered_cases.npy", reordered)
    oindex.to_csv(OUT / "reordered_index.csv", index=False)

    # Reordered rows must reproduce the rebuilt series exactly where both exist.
    rkey = {(int(y), int(n)): i for i, (y, n) in enumerate(zip(rindex.year, rindex.week_no, strict=True))}
    both = [(i, rkey[(int(y), int(n))]) for i, (y, n) in
            enumerate(zip(oindex.year, oindex.week_no, strict=True)) if (int(y), int(n)) in rkey]
    mism = sum(int(not np.array_equal(reordered[i, :, 0], rebuilt[j, :, 0])) for i, j in both)

    summary = {
        "rebuilt": {"weeks": int(len(rindex)), "first": f"{rindex.year.iloc[0]}-W{rindex.week_no.iloc[0]}",
                    "last": f"{rindex.year.iloc[-1]}-W{rindex.week_no.iloc[-1]}",
                    "imputed_weeks": int((rindex.status == "imputed").sum()),
                    "corrected_weeks": int((rindex.status == "corrected").sum()),
                    "artifact_row": (int(rindex.index[rindex.is_artifact][0])
                                     if rindex.is_artifact.any() else None),
                    "national_total_check": rindex.attrs["national_check"]},
        "reordered": {"weeks": int(len(oindex)), "artifact_row": int(oindex.index[oindex.is_artifact][0]),
                      "rows_dropped_from_original": 459 - int(len(oindex))},
        "reordered_vs_rebuilt": {"shared_weeks": len(both), "value_mismatches": mism},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
