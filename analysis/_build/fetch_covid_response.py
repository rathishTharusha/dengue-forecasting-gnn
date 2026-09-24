"""Download the COVID-19 policy and mobility record, weekly, on our calendar.

Why this exists: the largest single loss in the frozen 9-origin protocol is the
window 2019-11 to 2020-06, where national cases collapse from 3057 to 299 a week
as movement restrictions begin (EXP-035, EXP-036). No compartmental model can
anticipate that from case history. Both series below were published within days
of the events they describe, so a forecaster at the time could have used them:
they are covariates under rule R2, **not** oracle knowledge under R4.

Sources
-------
1. **Policy stringency** -- Oxford COVID-19 Government Response Tracker (OxCGRT),
   Blavatnik School of Government, University of Oxford. Daily index 0-100 over
   closures, movement restrictions and stay-at-home orders. Licence CC BY 4.0.
   https://github.com/OxCGRT/covid-policy-tracker
2. **Mobility** -- Google COVID-19 Community Mobility Reports, percent change
   from a pre-pandemic baseline in six place categories. Free to use with
   attribution. https://www.google.com/covid19/mobility/

Both are **national** for Sri Lanka. Google published no sub-region rows for this
country, so the mobility columns are the same for every district; they can only
explain national-level timing, not spatial differences.

Coverage, and what is *not* filled
----------------------------------
Stringency runs 2020-01-01 to 2022-12-31, mobility 2020-02-15 to 2022-10-15. Weeks
outside those ranges are left **NaN**, never zero-filled -- rule R3. A modelling
script may treat pre-2020 stringency as 0 because these policies did not exist
then, but that is a modelling decision and belongs in the model, not in the data
file.

Usage::

    python analysis/_build/fetch_covid_response.py            # download + build
    python analysis/_build/fetch_covid_response.py --offline  # rebuild from data/raw
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "external" / "covid_response_weekly.csv"
INDEX = REPO / "data" / "corrected" / "rebuilt_index.csv"

OXCGRT_URL = ("https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/"
              "master/data/timeseries/stringency_index_avg.csv")
MOBILITY_URL = "https://www.gstatic.com/covid19/mobility/{year}_LK_Region_Mobility_Report.csv"
MOBILITY_YEARS = (2020, 2021, 2022)

MOBILITY_COLS = {
    "retail_and_recreation_percent_change_from_baseline": "mobility_retail_recreation",
    "grocery_and_pharmacy_percent_change_from_baseline": "mobility_grocery_pharmacy",
    "parks_percent_change_from_baseline": "mobility_parks",
    "transit_stations_percent_change_from_baseline": "mobility_transit",
    "workplaces_percent_change_from_baseline": "mobility_workplaces",
    "residential_percent_change_from_baseline": "mobility_residential",
}


def fetch(url: str, dest: Path, offline: bool) -> Path:
    """Download `url` to `dest` unless it is already there (or --offline)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (offline or dest.stat().st_size > 0):
        print(f"  have {dest.relative_to(REPO)} ({dest.stat().st_size:,} bytes)")
        return dest
    if offline:
        raise SystemExit(f"--offline but {dest} is missing")
    print(f"  GET {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())
    print(f"  wrote {dest.relative_to(REPO)} ({dest.stat().st_size:,} bytes)")
    return dest


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def daily_stringency(path: Path) -> pd.Series:
    """LKA's daily stringency index, indexed by date."""
    wide = pd.read_csv(path)
    row = wide[wide["country_code"] == "LKA"]
    if row.empty:
        raise SystemExit("no LKA row in the OxCGRT file")
    if len(row) > 1:                      # keep the national total if regions appear
        row = row[row["jurisdiction"] == "NAT_TOTAL"]
    series = row.iloc[0, 6:]
    series.index = pd.to_datetime(series.index, format="%d%b%Y")
    return pd.to_numeric(series, errors="coerce").dropna()


def daily_mobility(paths: list[Path]) -> pd.DataFrame:
    """Sri Lanka's national mobility change, indexed by date."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, parse_dates=["date"])
        nat = df[df["sub_region_1"].isna() & df["sub_region_2"].isna()]
        frames.append(nat.set_index("date")[list(MOBILITY_COLS)])
    out = pd.concat(frames).sort_index()
    return out.rename(columns=MOBILITY_COLS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="rebuild from files already in data/raw")
    args = ap.parse_args()

    print("sources")
    ox_path = fetch(OXCGRT_URL, RAW / "oxcgrt" / "stringency_index_avg.csv", args.offline)
    mob_paths = [fetch(MOBILITY_URL.format(year=y),
                       RAW / "google_mobility" / f"{y}_LK_Region_Mobility_Report.csv",
                       args.offline) for y in MOBILITY_YEARS]

    sti = daily_stringency(ox_path)
    mob = daily_mobility(mob_paths)
    print(f"\nstringency: {len(sti)} days, {sti.index.min().date()} to {sti.index.max().date()}")
    print(f"mobility:   {len(mob)} days, {mob.index.min().date()} to {mob.index.max().date()}")

    idx = pd.read_csv(INDEX, parse_dates=["week_start"])
    rows = []
    for _, r in idx.iterrows():
        a = r["week_start"]
        b = a + pd.Timedelta(days=7)
        s = sti[(sti.index >= a) & (sti.index < b)]
        m = mob[(mob.index >= a) & (mob.index < b)]
        row = {"row": int(r["row"]), "week_start": a.date().isoformat(),
               "stringency_index": round(float(s.mean()), 2) if len(s) else np.nan,
               "stringency_max": round(float(s.max()), 2) if len(s) else np.nan,
               "stringency_days": int(len(s))}
        for c in MOBILITY_COLS.values():
            row[c] = round(float(m[c].mean()), 2) if len(m) and m[c].notna().any() else np.nan
        row["mobility_days"] = int(len(m))
        rows.append(row)

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    have_s = int(out.stringency_index.notna().sum())
    have_m = int(out.mobility_retail_recreation.notna().sum())
    print(f"\nwrote {OUT.relative_to(REPO)}")
    print(f"  {len(out)} weeks | stringency on {have_s} | mobility on {have_m}")
    peak = out.loc[out.stringency_index.idxmax()] if have_s else None
    if peak is not None:
        print(f"  strictest week: {peak.week_start}  index {peak.stringency_index}")
    print("\nchecksums for data/external/source_manifest.csv")
    for p in [ox_path, *mob_paths]:
        print(f"  {p.relative_to(RAW)}  {p.stat().st_size:>9,}  {sha256(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
