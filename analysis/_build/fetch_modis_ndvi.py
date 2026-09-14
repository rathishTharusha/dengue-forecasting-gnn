"""Fetch a dated NDVI record per district from MODIS, and align it to the report weeks.

Why this exists
---------------
Liu et al. (2025) found mean NDVI among the strongest drivers of dengue in Sri
Lanka. ERA5 has no vegetation index, and the processed array's NDVI channel is
suspect: ``docs/ARRAY_AUDIT.md`` shows its companion "lagged" channels were
shifted the wrong way. NASA's usual download tools need an Earthdata login; the
ORNL DAAC MODIS/VIIRS subset web service does not.

Source
------
MOD13Q1 v061, Terra MODIS Vegetation Indices, 16-day composite, 250 m
(Didan K. 2021, doi:10.5067/MODIS/MOD13Q1.061), via the ORNL DAAC MODIS and VIIRS
Land Product Subsets RESTful web service: https://modis.ornl.gov/data/modis_webservice.html
Each composite is already a 16-day maximum-value composite, which suppresses cloud.

Sampling
--------
A box of +/- ``KM`` km around the same interior point per district used for ERA5,
averaging valid land pixels (fill values and water are excluded). A box is a
local sample, not a district mean, and is recorded as such.

Weekly alignment -- as-of, never interpolated
--------------------------------------------
Each report week takes the **latest composite that was already available when the
week began**: its 16-day window must have ended, plus ``AVAILABILITY_DAYS`` for
processing and release, before the week's first day. Nothing is interpolated.
Interpolating between composites would both invent values and read the *next*
composite, which lies in the future relative to the week.

``AVAILABILITY_DAYS`` is an assumption, not a measured release latency, and is set
conservatively to one full composite period (16 days).

Run::

    python analysis/_build/fetch_modis_ndvi.py fetch     # resumable; ~650 requests
    python analysis/_build/fetch_modis_ndvi.py weekly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
POINTS = REPO / "data" / "raw" / "era5_openmeteo" / "points.json"
RAW = REPO / "data" / "raw" / "modis_ndvi"
INDEX = REPO / "data" / "corrected" / "rebuilt_index.csv"
OUT = REPO / "data" / "external"

API = "https://modis.ornl.gov/rst/api/v1"
PRODUCT, BAND = "MOD13Q1", "250m_16_days_NDVI"
KM = 5
START, END = "2012-09-01", "2024-03-31"   # starts early so lags of ~17 weeks have data
SCALE, FILL = 0.0001, -3000


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt == 5:
                raise
            wait = 30 * (attempt + 1)
            print(f"    retry in {wait}s after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _cached(path: Path) -> bool:
    """A cache file counts only if it parses; a write cut off mid-way is redone."""
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except json.JSONDecodeError:
        path.unlink()
        return False


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _fetch_district(name: str, lat: float, lon: float) -> str:
    dates_path = RAW / f"{name}__dates.json"
    if not _cached(dates_path):
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon})
        _write_atomic(dates_path, get_json(f"{API}/{PRODUCT}/dates?{q}"))
    dates = [d for d in json.loads(dates_path.read_text(encoding="utf-8"))["dates"]
             if START <= d["calendar_date"] <= END]
    chunks = [dates[i:i + 10] for i in range(0, len(dates), 10)]
    for i, chunk in enumerate(chunks):
        path = RAW / f"{name}__{i:03d}.json"
        if _cached(path):
            continue
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "band": BAND,
                                    "startDate": chunk[0]["modis_date"],
                                    "endDate": chunk[-1]["modis_date"],
                                    "kmAboveBelow": KM, "kmLeftRight": KM})
        _write_atomic(path, get_json(f"{API}/{PRODUCT}/subset?{q}"))
        time.sleep(1.0)
    return f"  {name:13s} {len(dates)} composites in {len(chunks)} requests"


def fetch(workers: int = 4) -> None:
    """Resumable. A few districts at a time: each request takes ~11 s server-side,
    so one-at-a-time is ~2 h, and modest parallelism is polite to a public service."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    RAW.mkdir(parents=True, exist_ok=True)
    points = json.loads(POINTS.read_text(encoding="utf-8"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(_fetch_district, n, lat, lon) for n, (lat, lon) in points.items()]
        for job in as_completed(jobs):
            print(job.result(), flush=True)


def composites() -> pd.DataFrame:
    """One NDVI value per (district, composite date): mean of valid pixels in the box."""
    recs = []
    for path in sorted(RAW.glob("*__[0-9][0-9][0-9].json")):
        name = path.name.split("__")[0]
        for s in json.loads(path.read_text(encoding="utf-8")).get("subset", []):
            vals = np.asarray(s["data"], dtype=float)
            ok = vals > FILL
            if ok.sum() == 0:
                continue
            recs.append({"district": name, "date": s["calendar_date"],
                         "ndvi": float(vals[ok].mean() * SCALE),
                         "valid_pixels": int(ok.sum()), "pixels": int(vals.size)})
    return pd.DataFrame(recs).drop_duplicates(["district", "date"])


#: Days after a composite's 16-day window ends before it counts as available.
#: Conservative assumption (one composite period); not a measured release latency.
AVAILABILITY_DAYS = 16


def weekly() -> pd.DataFrame:
    """Latest already-available composite for each district and report week."""
    comp = composites()
    comp["period_start"] = pd.to_datetime(comp["date"])
    comp["available_from"] = comp["period_start"] + pd.Timedelta(days=16 + AVAILABILITY_DAYS)
    index = pd.read_csv(INDEX, parse_dates=["week_start"])
    rows = []
    for name, g in comp.groupby("district"):
        g = g.sort_values("available_from").reset_index(drop=True)
        avail = g["available_from"].to_numpy()
        for r in index.itertuples(index=False):
            k = int(np.searchsorted(avail, np.datetime64(r.week_start), side="right")) - 1
            if k < 0:
                raise SystemExit(f"{name}: no composite available before {r.week_start.date()}")
            rows.append({"row": r.row, "year": r.year, "week_no": r.week_no,
                         "week_start": r.week_start.date().isoformat(), "district": name,
                         "ndvi": float(g.loc[k, "ndvi"]),
                         "composite_start": g.loc[k, "period_start"].date().isoformat(),
                         "composite_available_from": g.loc[k, "available_from"].date().isoformat()})
    out = pd.DataFrame(rows).sort_values(["row", "district"])
    if (pd.to_datetime(out["composite_available_from"]) > pd.to_datetime(out["week_start"])).any():
        raise AssertionError("a week uses a composite not yet available")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "modis_ndvi_weekly_by_district.csv", index=False)
    comp.drop(columns=["period_start", "available_from"]).to_csv(
        OUT / "modis_ndvi_composites_by_district.csv", index=False)
    return out


def validate() -> list[tuple[int, float]]:
    """Which shift of MODIS NDVI best explains the processed array's ``minNdvi`` channel?

    ``docs/DATA.md`` says channel 10 is NDVI lagged 17 weeks, i.e. row k holds NDVI
    from 17 weeks earlier. ``docs/ARRAY_AUDIT.md`` dates the array's climate rows
    (row k = week of 2021-01-01 + 7(k - 398) days) and found the 12-week "lags"
    pointing the wrong way. This tests the NDVI channel the same way: correlate
    district-demeaned anomalies at every shift and report the best.

    Positive = row k holds NDVI from that many weeks **earlier** (the documented
    direction); negative = from that many weeks **later**.
    """
    comp = composites()
    comp["centre"] = pd.to_datetime(comp["date"]) + pd.Timedelta(days=8)
    names = sorted(json.loads((REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json")
                              .read_text(encoding="utf-8")))
    arr = np.load(REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy",
                  allow_pickle=True).astype(float)
    dates = pd.Timestamp("2021-01-01") + pd.to_timedelta((np.arange(arr.shape[0]) - 398) * 7, unit="D")
    mid = dates + pd.Timedelta(days=3, hours=12)
    keep = [n for n in names if n in set(comp.district)]
    x = arr[:, [names.index(n) for n in keep], 10]
    x = x - x.mean(0)
    series = {n: g.sort_values("centre") for n, g in comp.groupby("district")}
    scores = []
    for lag in range(-30, 31):
        t = (mid - pd.Timedelta(weeks=lag)).astype("int64").to_numpy()
        e = np.column_stack([np.interp(t, series[n]["centre"].astype("int64").to_numpy(),
                                       series[n]["ndvi"].to_numpy()) for n in keep])
        e = e - e.mean(0)
        scores.append((lag, float(np.corrcoef(x.ravel(), e.ravel())[0, 1])))
    return sorted(scores, key=lambda s: -s[1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "weekly", "validate"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.step == "fetch":
        fetch()
    elif args.step == "validate":
        top = validate()[:5]
        print("best shifts for array minNdvi vs MODIS NDVI (positive = weeks earlier, as documented):")
        for lag, r in top:
            print(f"  {lag:+3d} weeks  r = {r:.3f}")
    else:
        out = weekly()
        print(f"wrote {len(out)} rows, {out.district.nunique()} districts, "
              f"NDVI range {out.ndvi.min():.3f}-{out.ndvi.max():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
