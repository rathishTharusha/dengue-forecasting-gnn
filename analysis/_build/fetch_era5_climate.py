"""Fetch a fully dated climate record per district from ERA5, and aggregate it weekly.

Why this exists
---------------
The processed array's climate channels carry no dates, and ``docs/ARRAY_AUDIT.md``
shows they sit on a different timeline from its cases. The array also has no
GLDAS data for Jaffna and stops in March 2022. A climate record that is dated at
the source fixes all three, and it doubles as an independent check on the
re-dating of the array's own climate.

Source
------
ERA5 reanalysis (Copernicus Climate Change Service), through the Open-Meteo
Historical Weather API, which needs no account: https://open-meteo.com/en/docs/historical-weather-api
ERA5 is also what Liu et al. (2025) used for their Sri Lanka SEIR-LSTM.
Attribution: Hersbach H. et al. (2023) ERA5 hourly data, C3S CDS,
doi:10.24381/cds.adbb2d47; data served by Open-Meteo.com under CC BY 4.0.

One interior point per district, from the same GADM 4.1 boundaries the original
pipeline used (``gadm41_LKA_1.json`` in the authors' repository). A single point
is a simplification of an area average and is recorded as such.

Run::

    python analysis/_build/fetch_era5_climate.py fetch
    python analysis/_build/fetch_era5_climate.py weekly
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
GADM = REPO / "data" / "raw" / "disease_modeling_MLOS2" / "Data" / "Countries" / "gadm41_LKA_1.json"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
RAW = REPO / "data" / "raw" / "era5_openmeteo"
OUT = REPO / "data" / "external"
INDEX = REPO / "data" / "corrected" / "rebuilt_index.csv"

API = "https://archive-api.open-meteo.com/v1/archive"
DAILY = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
         "precipitation_sum", "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean"]
START, END = "2013-01-01", "2024-03-31"


def district_points() -> dict[str, tuple[float, float]]:
    """An interior point per district: the centroid if it lies inside, else a representative point."""
    from shapely.geometry import shape

    gadm = json.loads(GADM.read_text(encoding="utf-8"))
    names = set(json.loads(ADJ.read_text(encoding="utf-8")))
    pts = {}
    for ft in gadm["features"]:
        name = ft["properties"]["NAME_1"]
        if name not in names:
            continue
        geom = shape(ft["geometry"])
        p = geom.centroid if geom.contains(geom.centroid) else geom.representative_point()
        pts[name] = (round(p.y, 4), round(p.x, 4))
    missing = names - set(pts)
    if missing:
        raise SystemExit(f"no GADM polygon for {sorted(missing)}")
    return dict(sorted(pts.items()))


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    pts = district_points()
    (RAW / "points.json").write_text(json.dumps(pts, indent=1), encoding="utf-8")
    for name, (lat, lon) in pts.items():
        path = RAW / f"{name}.json"
        if path.exists():
            continue
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": START,
                                    "end_date": END, "daily": ",".join(DAILY), "models": "era5",
                                    "timezone": "Asia/Colombo"})
        # The free tier weights a request by its length: an 11-year, 6-variable
        # query counts as ~175 calls against a 600-per-minute limit. So pace
        # requests, and back off and retry when the API answers 429.
        for attempt in range(6):
            try:
                with urllib.request.urlopen(f"{API}?{q}", timeout=120) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 5:
                    raise
                print(f"  {name}: rate limited, waiting 75s (attempt {attempt + 1})", flush=True)
                time.sleep(75)
        if "daily" not in payload:
            raise SystemExit(f"{name}: unexpected response {str(payload)[:200]}")
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"  {name:13s} ({lat}, {lon}) -> {len(payload['daily']['time'])} days", flush=True)
        time.sleep(25)


def weekly() -> pd.DataFrame:
    """Mean of each variable over the 7 days starting at each report week.

    Weeks follow ``data/corrected/rebuilt_index.csv``, so the climate lines up
    with the rebuilt case series row for row. Precipitation is a weekly total.
    """
    index = pd.read_csv(INDEX, parse_dates=["week_start"])
    frames = []
    for path in sorted(RAW.glob("*.json")):
        if path.name == "points.json":
            continue
        d = json.loads(path.read_text(encoding="utf-8"))["daily"]
        df = pd.DataFrame(d)
        df["time"] = pd.to_datetime(df["time"])
        daily = df.set_index("time")
        for _, row in index.iterrows():
            span = daily.loc[row.week_start: row.week_start + pd.Timedelta(days=6)]
            rec = {"row": int(row.row), "year": int(row.year), "week_no": int(row.week_no),
                   "week_start": row.week_start.date().isoformat(), "district": path.stem,
                   "days": int(len(span))}
            for v in DAILY:
                rec[v] = float(span[v].sum() if v == "precipitation_sum" else span[v].mean())
            frames.append(rec)
    out = pd.DataFrame(frames).sort_values(["row", "district"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "era5_weekly_by_district.csv", index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fetch", "weekly"])
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.step == "fetch":
        fetch()
    else:
        out = weekly()
        print(f"wrote {len(out)} rows: {out.district.nunique()} districts x {out.row.nunique()} weeks; "
              f"incomplete weeks: {int((out.days < 7).sum())}; any NaN: {bool(out.isna().any().any())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
