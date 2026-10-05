"""District population, with every figure usable only from the date its document was published.

Rule (fixed in advance, ``docs/WEEK_INDEX_FIX.md``): for a forecast week starting on ``d``, take the
estimate with the latest reference date that is not after ``d`` among the documents whose
``available_from`` is not after ``d``. If no document is available yet (weeks before the first
verified publication date), use the earliest one and record the week as ``pre_availability``.

Documents and their publication dates. The date is the file's HTTP ``Last-Modified`` header, or its
PDF creation date where the header is absent. It is an upper bound for the true release date, so
it never lets a figure in early.

  census_2012        DCS "Census of Population and Housing 2012", district reports
                     (statistics.gov.lk .../cph2011/Pages/Activities/Reports/District/<name>.pdf,
                     Last-Modified 2015-03-10); reference date 2012-03-20 (census day)
  midyear_2014_2024  DCS "Mid-year population by district and sex, 2024" (estimates 2014 to 2024)
                     https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/
                     Mid-year_population_by_district_and_sex_2024.pdf
                     PDF created / Last-Modified 2024-09-19; reference dates 1 July of each year
  midyear_2025       DCS "Mid-year population estimates by district and sex, 2025*" (provisional)
                     .../Mid-year_population_by_district_and_sex_2025.pdf, Last-Modified 2025-11-03;
                     reference date 2025-07-01

Earlier editions of the mid-year table are no longer on the DCS site, so estimates for 2014 to 2023
can only be dated to the 2024 file. This is stricter than the earlier rule (the previous year's
estimate from the 2024 file for every week of a year), which let figures in up to ten years early.

    python analysis/_build/population_vintages.py          # write data/external/population_vintages.csv
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
EXT = REPO / "data" / "external"
OUT = EXT / "population_vintages.csv"
URL25 = "https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/Mid-year_population_by_district_and_sex_2025.pdf"
DCS_TO_GRAPH = {"Nuwara-eliya": "NuwaraEliya", "Monaragala": "Moneragala", "Baticaloa": "Batticaloa"}
UA = {"User-Agent": "Mozilla/5.0 Chrome/124"}


def district_names() -> list[str]:
    return sorted(json.loads((REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text("utf-8")))


def parse_2025() -> pd.Series:
    import pymupdf

    raw = urllib.request.urlopen(urllib.request.Request(URL25, headers=UA), timeout=90).read()
    text = "\n".join(p.get_text() for p in pymupdf.open(stream=raw, filetype="pdf"))
    tokens = [t.strip() for t in text.split("\n") if t.strip()]
    names, out = district_names(), {}
    for i, tok in enumerate(tokens):
        name = DCS_TO_GRAPH.get(tok, tok.replace("-", "").replace(" ", ""))
        if name in names and name not in out:
            out[name] = int(next(t for t in tokens[i + 1:] if re.fullmatch(r"[\d,]+", t)).replace(",", "")) * 1000
    assert sorted(out) == names, sorted(set(names) - set(out))
    total = int(next(t for t in tokens[tokens.index("Sri Lanka") + 1:] if re.fullmatch(r"[\d,]+", t)).replace(",", ""))
    assert abs(sum(out.values()) / 1000 - total) <= 25, (sum(out.values()) / 1000, total)   # thousands, rounding
    print("2025 file sha256", hashlib.sha256(raw).hexdigest()[:16])
    return pd.Series(out)


def build() -> pd.DataFrame:
    names = district_names()
    census = pd.read_csv(EXT / "census_2012_dcs_district_totals.csv").set_index("district")["population_2012"]
    mid = pd.read_csv(EXT / "district_population.csv")
    mid = mid[mid.district != "Sri Lanka"].pivot(index="year", columns="district", values="population_thousands") * 1000
    rows = []
    for d in names:
        rows.append(("census_2012", "2015-03-10", "2012-03-20", d, float(census[d])))
        for y in mid.index:
            rows.append(("midyear_2014_2024", "2024-09-19", f"{int(y)}-07-01", d, float(mid.loc[y, d])))
    p25 = parse_2025()
    for d in names:
        rows.append(("midyear_2025", "2025-11-03", "2025-07-01", d, float(p25[d])))
    return pd.DataFrame(rows, columns=["document", "available_from", "reference_date", "district", "population"])


def population_for_dates(week_starts: pd.Series) -> tuple[np.ndarray, pd.DataFrame]:
    """``(T, 25)`` persons and a per-week audit table (document, reference date, pre_availability)."""
    v = pd.read_csv(OUT, parse_dates=["available_from", "reference_date"])
    names = district_names()
    first = v["available_from"].min()
    pops, audit = [], []
    for d in pd.to_datetime(week_starts):
        avail = v[v["available_from"] <= d]
        pre = avail.empty
        if pre:
            avail = v[v["available_from"] == first]
        ref = avail[avail["reference_date"] <= d]["reference_date"].max() if (avail["reference_date"] <= d).any() \
            else avail["reference_date"].min()
        sel = avail[avail["reference_date"] == ref].drop_duplicates("district").set_index("district")
        pops.append(sel.loc[names, "population"].to_numpy(dtype=np.float64))
        audit.append({"week_start": d.date().isoformat(), "document": sel["document"].iloc[0],
                      "reference_date": ref.date().isoformat(), "pre_availability": bool(pre)})
    return np.stack(pops), pd.DataFrame(audit)


def main() -> int:
    df = build()
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
