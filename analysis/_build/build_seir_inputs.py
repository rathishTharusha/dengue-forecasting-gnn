"""Build the external inputs the SEIR-GNN needs and the processed array lacks.

Two tables, each regenerated from a named public source rather than typed in:

``data/external/district_population.csv``
    Mid-year population by district, 2014-2024, from the Department of Census
    and Statistics (Registrar General's Department). An SEIR model needs the
    denominator N for every district, and ``docs/DATA.md`` confirms the processed
    array does not carry it.

``data/external/calendar_index.csv``
    The calendar week behind every row of the processed array. The array has no
    date axis, and a compartmental model cannot run without one: susceptible
    depletion accumulates in real time, and the 2017 DENV-2 immunity reset has to
    land on a real week. Each row is matched to the Epidemiology Unit's dated
    weekly-report dump by its full 25-district case vector.

Run::

    python analysis/_build/build_seir_inputs.py population --pdf <DCS pdf>
    python analysis/_build/build_seir_inputs.py calendar \
        --raw "datasets/output_Dengue Fever.csv"

Sources are recorded in ``docs/SEIR_DATA_SOURCES.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "data" / "external"

POPULATION_URL = (
    "https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/"
    "Mid-year_population_by_district_and_sex_2024.pdf"
)

#: DCS spellings that differ from the adjacency list's district keys.
DCS_TO_GRAPH = {"Nuwara-eliya": "NuwaraEliya", "Monaragala": "Moneragala"}


def district_names() -> list[str]:
    """District order used by the processed array: ``sorted()`` of the JSON keys."""
    return sorted(json.loads(ADJ.read_text(encoding="utf-8")))


def build_population(pdf_path: Path) -> pd.DataFrame:
    """Parse the DCS mid-year table into long form, one row per district-year.

    The PDF splits 2014-2024 across three pages, each with a year header row, a
    Total/Male/Female row, then the national row and 25 districts. Values are in
    **thousands**. Years marked ``*`` are provisional, which is kept as a column
    rather than dropped, because 2018 onward is provisional and that covers most
    of the study period.
    """
    import pymupdf

    rows = []
    for page in pymupdf.open(pdf_path):
        for table in page.find_tables().tables:
            grid = table.extract()
            header = grid[0]
            years = [(i, str(c)) for i, c in enumerate(header) if c and str(c)[:4].isdigit()]
            for line in grid[2:]:
                name = (line[0] or "").strip()
                if not name:
                    continue
                for col, label in years:
                    total = line[col]
                    if total in (None, ""):
                        continue
                    rows.append({
                        "district": DCS_TO_GRAPH.get(name, name),
                        "year": int(label[:4]),
                        "population_thousands": int(str(total).replace(",", "")),
                        "provisional": label.endswith("*"),
                    })
    df = pd.DataFrame(rows).drop_duplicates(["district", "year"])

    expected = set(district_names())
    got = set(df["district"]) - {"Sri Lanka"}
    if got != expected:
        raise SystemExit(f"district mismatch: missing {expected - got}, unexpected {got - expected}")

    # The districts must sum to the national figure, to within rounding of 25
    # values reported in thousands. A parse that shifted a column would not.
    for year, grp in df.groupby("year"):
        national = int(grp.loc[grp.district == "Sri Lanka", "population_thousands"].iloc[0])
        summed = int(grp.loc[grp.district != "Sri Lanka", "population_thousands"].sum())
        if abs(national - summed) > 13:
            raise SystemExit(f"{year}: districts sum to {summed}k but national is {national}k")
    return df.sort_values(["year", "district"]).reset_index(drop=True)


#: Source labels that are not a prefix of their graph key. ``paha`` and the
#: truncated spellings come from the authors' own ``Configs/disease_config.json``;
#: ``Monaragala`` is the Epidemiology Unit's spelling of the graph's ``Moneragala``.
LABEL_ALIASES = {"paha": "Gampaha", "monaragala": "Moneragala", "nuwara": "NuwaraEliya"}

#: Labels deliberately excluded, as the authors' config excludes them. ``SRILANKA``
#: is the national total row. ``Kalmunai`` is a separate health division (RDHS)
#: that lies inside Ampara district; the benchmark's Ampara target excludes it, so
#: it is kept out here too for comparability. See docs/ARRAY_AUDIT.md.
EXCLUDED_LABELS = ("srilanka", "94srilanka", "kalmune", "kalmunei", "kalmunai")


def _canonical(label: str, names: list[str]) -> str | None:
    """Map the dump's truncated or misspelt district labels onto graph keys.

    An earlier prefix-only version silently failed on ``Monaragala`` and ``paha``,
    dropping Moneragala from every report and Gampaha from 30 of them.
    """
    key = str(label).lower().replace("-", "").replace(" ", "")
    if key in EXCLUDED_LABELS:
        return None
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for n in names:
        nk = n.lower().replace("-", "")
        if nk.startswith(key[:6]) or key.startswith(nk[:6]):
            return n
    return None


def build_calendar(raw_csv: Path) -> pd.DataFrame:
    """Match every array row to a dated Epidemiology Unit weekly report.

    Matching uses the whole 25-district case vector for the week, so a match is
    an exact reproduction of a published report rather than a guess from one
    number. ``abs_diff`` is the L1 distance to the best-matching report and
    ``runner_up_diff`` the distance to the next best; an exact match with a large
    runner-up gap is unambiguous.
    """
    names = district_names()
    cases = np.nan_to_num(np.load(NPY, allow_pickle=True))[..., 5]

    raw = pd.read_csv(raw_csv, encoding="utf-8", encoding_errors="replace")
    raw["start"] = pd.to_datetime(raw["TimeStampStart"], format="%d-%b-%Y %H:%M:%S",
                                  errors="coerce")
    raw["district"] = raw["Location Name"].map(lambda x: _canonical(x, names))
    raw["Cases"] = pd.to_numeric(raw["Cases"], errors="coerce")
    raw = raw.dropna(subset=["district", "start"])

    weekly = raw.groupby(["start", "district"])["Cases"].max().unstack().reindex(columns=names)
    weekly = weekly[weekly.notna().sum(axis=1) >= 20]
    source = raw.groupby("start")["Source File"].first()
    grid = weekly.to_numpy()

    out = []
    for week in range(cases.shape[0]):
        dist = np.nansum(np.abs(grid - cases[week]), axis=1)
        order = np.argsort(dist)
        date = weekly.index[order[0]]
        out.append({
            "array_week": week,
            "report_week_start": date.date().isoformat(),
            "abs_diff": float(dist[order[0]]),
            "runner_up_diff": float(dist[order[1]]),
            "national_cases": float(cases[week].sum()),
            "source_report": Path(str(source.get(date, ""))).name,
        })
    df = pd.DataFrame(out)
    step = pd.to_datetime(df["report_week_start"]).diff().dt.days
    df["days_since_previous_row"] = step

    # The report's own volume and number are more trustworthy than the dump's
    # parsed date, which is wrong for a handful of reports. The Weekly
    # Epidemiological Report numbers volumes by year: Vol 40 is 2013, Vol 50 is
    # 2023. That rule is checked against the parsed dates rather than assumed.
    parsed = df["source_report"].str.extract(r"(?i)vol[_ ]?(\d+)[_ ]?no[_ ]?(\d+)").astype(float)
    df["report_volume"] = parsed[0].astype("Int64")
    df["report_number"] = parsed[1].astype("Int64")
    df["report_year"] = df["report_volume"] + 1973
    dates = pd.to_datetime(df["report_week_start"])
    agrees = (df["report_year"] == dates.dt.year) | (
        (dates.dt.month == 12) & (df["report_year"] == dates.dt.year + 1)
    )
    if agrees.mean() < 0.99:
        raise SystemExit(f"volume->year rule holds for only {agrees.mean():.1%} of rows")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("population")
    p.add_argument("--pdf", required=True, type=Path,
                   help=f"Local copy of {POPULATION_URL}")
    c = sub.add_parser("calendar")
    c.add_argument("--raw", required=True, type=Path,
                   help="datasets/output_Dengue Fever.csv (Epidemiology Unit dump)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)

    if args.cmd == "population":
        df = build_population(args.pdf)
        path = OUT / "district_population.csv"
        df.to_csv(path, index=False)
        print(f"wrote {len(df)} rows -> {path}")
    else:
        df = build_calendar(args.raw)
        path = OUT / "calendar_index.csv"
        df.to_csv(path, index=False)
        exact = int((df["abs_diff"] == 0).sum())
        print(f"wrote {len(df)} rows -> {path} ({exact} exact matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
