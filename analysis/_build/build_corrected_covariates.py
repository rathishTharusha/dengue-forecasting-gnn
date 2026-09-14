"""Covariates for the rebuilt case series: dated climate, population and census.

Everything here is aligned **row for row** with ``data/corrected/rebuilt_index.csv``,
so ``rebuilt_cases.npy[t]``, ``rebuilt_climate_era5.npy[t]`` and
``rebuilt_population.npy[t]`` all describe the same week. That alignment is the
whole point: ``docs/ARRAY_AUDIT.md`` found the processed array's climate and cases
on different timelines.

Outputs
-------
``data/corrected/rebuilt_climate_era5.npy``   (weeks, 25, 6), channel names in the JSON sidecar
``data/corrected/rebuilt_population.npy``     (weeks, 25) persons, for the SEIR denominator
``data/external/district_census_2012.csv``    2012 census totals, over-60 counts, 2013 projection

Sources
-------
* Climate: ERA5 via Open-Meteo, one interior point per district
  (``fetch_era5_climate.py``). Jaffna is included; the original GLDAS pipeline had
  no grid cell inside it.
* Population 2014-2024: Department of Census and Statistics mid-year estimates
  (``data/external/district_population.csv``, 2018 onward provisional).
* Population 2013 and the census fields: 2012 Census of Population and Housing
  at GN-division level, with WFP/OCHA projections from the census district growth
  rates, published on HDX: https://data.humdata.org/dataset/sri-lanka-census-of-population-and-housing-2012
  (source field ``DCS LKA``).

Run::

    python analysis/_build/build_corrected_covariates.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
CORRECTED = REPO / "data" / "corrected"
EXTERNAL = REPO / "data" / "external"
HDX = REPO / "data" / "raw" / "hdx" / "lka_pop_censusproj2022_wfpocha.xlsx"

CLIMATE_VARS = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
                "precipitation_sum", "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean"]
HDX_ALIAS = {"Nuwara Eliya": "NuwaraEliya", "Monaragala": "Moneragala", "Mulaitivu": "Mullaitivu"}


def names() -> list[str]:
    return sorted(json.loads(ADJ.read_text(encoding="utf-8")))


def census() -> pd.DataFrame:
    df = pd.read_excel(HDX)
    df["district"] = df["DISTRICT_N"].astype(str).str.strip().replace(HDX_ALIAS)
    out = df.groupby("district").agg(
        census_2012=("TOT_POP", "sum"), over60_2012=("OVER60", "sum"),
        projection_2013=("PPROJ_13", "sum"), gn_divisions=("GND_C", "count")).reindex(names())
    if out.isna().any().any():
        raise SystemExit(f"census districts missing: {out.index[out.isna().any(axis=1)].tolist()}")
    out["over60_share_2012"] = out.over60_2012 / out.census_2012
    out["source"] = "DCS Census 2012 via HDX (WFP/OCHA); projection_2013 from census growth rates"
    return out.reset_index()


def climate(index: pd.DataFrame) -> np.ndarray:
    era = pd.read_csv(EXTERNAL / "era5_weekly_by_district.csv")
    cube = np.full((len(index), 25, len(CLIMATE_VARS)), np.nan, dtype=np.float32)
    col = {n: i for i, n in enumerate(names())}
    for rec in era.itertuples(index=False):
        cube[rec.row, col[rec.district]] = [getattr(rec, v) for v in CLIMATE_VARS]
    if np.isnan(cube).any():
        raise SystemExit("ERA5 weekly table does not cover every rebuilt week and district")
    return cube


def population(index: pd.DataFrame, cen: pd.DataFrame) -> np.ndarray:
    """Persons per district per week: the mid-year estimate for the report's year."""
    dcs = pd.read_csv(EXTERNAL / "district_population.csv")
    dcs = dcs[dcs.district != "Sri Lanka"].pivot(index="year", columns="district",
                                                 values="population_thousands") * 1000
    dcs.loc[2013] = cen.set_index("district")["projection_2013"]
    dcs = dcs.sort_index()[names()]
    years = index["year"].clip(upper=int(dcs.index.max()))
    return dcs.loc[years].to_numpy(dtype=np.float64)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    index = pd.read_csv(CORRECTED / "rebuilt_index.csv")
    cen = census()
    cen.to_csv(EXTERNAL / "district_census_2012.csv", index=False)

    cube = climate(index)
    np.save(CORRECTED / "rebuilt_climate_era5.npy", cube)
    (CORRECTED / "rebuilt_climate_era5.json").write_text(json.dumps({
        "shape": list(cube.shape), "channels": CLIMATE_VARS,
        "units": ["degC", "degC", "degC", "mm/week", "%", "m3/m3"],
        "aggregation": "7-day mean from week_start (precipitation: 7-day total)",
        "lag": "none -- each row is the same calendar week as the cases; apply lags in the model",
        "source": "ERA5 (Hersbach et al. 2023, doi:10.24381/cds.adbb2d47) via Open-Meteo, CC BY 4.0",
    }, indent=2), encoding="utf-8")

    pop = population(index, cen)
    np.save(CORRECTED / "rebuilt_population.npy", pop)

    print(f"census: {len(cen)} districts, national 2012 total {int(cen.census_2012.sum()):,}")
    print(f"climate: {cube.shape}, temperature {cube[..., 0].min():.1f}-{cube[..., 0].max():.1f} degC, "
          f"Jaffna mean {cube[:, names().index('Jaffna'), 0].mean():.2f} degC")
    print(f"population: {pop.shape}, national {pop[0].sum():,.0f} (2013) -> {pop[-1].sum():,.0f} "
          f"({int(index.year.iloc[-1])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
