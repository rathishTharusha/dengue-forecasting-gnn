# Data in this package

Everything the reproduction notebook (`../reproduce_full_paper.ipynb`) reads, plus the
source tables it was built from. The authoritative provenance, with download steps, is
`docs/DATA_PROVENANCE.md` and `docs/ARRAY_AUDIT.md` in the repository.

> **Corrected on 2026-09-25.** An earlier version of this file said that missing weeks
> were linearly interpolated, that NDVI was spline-interpolated, that ERA5 was averaged
> over district polygons with variables `t2m`/`tp`/`d2m`/`sp`/`u10`/`v10`, and that the
> serosurvey was Tissera et al. (2020). None of that matches the data. The descriptions
> below do.

## Files the notebook uses

| file | what it is |
|---|---|
| `rebuilt_cases_weekly.csv` | **The corrected case series used by every result.** 559 weeks (2013-W26 → 2024-W10) × 25 districts, rebuilt from all 552 weekly epidemiological reports, keyed by report volume and number. The 7 weeks with no published report are **left empty (NaN)** and flagged in the `missing` column; the harness drops any forecast window that touches one. Written by `scripts/build_full_paper_notebook.py` from `analysis/lib/corrected_data.py`. |
| `rebuilt_climate_weekly.csv` | The six ERA5 channels aligned row for row with the case series (`week` = row index): temperature mean/min/max, precipitation sum, relative humidity, soil moisture 0–7 cm. **No lags are baked in**; the harness applies a 2-week lag when it reads them. |

## Source tables

| file | what it is |
|---|---|
| `district_cases_weekly.csv` | The same 559-week series, but with the 7 missing weeks **filled in**. It is kept for reference only and is **not** used by any result: the protocol never interpolates. |
| `dengue_cases_raw.csv` | The benchmark authors' parsed source dump (one row per district-week, with the dates the parser assigned). Our rebuild keys weeks by report volume and number instead, because those parsed dates are not reliable (`docs/ARRAY_AUDIT.md`). |
| `era5_weekly_by_district.csv` | ERA5 via Open-Meteo, one interior point per district, aggregated to the rebuilt weeks. Jaffna included. |
| `modis_ndvi_weekly_by_district.csv` | MODIS MOD13Q1 v061 NDVI (250 m, 16-day), ±5 km box around each district's interior point. Taken **as of** each week: a week holds the latest composite already released before it began (16-day window + 16-day release delay). Never interpolated. |
| `district_census_2012.csv` | 2012 Census district metadata. The harness uses the previous year's official population for each week (DCS mid-year estimates from 2015). |
| `seroprevalence_nine_districts.csv` | IgG seroprevalence, ages 10–20, sampled 2022-09 to 2023-03: Jeewandara et al., medRxiv 10.1101/2023.04.23.23288986 (J Med Virol 2024). **Validation only**: it is hindsight knowledge for most of the study period. |
| `seir_parameters.json` | Transcribed SEIR–SEI parameters with a source key per value. The forecasting harness itself uses incubation 0.1/day, recovery 1/7 per day, reporting rate 1/11 and an initial susceptible fraction of 1 − 0.682 (a 2013–14 suburban Colombo serosurvey). |

## What the models may see

Cases up to week *t−1*, climate up to *t−2*, NDVI and population as of *t*, enforced by
`analysis/lib/corrected_data.py::LAGS` and by perturbation tests
(`tests/test_no_future_leakage.py`). The district graph is the benchmark's adjacency list
with self-loops, row-normalised.

## Why not the benchmark array

`sri_lanka_2013-2022_shifted.npy`, used by prior work, has rows out of date order (2023 in
every training split), case and climate columns from different weeks, "lagged" climate
channels shifted into the future, and a spreadsheet error at week 395. Section 2 of the
notebook shows the measurements.
