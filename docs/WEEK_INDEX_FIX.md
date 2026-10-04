# Week-index fix (branch `fix/week-index`)

## What was wrong

`data/corrected/rebuilt_index.csv` has one row per consecutive report week (559 rows,
2013 week 26 to 2024 week 10). Its `week_start` was parsed from the source dump, and
`build_corrected_cases.report_dates` only checked that a parsed date's **year** matched the
report's volume year. Eight rows had the right year and the wrong month:

| row | report | old `week_start` | correct | shift (days) |
|---|---|---|---|---|
| 0 | 2013 no. 26 | 2013-05-15 | 2013-06-15 | 31 |
| 209 | 2017 no. 26 | 2017-05-17 | 2017-06-17 | 31 |
| 297 | 2019 no. 10 | 2019-01-26 | 2019-02-23 | 28 |
| 346 | 2020 no. 7 | 2020-01-01 | 2020-02-01 | 31 |
| 378 | 2020 no. 39 | 2020-09-11 | 2020-09-12 | 1 |
| 388 | 2020 no. 49 | 2020-12-21 | 2020-11-21 | -30 |
| 537 | 2023 no. 41 | 2023-08-31 | 2023-09-30 | 30 |
| 554 | 2024 no. 6 | 2023-12-27 | 2024-01-27 | 31 |

The correct index is the weekly grid `2013-06-15 + 7 * row` (both ends are Saturdays,
2013-06-15 and 2024-02-24, 558 weeks apart). It agrees with the other 551 rows exactly.
`2013-05-15 .. 2024-02-24` is not a whole number of weeks, which is how the error was found.

## Everything that reads `week_start`

| Input | Reads | Fix |
|---|---|---|
| ERA5 climate (`era5_weekly_by_district.csv`, `rebuilt_climate_era5.npy`) | the 7 days from `week_start` | re-fetched the daily series and re-aggregated; 1184 of 83850 cells changed, all on the 8 rows |
| MODIS NDVI weekly table | as-of composite before `week_start` | rebuilt from the committed composites (keyed by calendar date); 175 cells on 7 rows |
| COVID response weekly table | stringency and mobility over the week | rebuilt; stringency changed on rows 346 and 388, mobility on rows 378 and 388 |
| Seasonal features (`seirgnn2/core.py::seasonal_features`, Kaggle notebook) | day of year of `week_start[i]` | follows from the index |
| Kaggle notebook | its own copy of the date parsing | `full_paper/kaggle/src/20_data.py` now snaps to the grid and asserts it |
| Population | `year` column, not the date | unaffected |
| Cases | row order | unaffected |

ERA5 re-fetched from Open-Meteo reproduced the old values on the 551 unaffected rows, so the
source data did not drift.

## Old versus new climate

Largest change per variable (on the 8 rows): temperature mean 1.81 degC, min 2.10, max 2.83,
precipitation 167.1 mm per week, relative humidity 15.4 points, soil moisture 0.41 m3/m3.
Best single-variable pooled r2 against cases at causal lags 2 to 8: **0.02910 -> 0.02934**
(soil moisture, lag 5, both). Lag-1 r2 is 0.8893 (cases only; unchanged).
Details: `analysis/results/week_index_climate_compare.json`.

## Affected windows

`analysis/_build/week_index_impact.py` counts forecast windows whose covariate rows include a
mis-dated week, per split and protocol (`analysis/results/week_index_impact.json`). Summed over the
three origins of the three-origin protocol: season 9 train / 3 val / 5 test windows; climate
lags 2-4: 27 / 6 / 12; climate blocks 2-13: 127 / 18 / 50; blocks 2-25: 247 / 30 / 91; COVID
policy and mobility 9 / 2 / 4 (only rows 346, 378, 388 changed values).

## Results that must be re-run

See EXP-062 in `docs/EXPERIMENT_LOG.md`. Nine-origin arms of the frozen protocol that use cases
only (`seirgnn2/results/frozen9_*`) are not affected.

Older `seirgnn2/results/*.json` grids that used seasonal features or climate (for example `screen`, EXP-032 to EXP-049) were also trained on mis-dated rows by the same few windows. The paper does not use them, and they are not re-run.
