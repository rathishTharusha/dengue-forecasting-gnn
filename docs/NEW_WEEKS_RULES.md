# Rules for the new weeks (2024 W11 to 2026 W32), fixed before the data were parsed

Written and committed before `analysis/_build/parse_new_weeks.py` was run on any report. No model
has been run on these weeks. A header peek at five reports (printed week dates only, no case
counts) was done to design rule R4; it is logged here.

## Source
Weekly Epidemiological Reports, Epidemiology Unit, Ministry of Health, Sri Lanka, listed at
https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report. Expected reports:
Vol. 51 No. 11 to No. 52, Vol. 52 No. 1 to No. 52, Vol. 53 No. 1 to No. 33 (127 reports). File names,
sizes and SHA-256 are in `data/external/wer_new_manifest.csv`; the PDFs are in `data/raw/wer_new`
(git-ignored).

## Parsing
Table 1 ("Distribution of Notified Diseases reported by Medical Officers of Health"), the "Dengue
Fever" block, row **B** (cumulative this year) and row **A** (this week), 25 districts plus
Kalmunai plus the national total, the same layout and code path as `extract_week1_correction`.
The week's date range and printed week number are read from the table title.

## Automatic quality checks and corrections (all applied in code)
- **R1, district sum.** Row A is accepted only if the 25 districts plus Kalmunai equal the printed
  national total.
- **R2, formula pattern.** Row A is flagged if at least 8 of its 24 inner cells equal the sum of the
  two cells before them (within 1). The known formula error has 15; no other report of the 552 earlier
  ones has more than 3.
- **R3, replacement.** If R1 or R2 fails, the weekly vector is replaced by the cumulative difference
  (row B minus row B of the previous report in the same volume; row B itself in week 1 of a volume),
  provided that vector passes R1 and R2. Otherwise the week is **missing**. Nothing is interpolated or
  filled. A row A that passes R1 and R2 is always used, and a mismatch with the cumulative
  difference is logged as a warning.
- **R4, dates.** The week start is the printed week's end date minus 6 days. Consecutive printed starts
  must be 5 to 9 days apart; anything else is logged. (The header peek found Saturday starts in 2024
  and 2025 and a Monday start in the last report, so the new weeks are not on the old 7-day grid;
  covariates are joined by the printed dates.)
- **R5, missing reports.** An expected report with no file, an unreadable table, or two files that
  disagree is a missing week. Windows touching a missing week are dropped, as for the existing series.
- **R6, log.** Every correction, warning and missing week is written to
  `data/new_weeks/corrections.json`; counts are in `data/new_weeks/qc_report.json`.

## Covariates for the new weeks
ERA5: the same Open-Meteo call and 7-day aggregation from the printed week start. Population: by
publication date (`analysis/_build/population_vintages.py`). NDVI is not used by any arm in the
preregistered list and is not built for the new weeks.

## What is not allowed
No change to a rule after the first parse. If a rule turns out to be wrong, the change is logged in
`docs/EXPERIMENT_LOG.md` with the reason and the new weeks are re-parsed in full.
