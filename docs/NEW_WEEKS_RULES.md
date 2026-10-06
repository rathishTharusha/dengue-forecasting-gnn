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

## Amendment 1 (2026-10-06): parser defect found before any count was read

The first parse (at commit `0972363`) marked all 127 reports missing: 123 with "column count 28",
4 with "dengue table not found". PDF table extraction adds a trailing empty column to otherwise
valid Table 1 rows, so the 27-column check rejected every one. Only the header labels and
yes/no table-structure flags were inspected to diagnose this. **No epidemiological value was
inspected before this amendment**, and no model has been run on these weeks.

The parser (`table_columns` in `analysis/_build/parse_new_weeks.py`) now:
- removes **trailing** blank cells only; no internal cell is dropped and no column is shifted;
- requires exactly 27 header labels after that, validated **before** any count is read;
- accepts a label only if it equals its expected column in order, or is one of the two listed
  spellings (`Nuwara Eliya` -> `NuwaraEliya`, `Monaragala` -> `Moneragala`);
- makes the week missing (R5) if a data row has any non-blank cell beyond the 27 columns.

Rules R1-R6 are unchanged. The 4 reports whose Table 1 is not found (Vol. 52 No. 2; Vol. 53
No. 15, 19, 28) stay missing under R5; no special rule is written for them. Tests:
`tests/test_parse_new_weeks.py`. The new weeks are re-parsed in full from the PDFs after this
amendment is committed.

## Amendment 2 (2026-10-06): data-construction defects exposed by run 1, after results were seen

**This amendment was made after the first prospective run (EXP-063 run 1, commit `708f290`) had
been scored.** Run 1 showed every arm at RMSE ~284 and the cause was traced to two defects in
how the weekly series is built from the reports, not to any model. The corrections below follow
from what the reports mean and from data integrity; none depends on which model scores better.
Run 1 is invalid and is reported only as a protocol deviation. The run made after this amendment
is an amended prospective evaluation, not an untouched one: run 1 had already shown that the
horizon-1 errors of all arms were close to persistence.

1. **R3 keys on the printed epidemiological week, not the report number.** In these volumes
   report No. N carries printed week N-1, so report No. 1 is week 52 of the previous year and
   its row B is the whole year (Vol. 53 No. 1 entered as 50,052 cases). Now: row B is the weekly
   count only in the report whose **printed week is 1**. Otherwise the cumulative difference is
   row B minus row B of the **immediately preceding report**, and only if that report's printed
   week is exactly one less (this crosses volume boundaries; nothing is bridged over a missing or
   unreadable report). Without such a predecessor there is no replacement.
2. **R7, repeated tables.** If row A of the 25 districts equals the immediately preceding
   report's row A exactly, row A is treated as failed (Vol. 52 No. 8 and Vol. 53 No. 27 reprint
   the previous week). The cumulative difference is used if it passes the normal checks,
   otherwise the week is missing.
3. **R3 replacement validity.** A cumulative difference is used only if it passes R1 and R2 and
   has no negative entry.
4. **Cross-week QC, logged (R6) and counted in `qc_report.json`, never used to drop a week on
   its own:** a cumulative row that decreases between consecutive printed weeks of one year;
   row B differing from row A in a printed week 1; a repeated row A.

No rule on the size of a weekly count is added: large weeks are kept when the report is
internally consistent. R1, R2, R4, R5, R6 and amendment 1 are unchanged. Tests:
`tests/test_parse_new_weeks.py`. All 127 reports are re-parsed from the PDFs.

## What is not allowed
No change to a rule after the first parse. If a rule turns out to be wrong, the change is logged in
`docs/EXPERIMENT_LOG.md` with the reason and the new weeks are re-parsed in full.
