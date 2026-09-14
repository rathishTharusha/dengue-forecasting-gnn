# Audit — the time axis of `sri_lanka_2013-2022_shifted.npy`

**Status:** diagnosed, and corrected case series built in `data/corrected/`. The
original array is unchanged. Re-running the benchmark on the corrected data is
in progress (see the last section).

`docs/DATA.md` describes the array as *"459 consecutive weeks spanning
2013–2022"*. The array itself has no date axis, so that description was never
checked. This audit checks it.

Reproduce with:

```bash
python analysis/_build/build_seir_inputs.py calendar --raw "datasets/output_Dengue Fever.csv"
python analysis/_build/audit_array_order.py --raw "datasets/output_Dengue Fever.csv"
```

Outputs: `analysis/results/array_audit/audit.json` and five figures.
`data/external/calendar_index.csv` is the table behind every chart.

---

## How rows were dated

Each row's 25-district case vector was matched against the Epidemiology Unit's
weekly-report dump. **452 of 459 rows reproduce a named report exactly.** Each
report's volume number gives its year: Volume N is year 1973 + N, which holds for
457 of 459 rows. The two exceptions are dates the dump itself parsed wrongly. The
authors' public `sri_lanka_2013-2022_vertical.csv` confirms array row 0 is their
row 315, with zero difference.

Everything below was then re-tested with checks that **don't reuse the matching**.

---

## Findings

### 1. The rows are not in date order — confirmed
![](../analysis/results/array_audit/1_row_to_date.png)

- **Rows 0–47 are 2023** (report Volume 50). Row 49 onward runs from May 2013 to
  March 2022.
- Likely cause: in 2023 the report files were renamed to `en_…Vol_50…`, which
  sorts before the older `vol_40…` names.
- **Rows 444 and 445 are identical**: the same week stored twice.
- **7 rows (48, 49, 50, 54, 297, 301, 434) match no report exactly.** Their dates
  are nearest-report guesses and shouldn't be trusted.

### 2. Cases and climate follow different timelines — confirmed, and the most serious finding
![](../analysis/results/array_audit/2_two_timelines.png)
![](../analysis/results/array_audit/3_climate_alignment.png)

The case column was reordered. **The climate columns were not.**

- An annual cycle fits temperature with **R² 0.82 in row order**, but only **0.29
  against the case dates**. Humidity, soil moisture, precipitation and NDVI all
  show the same pattern.
- Temperature runs smoothly through row 48 with no break. The temperature peak
  comes every **53 rows on average**, about one per year.
- **Two independent measurements of the drift agree to within 2.8 weeks on
  average.** One uses only the report dates; the other uses only where the
  temperature peaks fall. Both say the cases run steadily further ahead of the
  weather: about 31 weeks by row 107, and about 48–55 weeks from 2019 onward.

**What that means:** in most rows the model sees cases from one week next to
weather from a different week. By 2019–2021 that's about a year earlier. This may
explain why climate inputs made every model worse (EXP-024, the covariate sweep),
**but that hasn't been tested.**

**What this can't tell us:** the absolute offset. The drift is measured relative to
row 54, and the misalignment there is unknown. The 2023 block's climate dates are
also unknown.

### 3. Weeks are missing, and some were dropped during processing — confirmed
![](../analysis/results/array_audit/4_week_coverage.png)

- **49 weekly reports exist in the source dump but aren't in the array.** These
  were lost in processing, not missing from the source.
- 2014 is about half missing. **2022 stops in March** even though the reports from
  April onward are in the source. Late 2023 is also absent.
- Other gaps have no report in the source at all.

### 4. Every training split contains 2023 — confirmed
![](../analysis/results/array_audit/5_protocol_folds.png)

Rows 3–47, the 2023 reports, sit inside the training set of all three folds of the
frozen protocol. The test periods are 2018-03 → 2019-07, 2019-07 → 2020-10 and
2020-11 → 2022-03, so **every model trained on data from after its test period.**

### 5. The week-395 spike is dated — confirmed
It's the report for the week starting **2020-12-26** (Volume 48, no. 2), at the
turn of the year. That fits the backlog explanation. The date is certain; the
cause is still an interpretation.

---

## Checks that were weak, and corrections

- **Official annual totals.** Array sums for 2017, 2018 and 2019 come to 85–90% of
  the published national totals. The weekly reports are provisional, so some
  shortfall is expected. The ratios between years favour the calendar index
  (log error 0.026) over reading the array as in order (0.063), but that's
  **supporting evidence, not proof**.
- **The splice isn't visible in the cases alone.** The case jump at row 47→48 sits
  at the 98.9th percentile, but a jump inside 2023 is nearly as large. The strong
  evidence for the splice is the report matching, not the case numbers.
- **Correction to `SEIR_DATA_SOURCES.md`.** It said ~135 calendar weeks were
  missing and listed 7 "duplicates". The first figure mixed together out-of-order
  steps, source gaps and dropped reports; the breakdown above replaces it. The 7
  were unmatched rows, and the one true duplicate is rows 444/445.

---

## Not yet measured

- **How much any of this changes existing results.** Persistence is measured
  (below). Model results are running on Kaggle.
- **The absolute weather-to-case offset.** Pinning it down needs the satellite
  files' own timestamps, which aren't in the authors' repository.

## Decisions for the team

1. **The SEIR-GNN needs real, ordered time.** Build its input from the dated
   reports, not from the array order.
2. **The existing benchmark.** Re-running it on a corrected array invalidates
   every table in the paper. The alternative is to report the audit as a
   limitation. `CLAUDE.md` says re-deriving the array is its own task, so this
   needs a team decision.
3. **Climate inputs.** They can't be judged until they're aligned with the cases.
   Every "climate doesn't help" result so far was measured on misaligned rows.

---

## Correction (done) and re-run (in progress)

`analysis/_build/build_corrected_cases.py` builds two case-only series in
`data/corrected/`. The original array is left untouched.

| Dataset | Weeks | Span | What it isolates |
|---|---|---|---|
| `reordered` | 451 | the benchmark's own rows, in true order, duplicate removed | the effect of **ordering** alone |
| `rebuilt` | 559 | every source report, 2013-W26 → 2024-W10, 7 interpolated weeks flagged | the corrected dataset going forward |

Time is keyed by report volume and number, not the dump's parsed date. **All 451
`reordered` weeks match `rebuilt` exactly**, so the array's values are the
published values; only their order and coverage were wrong.

Found while building it:

- **A name-mapping bug in the first audit pass.** `Monaragala` and the truncated
  label `paha` weren't mapped, which dropped Moneragala from every report and
  Gampaha from 30. With the fix, the 452 exact matches still hold on all 25
  districts.
- **Kalmunai.** The source reports the Kalmunai health division separately. It
  lies inside Ampara, and the authors' config excludes it, so the benchmark's
  Ampara target is Ampara RDHS only. That definition is kept for comparability.
- **The national-total check.** Districts plus Kalmunai equal the Epidemiology
  Unit's own national total in **432 of 552 reports**. In the rest, the parsed
  national figure is typically a tenth of the district sum, which looks like a
  parsing loss in that one field.
- **The week-395 report looks misparsed, not only backlogged.** Every field in it
  is anomalous: districts sum to 7,165 (neighbouring weeks are ~400), Kalmunai
  shows 752 (normally ~30), and the national total shows 35. This is likely but
  not confirmed; checking would need the original PDF (Vol 48 no. 02).

### The persistence floor moves

Same code, frozen protocol, artifact located by report, windows with interpolated
targets excluded:

| Dataset | RMSE (all windows) | RMSE (artifact-free) |
|---|---:|---:|
| `original` | 44.795 | **29.521** (reproduces the paper) |
| `reordered` | 48.265 | **31.089** |
| `rebuilt` | 49.870 | **36.053** |

The test periods themselves change. The artifact moves from the 0.85 fold to the
0.70 fold, and `rebuilt`'s last fold covers 2022–2024, including the large 2023
epidemic. **Model results on the corrected data are pending** (Kaggle kernels
`corrected-benchmark-*`, compared with `compare_corrected.py`).

### Climate is not corrected yet
Both corrected datasets are **cases only**. The climate channels' calendar dates
still aren't established. The authors' repository contains one week of GLDAS,
GPM and NDVI files (1–8 January 2021), which could anchor the climate timeline by
exact match. That hasn't been done.
