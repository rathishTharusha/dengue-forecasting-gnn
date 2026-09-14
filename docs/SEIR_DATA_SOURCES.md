# SEIR-GNN — external data and where it came from

Phase 1 of the SEIR-GNN plan: the inputs a compartmental model needs that the
processed array does not carry. Every number here traces to a named public
source. Nothing is estimated or back-filled without saying so.

| File | What | Built by |
|---|---|---|
| `data/external/district_population.csv` | Mid-year population, 25 districts, 2014–2024 | `build_seir_inputs.py population` |
| `data/external/calendar_index.csv` | Calendar week behind every array row | `build_seir_inputs.py calendar` |
| `data/external/seir_parameters.json` | Stage durations, seroprevalence, ascertainment, serotypes | transcribed, with a source key per value |

---

## 1. District populations

**Source:** Department of Census and Statistics, Sri Lanka —
[Mid-year Population Estimates by District & Sex, 2014–2024](https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/Mid-year_population_by_district_and_sex_2024.pdf).
Compiled from the Registrar General's Department, based on the Census of
Population and Housing 2012.

- Units are **thousands**.
- **2018–2024 are marked provisional** in the source. That is most of the study
  period, so the flag is kept as a column.
- Parsed from the PDF's table structure, not typed in. The build checks that the
  25 districts sum to the national figure every year, within rounding.
- Two spellings are mapped onto the graph's keys: `Nuwara-eliya` →
  `NuwaraEliya` and `Monaragala` → `Moneragala`.

**Gap: 2013 is not in this table.** Populations grow about 1% a year, so reusing
2014 for 2013 is a small error, but it should be stated wherever it's used. The
2012 census district totals would close the gap. They exist on
[statistics.gov.lk](https://www.statistics.gov.lk/) and
[HDX](https://data.humdata.org/dataset/sri-lanka-census-of-population-and-housing-2012),
but weren't retrieved here.

---

## 2. Calendar index — and a problem with the array

The processed array has no date axis. An SEIR model can't run without one,
because susceptible depletion accumulates in real time and the 2017 DENV-2
immunity reset has to land on a real week.

**Method.** Each array row is matched to the Epidemiology Unit's dated
weekly-report dump (`datasets/output_Dengue Fever.csv`) by its full 25-district
case vector. The authors' own
[`sri_lanka_2013-2022_vertical.csv`](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2)
confirms array row 0 is their row 315, with zero difference.

**Result: 452 of 459 rows match a published weekly report exactly.** The report's
volume number gives its year (**Volume N = 1973 + N**, which holds for 457 of 459
rows).

### The array is not a chronological 2013–2022 series

| Array rows | Report volume | Calendar |
|---|---|---|
| **0–47** | **Vol 50** | **2023** |
| 48–50 | mixed, inexact | splice point |
| 49–76 | Vol 40 | 2013 (28 weeks) |
| 77–99 | Vol 41 | 2014 (**23 weeks**) |
| 100–145 | Vol 42 | 2015 |
| 146–192 | Vol 43 | 2016 |
| 193–243 | Vol 44 | 2017 |
| 244–291 | Vol 45 | 2018 |
| 292–341 | Vol 46 | 2019 |
| 342–393 | Vol 47 | 2020 |
| 394–445 | Vol 48 | 2021 |
| 446–458 | Vol 49 | 2022 (**13 weeks**) |

This is verified from the source filenames, not inferred. Rows 0–47 come from
reports named `en_<hash>_Vol_50_no_XX-english.pdf`, and row 51 onward from
`vol_40_no_26_english.pdf` and similar. **The likely cause is a lexical sort:**
the 2023 reports switched to an `en_…` filename prefix, which sorts before
`vol_…`. That would put 2023 at the front of the authors' disease table.

Inside the 2013→2022 stretch, **49 weekly reports that exist in the source never
reached the array**, and some weeks have no report in the source at all. 2014 is
about half absent, and 2022 stops in March. Rows 444 and 445 are the same week
stored twice. Seven rows (48, 49, 50, 54, 297, 301, 434) match no report
exactly, so their dates are unreliable.

> **Correction.** An earlier version of this section said "~135 calendar weeks
> missing" and listed seven duplicates. The first figure mixed together
> out-of-order steps, gaps in the source and dropped reports. The seven were
> unmatched rows, not duplicates. `docs/ARRAY_AUDIT.md` has the re-checked
> breakdown, including a finding not listed here: **the climate columns follow
> a different timeline from the cases.**

**What it means:**

- **For the SEIR:** a compartmental model needs real, ordered time. It can't run
  on this array as ordered. The calendar index gives the true order.
- **For every existing result:** the training split of every rolling-origin fold
  includes the 2023 block, which is later than every test period. That's
  temporal leakage. Separately, windows that cross a gap treat a 2–5 week jump
  as "last week", which breaks the lag-1 assumption persistence relies on. The
  size of either effect **has not been measured.**
- **Week 395 is dated.** It's the report for the week starting **2020-12-26**
  (Vol 48, no. 2) — the year boundary. That fits the "administrative backlog"
  reading of the spike.
- **January 2017 is array row 193.**

**Deciding what to do is a project decision, not a data-cleaning step.**
`CLAUDE.md` is explicit that re-deriving the array invalidates the whole ablation
table.

---

## 3. Epidemiological parameters

Full values and one source key per value are in `seir_parameters.json`.

### Stage durations
- **Intrinsic incubation** (human E→I): mean **5.9 days**, 95% range 3.4–10 —
  [Chan & Johansson 2012](https://doi.org/10.1371/journal.pone.0050972).
- **Extrinsic incubation** (mosquito): **15 days at 25 °C**, **6.5 days at 30 °C**
  — same source. Only needed for the SEIR-SEI arm.
- **Human infectious period:** 2–7 days (WHO, via
  [Yi et al. 2021](https://doi.org/10.1109/ACCESS.2021.3129997)). Phaijoo & Gurung
  use 3.04 days; Liu et al. use 7.

### Starting immunity (any-serotype IgG)
- Colombo children under 12, 2008–09: **51.4%**, and a force of infection of
  **14.1%/yr** (95% CI 12.7–15.6) —
  [PLoS NTD 2013](https://doi.org/10.1371/journal.pntd.0002259).
- Suburban Colombo 2013–14: **adults 90.8%, children 50.7%, overall 68.2%**
  (n=1,689) — [PLoS ONE 2015](https://doi.org/10.1371/journal.pone.0144799).
- Nine districts, ages 10–20, 2022–23: **24.8% overall**, from **14.2% in Badulla**
  to **54.3% in Trincomalee** (n=5,207) —
  [Jeewandara et al., J Med Virol 2024](https://doi.org/10.1002/jmv.29394). The
  per-district table for the other seven districts is in the paper. The
  publisher and medRxiv both block automated access, so it wasn't retrieved.

IgG does not tell us which serotype someone is immune to. The model needs
**susceptibility to the circulating serotype**. These figures give an upper bound
on immunity, not a direct S₀.

### Serotype timeline
- 2009: new DENV-1 genotype. **DENV-2 and DENV-3 not detected 2009 – mid-2016** —
  [PLoS NTD 2021](https://doi.org/10.1371/journal.pntd.0009624).
- **Late 2016–2017: DENV-2 cosmopolitan genotype, new to Sri Lanka.** 39 of 44
  positives from Dec 2016 – Dec 2017 were DENV-2 —
  [Emerg Infect Dis 2020](https://doi.org/10.3201/eid2604.190435).
- **Late 2019–2020: DENV-3 re-emerges** — PLoS NTD 2021.

That gives **two** candidate immunity resets inside our data: 2017 (DENV-2) and
late 2019 (DENV-3). Liu et al. modelled only the first.

### Ascertainment — no single answer exists
| Evidence | Implied factor |
|---|---|
| Paediatric cohort, Colombo 2008–10: 40 inapparent vs 27 apparent ([PMC4375390](https://pmc.ncbi.nlm.nih.gov/articles/PMC4375390/)) | ~2.5× infections per apparent case |
| Children under 12: ~30 primary infections per notified case (PLoS NTD 2013) | ~30× |
| Global: 390M infections vs 96M apparent ([Bhatt et al., Nature 2013](https://doi.org/10.1038/nature12060)) | ~4× |
| Liu et al. 2025 (from Bhatt) | 11× |
| Southeast Asia, symptomatic per reported ([Undurraga et al. 2013](https://doi.org/10.1371/journal.pntd.0002056)) — **Sri Lanka not included** | 7.6× (7.0–8.8) |

The evidence spans an order of magnitude. **Ascertainment has to be a sensitivity
parameter, not a constant.**

---

## Not yet sourced
- **2012 census district totals** (to cover 2013).
- **Per-district seroprevalence** for the seven remaining districts in Jeewandara
  et al. 2024.
- **Mobility.** Liu et al. used a radiation model built from population and
  district locations. District boundaries are available from
  [HDX](https://data.humdata.org/). It's only needed if we move beyond the
  adjacency graph.
