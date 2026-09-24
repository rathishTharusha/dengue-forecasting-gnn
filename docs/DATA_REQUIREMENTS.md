# Data we still need — what each one would buy, and how to get it

The forecast is **information-limited, not model-limited**. That is measured, not
assumed:

| measurement | value | where |
|---|---|---|
| corrected forecast vs last-value persistence | **r = 0.97 – 0.99** | EXP-035 |
| growth predictability from cases, climate, NDVI, neighbours | **r² ≈ 0** | EXP-024, EXP-034 |
| damping weight the model chooses for its own correction | **α ≈ 1/3** | EXP-035 |
| routes closed with evidence | architecture, ensembling, longer horizon, more climate | EXP-033 – EXP-036 |

So the next real improvement comes from **inputs the model does not have**, not
from a better model. This file lists those inputs, what each would fix, where to
get it, and how to check it is worth collecting before spending weeks on it.

---

## The rule: measure the value before you collect

Every candidate gets a **proxy test** first, at a cost of ~20 minutes:

1. Build the crudest possible stand-in from what is already known (a date, an
   indicator, a published annual figure).
2. Feed it to the frozen protocol — 9 disjoint origins × 3 seeds.
3. If the stand-in does not move the windows it is supposed to move, the real
   dataset almost certainly will not either.

This has already paid for itself once. The serotype-switch oracle (EXP-036) used
two published dates and showed the gain is **confined to epidemic-growth
windows** — 46.6 → 44.4 at origin 0.55, nothing at origin 0.60. That told us
serotype data is worth having *and* that it will not fix the 2020 collapse,
before anyone wrote to a lab.

**Rule R2 still applies to every row below.** A forecaster may only use what was
available at the forecast origin. Anything published after the fact goes in an
`oracle_*` arm under rule R4 and is never a finalist.

---

## Where the model currently loses

The two failing windows, from EXP-035 and EXP-036:

| window | period | national cases | what we do | error |
|---|---|---|---|---|
| origin 0.55 | 2019-05 → 2019-11 | rising 1386 → 2566/wk | under-predict | bias −12.9 |
| origin 0.60 | 2019-11 → 2020-06 | collapsing 3057 → 299/wk | over-predict | bias +19.8 |

Everything below is aimed at one of those two, or at what persistence cannot do
at all.

---

# Priority 1 — free, legitimately real-time, and aimed at the largest loss

## 1. Government response stringency

The 2020 collapse is not epidemiology. Cases fell ~90% as movement restrictions
began. No compartmental model can anticipate that from case history.

| | |
|---|---|
| **What** | Daily policy-stringency index for Sri Lanka (school/workplace closure, movement restriction, stay-at-home orders), 0–100 |
| **Source** | Oxford COVID-19 Government Response Tracker (OxCGRT), Blavatnik School of Government — `https://github.com/OxCGRT/covid-policy-dataset` |
| **Coverage** | 2020-01 to 2022-12, national (`LKA`); some sub-national series exist |
| **Granularity needed** | daily → aggregate to the project's week start |
| **Licence** | CC BY 4.0 — usable, cite the tracker |
| **Real-time?** | **Yes.** Published within days of the policy change, so a forecaster in 2020 could have used it. This is a covariate, *not* an oracle. |
| **Fixes** | origin 0.60, and the 2021 windows where cases stay suppressed |
| **Plugs into** | a covariate channel in `features_train_only()`, and optionally a multiplier on β (contact rate falls when movement is restricted — the mechanistically correct place for it) |
| **Effort** | one afternoon |

**Proxy test first:** build a 0/1 lockdown indicator from the published Sri Lankan
restriction dates, run the 9-origin protocol, and look only at origin 0.60. If the
indicator does not reduce the +19.8 bias, the full index will not either.

## 2. Human mobility

Dengue moves with people, not with mosquito flight. The project has already shown
that the **geographic adjacency graph adds nothing** (EXP-033, and the earlier
adaptive-graph result). A movement graph is the version of that idea that has not
been tested.

| | |
|---|---|
| **What** | Change in movement relative to a baseline, by district or province |
| **Sources** | Google COVID-19 Community Mobility Reports (archived, 2020-02 → 2022-10); Meta *Movement Range Maps* via the Humanitarian Data Exchange, `https://data.humdata.org` |
| **Granularity needed** | daily by sub-region → weekly by district |
| **Licence** | Google: free with attribution. Meta/HDX: check the dataset's own terms |
| **Real-time?** | Yes, both were published with a few days' lag |
| **Limitation** | **2020 onwards only.** Useless for 2013–2019, so it can only help the late windows |
| **Fixes** | origin 0.60 onward; and gives the graph a reason to exist |
| **Plugs into** | the adjacency used to build `hat_A` in `run_s5_seir_gnn_v2.py`, replacing geographic neighbours with a row-normalised movement matrix |
| **Effort** | two days, mostly district-name alignment |

---

# Priority 2 — already in the repository, never used

## 3. Climate extremes, not weekly means

EXP-024 tested **weekly means** of ERA5 channels and found r² = 0.021 against
cases and −0.03 against log growth. That is a null result for *means*. Mosquito
breeding responds to things a weekly mean erases.

| | |
|---|---|
| **What** | Derived from `data/external/era5_weekly_by_district.csv` and the hourly ERA5 archive: days above/below a rainfall threshold, consecutive dry days followed by rain, days in the 22–32 °C transmission window, diurnal temperature range, and the temperature-dependent extrinsic incubation period |
| **Source** | Copernicus Climate Data Store (already used — see `DATA_PROVENANCE.md` §3); hourly re-download needed only for the intra-week features |
| **Licence** | Copernicus, already cleared |
| **Real-time?** | ERA5 has a ~5-day lag; ERA5T is near-real-time. Acceptable at a 3-week horizon |
| **Fixes** | possibly the epidemic-growth windows — dry-spell-then-rain is the classic container-breeding trigger |
| **Plugs into** | additional channels in `features_train_only()`; the EIP is a principled input to a temperature-dependent `omega` |
| **Effort** | one day for the weekly-file derivations, three for hourly |

**Why it is worth a try despite the null:** the null was measured on means. A
quantity that a mean destroys has not been tested, and this costs no new
collection.

## 4. The serotype timeline already written down

`SEIR_DATA_SOURCES.md` §"Serotype timeline" already records, with citations:
DENV-2 cosmopolitan genotype late 2016–2017, DENV-3 re-emerging late 2019–2020,
and DENV-2/3 absent 2009 to mid-2016. EXP-036 used only the two switch dates.

The richer version — **proportions per serotype over time** — is the next step,
and much of it is extractable from the published papers those citations point to.

| | |
|---|---|
| **What** | Share of DENV-1/2/3/4 among typed specimens, by quarter or year, nationally |
| **Sources** | The three papers cited in `SEIR_DATA_SOURCES.md`; Medical Research Institute (MRI), Colombo; the Malavige group at Sri Jayewardenepura |
| **Real-time?** | **No** — published years later. Digitised from literature it is an **oracle** under R4, and can only quantify the value of surveillance, never forecast |
| **Fixes** | epidemic-growth windows (measured: 46.6 → 44.4 at origin 0.55) |
| **Effort** | two days of careful reading and transcription |

---

# Priority 3 — one request away

These need a letter from the department. Ask early; replies take weeks.

## 5. Entomological surveillance

The model is SEIR–**SEI**. The mosquito half currently has **no data feeding it at
all** — every vector quantity is a literature constant.

| | |
|---|---|
| **What** | Larval indices (Breteau, House, Container), ovitrap positivity, adult trap counts |
| **Holder** | National Dengue Control Unit, Ministry of Health — `https://www.dengue.health.gov.lk` |
| **Granularity wanted** | district × month, 2013 onwards |
| **Real-time?** | Collected routinely; publication lag unknown — ask |
| **Fixes** | the mosquito side of the mechanism, which is currently assumption-only; plausibly the growth windows |
| **Effort** | a request, then alignment |

## 6. Testing and confirmation effort

Our reporting rate ρ is a single learned constant. Some of what we score as error
is reporting behaviour, not disease.

| | |
|---|---|
| **What** | Specimens tested per week, NS1 vs IgM confirmations, hospital admissions vs notifications |
| **Holder** | Epidemiology Unit (`https://www.epid.gov.lk`) and MRI |
| **Fixes** | separates "more disease" from "more testing"; would also explain part of the 2020 drop |
| **Effort** | a request |

## 7. Commuting matrix

| | |
|---|---|
| **What** | Origin–destination flows between districts (work/school travel) |
| **Holder** | Department of Census and Statistics — `http://www.statistics.gov.lk`, Census of Population and Housing 2012, migration and travel tables |
| **Real-time?** | Static — one matrix, applied to all weeks. That is a limitation, not a leak |
| **Fixes** | replaces geographic adjacency, which is measurably useless |
| **Effort** | one day if the table is published; a request if not |

---

# Priority 4 — valuable, realistically out of reach for this project

- **Intervention logs** — fogging rounds, clean-up campaigns, the 2017 emergency
  response. Would explain post-peak collapses. Rarely digitised.
- **Age and severity breakdown of notified cases.** A shift in the age
  distribution is an early signature of a serotype change, and it is in the
  notification forms.
- **Private-sector notifications.** A known and unquantified gap in the case data.

---

# Summary

| # | dataset | fixes | free | real-time | effort | expected value |
|---|---|---|---|---|---|---|
| 1 | Policy stringency (OxCGRT) | 2020 collapse | yes | yes | hours | **high** |
| 2 | Mobility (Google / Meta) | 2020+, and the graph | yes | yes | days | high |
| 3 | Climate extremes from existing ERA5 | growth windows | yes | yes | 1–3 days | medium |
| 4 | Serotype proportions | growth windows | yes | **no — oracle** | days | medium, measurable only |
| 5 | Entomological indices | the vector half | request | ask | weeks | medium |
| 6 | Testing effort | reporting vs disease | request | ask | weeks | medium |
| 7 | Commuting matrix | the graph | likely | static | days | low–medium |

## Do not collect

Measured nulls — spending effort here is not justified by the evidence:

- **More weekly-mean climate channels.** r² = 0.021 against cases, −0.03 against
  log growth (EXP-024). Adding channels of the same kind changed nothing
  (EXP-032).
- **Finer NDVI.** Tested, no effect.
- **More architectures or an ensemble of them.** Every candidate collapses onto
  the same persistence-like forecast, r = 0.97–0.99 (EXP-035).

## How to ask

A request to NDCU, MRI or the Epidemiology Unit should state: the research group
and supervisor; that this is an unfunded undergraduate research project; the exact
variables, granularity and years wanted; that results will be published openly
with the source credited; and that no individual-level data is requested. Attach
the project's one-page summary. Send early — a reply in under a month is a good
outcome.

---

*Written after EXP-032 – EXP-037. Each "fixes" column entry refers to a window in
the frozen 9-origin protocol, and every measured figure quoted here is
reproducible from `analysis/_build/`.*
