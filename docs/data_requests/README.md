# Data requests — who to write to, and what to ask for

Four institutions hold the data this project still needs. Each letter in this
folder is self-contained and ready to go on department letterhead or into an
email body.

| # | letter | institution | what it unlocks | measured value |
|---|---|---|---|---|
| 01 | [NDCU — entomological indices](01_ndcu_entomological.md) | National Dengue Control Unit | the mosquito half of SEIR–SEI, which currently has **no data at all** | untested — the vector side is pure assumption today |
| 02 | [Epidemiology Unit — testing effort](02_epid_testing_effort.md) | Epidemiology Unit, Ministry of Health | separates "more disease" from "more testing"; the reporting rate is one learned constant today | would also explain part of the 2020 drop |
| 03 | [MRI — serotype composition](03_mri_serotype.md) | Medical Research Institute | susceptible replenishment at a serotype switch | **measured**: switch timing alone moves the 2019 growth window from 46.6 to 44.4 RMSE, past the 45.5 baseline |
| 04 | [DCS — commuting matrix](04_dcs_commuting.md) | Department of Census and Statistics | replaces the geographic adjacency graph, which is measurably useless | graph currently contributes nothing (EXP-033) |

Attach [`annex_project_summary.md`](annex_project_summary.md) to every request.

## Before sending

1. **Fill every `[ ]` placeholder** — supervisor's name and title, department
   head, sender's contact details, date. Nothing should go out with a bracket
   left in it.
2. **Check the current addressee and address on the institution's website.**
   Directors change; this folder deliberately does not hard-code names or
   postal addresses that may be stale.
3. **Get your supervisor's signature or an explicit endorsement in the email.**
   An undergraduate request with a named academic supervisor behind it is
   treated very differently from one without.
4. **Send early.** A reply inside a month is a good outcome. Letters 01 and 03
   are the two worth chasing if only some arrive.

## Ground rules kept in every letter

- **No individual-level or patient-identifiable data is requested.** Everything
  asked for is aggregate: counts by district and month, or by serotype and
  quarter. This is stated explicitly, because it is the question every data
  custodian asks first.
- **The purpose is stated plainly** — an unfunded undergraduate research
  project, results published openly, source credited.
- **Exact variables, granularity and period are specified**, so the custodian
  can judge the effort in one reading rather than writing back to ask.
- **A fallback is offered** where a coarser version would still be useful, so a
  partial reply is easy to give.

## Tracking

| # | sent | acknowledged | reply | outcome |
|---|---|---|---|---|
| 01 NDCU | | | | |
| 02 Epidemiology Unit | | | | |
| 03 MRI | | | | |
| 04 DCS | | | | |

## If a request is declined

Record it here and in `docs/DATA_REQUIREMENTS.md`. A documented refusal is a
legitimate limitation to state in the write-up — "these data exist but were not
available to us" is a far stronger sentence than silence, and it tells the next
group where the wall is.
