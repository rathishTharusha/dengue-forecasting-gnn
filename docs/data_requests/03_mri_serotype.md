**[ Department letterhead ]**

Date: **[ DD Month YYYY ]**

The Director
Medical Research Institute
Ministry of Health
**[ address — verify on www.mri.gov.lk ]**

Dear Dr **[ surname ]**,

## Request for aggregate dengue serotype surveillance data for an undergraduate forecasting study

We are a final-year undergraduate group in the Department of Computer Science and
Engineering, University of Moratuwa, working on a research project that forecasts
district-level dengue incidence in Sri Lanka. The project is supervised by
**[ supervisor's name and title ]** and is unfunded academic work.

This request is the one our own results point to most directly, so we set out the
evidence briefly.

### Why serotype data, specifically

Our model represents the susceptible population explicitly. When the dominant
serotype changes, a large share of the population loses effective protection, and
the model has no way to know this has happened.

We tested how much that matters, using only the two switch dates already
published in the literature — the DENV-2 cosmopolitan genotype in late 2016–2017
and DENV-3 re-emerging in late 2019. Supplying those two dates to the model
improved forecast error in the 2019 epidemic-growth window from RMSE 46.6 to
44.4, moving it past the naive baseline of 45.5 — the only condition under which
our model has beaten that baseline on a growing epidemic. Two dates did this;
an actual serotype series would carry considerably more.

We should be clear that in that experiment the dates were used with hindsight,
which is why we treat the result as a measurement of what surveillance is worth
rather than as a forecast.

### Data requested

All aggregate. **No individual-level or patient-identifiable data is requested.**

| Variable | Granularity | Period |
|---|---|---|
| Number of specimens serotyped | quarter (month if available) | 2013 to present |
| Count or share by serotype (DENV-1 / 2 / 3 / 4) | quarter × national | 2013 to present |
| The same by district or province, where the sampling supports it | quarter | 2013 to present |
| Genotype where routinely determined | year | 2013 to present |
| Sentinel sites contributing specimens, and any change in that panel over time | — | 2013 to present |

The last row is requested because a change in which sites contribute will change
the apparent serotype mix without any change in circulation, and we would rather
model that than mistake it for signal.

**If the full series is not readily available**, annual national serotype shares
alone would still be a substantial improvement on the two dates we have, and we
would be glad to receive whatever period is straightforward to compile.

### Purpose and handling

- Purpose: research only, to test whether serotype composition improves
  forecasts of dengue incidence, and to quantify what routine serotype
  surveillance would be worth to a forecasting system.
- Aggregate use only; no redistribution of any file provided.
- Findings published openly, with the Medical Research Institute credited.
- We would be glad to share the results with the Institute before publication,
  and to present the analysis of what serotype information contributes, which may
  be useful in arguing for surveillance resourcing.
- Any format is workable — CSV or Excel is easiest; published summary tables are
  acceptable and we will transcribe them.

### Contact

**[ Name ]**, **[ Registration number ]**
Department of Computer Science and Engineering, University of Moratuwa
Email: **[ email ]** · Telephone: **[ number ]**

Supervisor: **[ supervisor's name, title, email ]**

A one-page summary of the project is attached.

Yours sincerely,

**[ signature ]**

**[ Name ]**
on behalf of Group 05, CS3631 Research Project
Department of Computer Science and Engineering, University of Moratuwa

*cc: **[ Head of Department ]**, Department of Computer Science and Engineering*
