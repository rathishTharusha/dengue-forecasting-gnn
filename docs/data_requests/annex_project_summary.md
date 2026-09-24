# Annex — one-page project summary

*Attach this to every data request.*

---

## Forecasting district-level dengue incidence in Sri Lanka

**Group 05, CS3631 Research Project**
Department of Computer Science and Engineering, University of Moratuwa
Supervisor: **[ supervisor's name and title ]**
Unfunded undergraduate research. Code and results published openly.

### What the project does

We forecast weekly notified dengue cases for each of Sri Lanka's 25 districts,
three weeks ahead, and we test honestly whether the forecasts are any good.

The case series is the Epidemiology Unit's Weekly Epidemiological Reports,
2013–2024, digitised and verified district by district against every published
report. Alongside it we use ERA5 climate reanalysis, MODIS vegetation index,
Department of Census and Statistics population figures, and published
epidemiological parameters.

The model combines a **compartmental SEIR–SEI epidemic model** — the standard
representation of dengue transmission between people and mosquitoes — with a
**graph neural network** that learns how transmission varies across districts and
over time.

### What we have found so far

1. **The mechanism matters.** Putting the compartmental model in the forecasting
   path reduces error by 39% compared with an otherwise identical neural network
   without it. This holds across nine independent evaluation periods and two
   versions of the dataset.

2. **Accuracy is limited by information, not by method.** A naive forecast that
   simply repeats last week's count is extremely hard to beat at this horizon,
   and our best model only matches it. We have tested and ruled out four ways
   around this — larger models, combining models, forecasting further ahead, and
   adding more climate data. None of them helps, because none of them adds
   information.

3. **The remaining errors are explainable.** The model is accurate in ordinary
   years and fails in two specific periods: the 2019 epidemic's growth phase, and
   the 2020 collapse in notifications during COVID-19 movement restrictions. For
   the second, public policy and mobility data explains the drop. For the first,
   the missing information appears to be which dengue serotype is circulating.

### What we are asking for, and why

The project's conclusion is that further progress needs **better inputs, not
better algorithms**. We have measured, where possible, how much each missing
input would be worth before asking for it — for example, supplying only the two
published serotype-switch dates already improves the 2019 forecast enough to pass
the naive baseline for the first time.

The data requested in the accompanying letter is aggregate only. No
individual-level or patient-identifiable information is requested, and nothing
provided would be redistributed.

### What we can offer in return

- Our verified 2013–2024 district-level case table, with the verification results
  against every published report, and notes on two data-quality issues we found
  in the source PDFs.
- All code, openly licensed and documented, including the reproduction of a
  published international benchmark on Sri Lankan data.
- A measurement of what each data source is worth to forecasting accuracy, which
  may be useful evidence when arguing for surveillance resourcing.
- A presentation of results to the institution at its convenience.

### Contact

**[ Name ]** · **[ email ]** · **[ telephone ]**
Supervisor: **[ supervisor's name, title, email ]**
Department of Computer Science and Engineering, University of Moratuwa
