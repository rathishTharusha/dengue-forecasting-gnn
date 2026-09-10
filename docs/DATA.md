# Data

## Tracked in this repo

| File | Shape / size | What it is |
|---|---|---|
| `notebooks/baseline/sri_lanka_2013-2022_shifted.npy` | `(459, 25, 11)` | Processed feature array — 459 weeks × 25 districts × 11 features |
| `notebooks/baseline/sri_lanka_adj_list.json` | 25 nodes | District adjacency list used to build the graph |

These two are small and are the *only* files the Phase-1 baseline notebook needs.

### The array

- **Axis 0 — time.** 459 consecutive weeks spanning 2013–2022.
- **Axis 1 — space.** 25 Sri Lankan districts, in the order given by the adjacency JSON.
- **Axis 2 — features.** 11 channels, identified exactly below.

### The 11 channels, by name and lag

Earlier revisions of this file described the channels from the paper's prose and
got it wrong. They are now identified empirically: every channel was matched
against the **named** columns of `Data/Datasets/sri_lanka_2013-2022_vertical.csv`
in the authors' repository, at every lag, with **100% exact agreement** on all
eleven. The derivation runs in
[`analysis/notebooks/E1_dataset_eda.ipynb`](../analysis/notebooks/E1_dataset_eda.ipynb) (finding F1).

| index | channel | lag (weeks) |
|---|---|---|
| 0 | `meanTair_F_Inst` | 0 |
| 1 | `minTair_F_Inst` | 0 |
| 2 | `maxTair_F_Inst` | 0 |
| 3 | `meanQair_F_Inst` (specific humidity) | 0 |
| 4 | `meanSoilmoi0_10Cm_Inst` | 0 |
| **5** | **`cases`** ← forecast target | 0 |
| 6 | `meanCanopint_Inst` | 12 |
| 7 | `meanPrecipitationcal` | 12 |
| 8 | `minPrecipitationcal` | 12 |
| 9 | `maxPrecipitationcal` | 12 |
| 10 | `minNdvi` | 17 |

**Corrections to what this file used to say:**

* There is **no land-surface-temperature channel**. The three temperature
  channels are 2 m air temperature from GLDAS, in **Kelvin**.
* `meanNdvi`, `maxNdvi` and `meanPsurf_F_Inst` exist in the source CSV but were
  **excluded** from the array. Only `minNdvi` survives, at lag 17.
* The lags above match the paper's §IV exactly (precipitation 12, minimum NDVI
  17, mean canopy 12), which independently confirms this is the array the paper
  describes.

> Covariates are already **lag-shifted** (hence `_shifted` in the filename),
> reflecting mosquito-breeding and infection-to-diagnosis intervals. Do **not**
> apply a second lag on top of these.

### ⚠️ 4% of the weather values are `0`, not missing

The released array has zero NaNs because the authors applied `np.nan_to_num` on
load. The source CSV is ~4% missing in the GLDAS columns, and those gaps became
**zeros** — which for a Kelvin temperature is 290 K away from every real value.

Under z-scoring the damage is severe:

```
meanTair std with zeros:    58.72     without zeros: 2.23
real 13.8 K spread        = 0.23 sd   should be:     6.16 sd
usable dynamic range compressed to 3.8% (26x)
```

This does **not** affect any reproduced result — the five GNNs in Weng et al. run
with `use_disease_only=True` and never read these channels. It matters a great
deal for anything this project builds that *does* use them: a physics-informed
loss reading temperature, or a GAN conditioned on meteorology, would be reading a
corrupted signal.

**Before using covariates:** mask rather than zero-fill, or interpolate (the gaps
are scattered, not one block), and normalise over observed values only.

### ⚠️ Week 395 is a reporting artifact

One week carries 29% of its year's national cases, at 19× the local median, with
18 of 25 districts spiking simultaneously. Epidemics do not do that; reporting
backlogs do. Any model trained or scored across it is fitting an artifact, and a
generative model trained on this array will learn to reproduce it.

### The unshifted array can be reconstructed

Earlier notes recorded the unshifted dataset as unavailable. It is not: the
shifted array is exactly `nan_to_num(reconstruction from vertical.csv)` —
verified — so taking every column at lag 0 rebuilds the unshifted version. E1
does this and writes
`analysis/results/sri_lanka_2013-2022_unshifted.npy`. That potentially reopens
the ten "Unshifted Dataset" rows of Weng et al.'s Table I.

### Target distribution — the fact that drives the modelling

| Statistic | Value |
|---|---|
| Median weekly cases | 13 |
| Max weekly cases | 2631 |
| Zero-case district-weeks | 9.7% |
| Lag-1 autocorrelation | ≈ 0.68 |

Consequences, all of which are already baked into the baseline:

- Plain MAPE explodes on zero weeks → we report **SMAPE** plus a zero-masked **MAPE(≥1)**.
- MSE on the raw scale is hijacked by the tail → we model in **log1p** space.
- Strong autocorrelation makes persistence a hard baseline → the GNN predicts the
  **residual over persistence**, not the absolute count.

See [`decisions/0001-baseline-training-refinements.md`](decisions/0001-baseline-training-refinements.md).

### Graph

Districts are nodes. Edges encode inter-district adjacency built from district-to-district
distances (reliable human-mobility data is unavailable for Sri Lanka at this resolution).
With self-loops: **141 directed edges**. Replacing this hand-built adjacency with a learned
adaptive one is Contribution (c).

---

## Not tracked (git-ignored, local only)

| Path | Size | Why it's ignored |
|---|---|---|
| `datasets/output_Dengue Fever.csv` | 3.5 MB | Raw multi-source dump; superseded by the processed `.npy`. Columns: `Disease Name, Cases, Location Name, Country Code, Region Type, Lattitude, Longitude, Region Boundary, TimeStampStart, TimeStampEnd, Source File`. Row grain = one district-week, sourced from Epidemiology Unit weekly report PDFs. |
| `papers/` | ~4.5 MB | Copyrighted third-party PDFs — not ours to redistribute. |
| `reference_repo/` | small | Clone of [`MLOpenSourceOpenScience/disease_modeling_MLOS2`](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2). Read-only cross-check material belonging to its authors. Clone it yourself if you need it. |
| `CS3631 - Project - 2026.pdf` | 135 KB | Course handout. |

If a new teammate needs these, share the Drive folder — do not commit them.

---

## Provenance

- **Case counts** — Sri Lanka Ministry of Health / Epidemiology Unit weekly reports,
  2013–2022, per district.
- **Environmental covariates** — NASA EarthData satellite products, aggregated to weekly
  resolution per district.
- **District boundaries** — GADM shapefile; each district is a GIS polygon.
- **Assembly and lag-shifting** follow Weng et al. (2024), *Graph Representation Learning for
  Dengue Forecasting*, IEEE BigData — we reproduce their benchmark before adding anything.

### Optional generalization set

[OpenDengue](https://opendengue.org/) (Clarke et al., *Scientific Data* 11:296, 2024) —
>56M dengue records across 102 countries. Reserved as an out-of-country generalization test;
not part of the core protocol.

---

## Licensing

The MIT license in this repo covers **code only**. The datasets carry their sources' own terms
(Ministry of Health reports, NASA EarthData, GADM), and the papers in `papers/` are
copyrighted by their publishers. Cite the sources above in any write-up; do not re-publish the
raw data without checking each source's terms.

---

## Regenerating the processed array

The `.npy` was produced upstream from the raw CSV + satellite covariates. `reference_repo/preprocess.py`
is the reference for how that assembly works. We have not re-run it — we consume the processed
array as given so our numbers stay comparable to the published benchmark. If a future stage
needs different features or a longer span, treat re-deriving the array as its own task, and
re-run the *entire* ablation table afterwards.
