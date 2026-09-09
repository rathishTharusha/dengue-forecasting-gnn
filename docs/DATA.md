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
- **Axis 2 — features.** 11 channels. **The forecast target is index 5 (weekly case counts).**
  The remaining channels are meteorological/geospatial covariates: precipitation, air
  temperature, land-surface temperature, humidity, NDVI, soil moisture, canopy.

> Covariates are already **lag-shifted** by their empirically optimal delay (hence
> `_shifted` in the filename) — e.g. precipitation ≈ 12 weeks, minimum NDVI ≈ 17 weeks —
> reflecting mosquito-breeding and infection-to-diagnosis intervals. Do **not** apply a second
> lag on top of these.

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

Districts are nodes. Edges encode **shared physical borders** between districts (not
distance or mobility data -- reliable human-mobility data is unavailable for Sri Lanka at this
resolution). With self-loops: **139 directed edges**. Replacing this hand-built adjacency with
a learned adaptive one is Contribution (c).

The original hand-built file had two one-directional entries that could not be real shared
borders (`Kandy -> Ampara`, `Kegalle -> Kalutara`, no reciprocal either way) -- caught during
Phase-2 review, 2026-09-09. Both were removed rather than symmetrised: verified against actual
district polygon geometry from [GADM 4.1](https://gadm.org) (level-1 administrative
boundaries, `buffer(50m).intersects()` between every pair), and neither pair shares a border.
That check confirms the corrected file matches GADM **exactly** -- all 139 edges, none missing,
none extra. See `docs/PHASE2_REVIEW.md` and `src/dengue_gnn/models.py`'s
`build_fixed_adjacency(require_symmetric=True)`, which now rejects a one-directional entry
outright so this class of error cannot silently return.

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
