# Frozen plan for the prospective test on 2024 W11 to 2026 W32 (EXP-063)

Written and committed **before** any case count from the new weeks is parsed and before any model
is run on them. `data/new_weeks/` does not exist at the time of this commit. After this commit the
evaluation design does not change because of final-test performance. Any change for another
reason (a bug, a missing input) is logged in `docs/EXPERIMENT_LOG.md` under EXP-063 with its
reason, everything affected is re-run, and both results are reported.

**Amendment 1 (2026-10-06, before any new-week count was parsed):** the AR baseline is recursive
AR(3), not one direct model per horizon; the bootstrap sensitivity block lengths are 4 and 12
(were 4 and 13); the inference unit is stated explicitly (section 8); the district-order mapping
of the parser output is stated (section 2). Made on external review of the plan, not on any
result.

The plan answers the four weaknesses an external review found in the current paper
(`.codex-paper-review/paper/full-paper-v3/FINAL.tex`): validation and test targets that overlap, a
graph with two wrong edges, no matched gated non-SEIR control, and no seasonal-naive or
autoregressive baseline. It also replaces development-period evidence with one genuinely later
period.

## 1. Order of work

| step | what | allowed to see new-week counts? |
|---|---|---|
| 1 | Commit this plan | no |
| 2 | Implement the purge, the final split, the two baselines and their tests; commit | no |
| 3 | Run the development tuning grid (section 6) on the existing 559 weeks; commit the chosen settings to `seirgnn2/results/prospective_frozen.json` | no |
| 4 | Parse the new weeks with the frozen rules in `docs/NEW_WEEKS_RULES.md`; the automatic checks R1-R6 are the only inspection of the counts | yes, from here on |
| 5 | Run every arm once on the final split (section 4); write one row per (arm, seed) and one row per (arm, seed, test start) | yes |
| 6 | Compute the statistics in section 8 and fill in EXP-063 | yes |

EXP-062 (the week-index re-run) uses only the development weeks and can run before or after
step 3. It does not change the arm list below.

## 2. Data

- **Development weeks:** rows 0-558 of `data/corrected/rebuilt_*` (2013-06-15 to 2024-02-24, last
  week 2024 W10), loaded through `analysis/lib/corrected_data.py` with the fixed week index
  (EXP-062 commits `ac89792`, `e2360c8`).
- **New weeks:** rows 559 onward, one row per Weekly Epidemiological Report, Vol. 51 No. 11 to
  Vol. 53 No. 33 (127 expected reports, `data/external/wer_new_manifest.csv`), parsed and corrected
  only by `docs/NEW_WEEKS_RULES.md` R1-R6. Missing weeks stay `NaN`; nothing is interpolated.
  The parser writes districts in report-column order; they are mapped by name to the graph's
  district order. Kalmunai and the national total are not districts and are not used, so Ampara
  is Ampara RDHS only, as in the development series.
- **Inputs:** case history only (`cases` lag 1, window 3) plus the population and cumulative-case
  state the SEIR head needs (`population` lag 0, by publication date). No climate, NDVI, season or
  COVID inputs: none of the arms below uses them.
- **Graph:** the corrected border graph, `notebooks/baseline/sri_lanka_adj_list.json` as of commit
  `3878e70` (57 GADM 4.1 shared borders, 139 directed entries with self-loops, enforced by
  `tests/test_adjacency_borders.py`). Every graph arm uses it. No result on the old 141-entry graph
  enters the final table.

## 3. Indexing and the purge rule

In the code a window is named by its **start** `i`, the first target row. Its inputs are rows
`i-3 .. i-1` (the forecast origin is `t = i-1`, the last observed week) and its targets are rows
`i, i+1, i+2`. Neighbouring starts therefore share `H-1 = 2` target rows.

**Purge rule.** At each train/validation and each validation/test boundary, the last `H-1 = 2`
forecast starts of the earlier partition are removed, so **no target row belongs to two
partitions.** Example: if the last training start is `i = s` (targets `s, s+1, s+2`), starts
`s+1` and `s+2` are used by nobody and the first validation start is `s+3` (targets
`s+3, s+4, s+5`). Inputs may come from an earlier partition; that is information available at
forecast time, not overlap.

A unit test asserts, for every fold built by the code, that the target rows of train, validation
and test are pairwise disjoint.

## 4. Splits

**Final split (the confirmatory result).** Let `D = 558`, the last development row.

- validation: the last 30 development starts whose targets all lie at or before `D`,
  i.e. `i = D-31 .. D-2`;
- training: every earlier start up to `i = D-34` (the purge removes `D-33` and `D-32`);
- test: every start `i >= D+1` whose three targets are new weeks (the starts `D-1` and `D` would
  straddle both periods and are not used).

Windows touching a missing week are dropped from every arm alike. Normalisation (log1p mean and
std) comes from training rows only. Each neural arm is trained once per seed on this split, with
early stopping on the validation partition, then frozen and applied to the whole test period with
no refitting.

**Development split (tuning and a secondary table).** The nine frozen origins in
`docs/PROTOCOL.md` with the purge applied: the last two training starts and the last two
validation starts are dropped at each origin. The test windows are unchanged, so the persistence
check value on test stays **28.5410**.

## 5. Arms

Fixed now. None is added, dropped or retuned after step 4.

| arm | config | role |
|---|---|---|
| **Adaptive SEIR-GNN** | `backbone="adaptive", adj_init="gwn", adj_shared=True, head="foi_res", lam_param="anchor", state_fit="encoder"` | the proposed model (EXP-061 finalist, the paper's name) |
| Adaptive gated non-SEIR | same encoder, `head="gated"` (same anchor, same gate, same init `a = -2`, no simulator) | matched control; the difference with the row above is the SEIR equations alone |
| Adaptive residual | same encoder, `head="residual"` | the best non-SEIR arm in EXP-061 |
| Persistence | `ŷ(i+h) = y(i-1)` for `h = 0, 1, 2` | floor |
| Seasonal naive | `ŷ(j) = y(j-52)` for each target row `j`; if that row is missing, persistence is used for that cell and the count of such cells is reported | classical baseline |
| AR(3) ridge | one-step ridge regression pooled over districts: `log1p y(t)` on `log1p` of lags 1-3 plus a district intercept, fitted on training starts only. Forecasts are **recursive**: the step-1 prediction is fed back as lag 1 for step 2, and so on; prediction `expm1` clipped at 0 | classical baseline |

All neural arms use `loss="mse_z"`, `dist="point"`, `epochs=300`, `patience=40`, `batch_size=32`,
`layers=2`, `dropout=0.1`, `weight_decay=1e-4`, seeds **0, 1, 2**.

## 6. Tuning, the same budget for every learned arm

Selection uses **validation RMSE only**, averaged over the nine purged development origins and the
three seeds.

- Each of the three neural arms: `lr ∈ {1e-3, 3e-3}` x `hidden ∈ {32, 64}` (4 settings, 108 runs
  per arm).
- AR(3) ridge: `alpha ∈ {0.01, 0.1, 1, 10, 100}`, chosen by the validation RMSE of its recursive
  3-step forecasts on the same development validation splits.
- Persistence and seasonal naive have nothing to tune.

The chosen settings and the full development table (validation and test, one row per arm, setting,
origin and seed) are committed in step 3, before step 4.

## 7. Metrics

RMSE on raw counts over every (test start, district, horizon) cell is the primary metric. MAE,
SMAPE and MAPE (truth >= 1), from `src/dengue_gnn/metrics.py`, and per-horizon RMSE are reported
for every arm. Per-calendar-year RMSE (2024, 2025, 2026) is reported as description only.

For each neural arm, each metric is computed per seed and averaged over the three seeds; the seed
range is reported next to it.

## 8. Statistical tests and multiplicity

Two **primary comparisons**, both on test RMSE of the final split:

- **P1:** Adaptive SEIR-GNN vs Adaptive gated non-SEIR (does the SEIR part add anything?)
- **P2:** Adaptive SEIR-GNN vs Persistence

**Inference unit: the paired forecast-start week.** The 25 districts of one week are one epidemic
state, not 25 independent observations, so they are never resampled separately.

Method: for each test start `i`, `L(i)` is the squared error averaged over the 25 districts, the 3
horizons and (for neural arms) the three seeds. `ΔRMSE = sqrt(mean L_A) - sqrt(mean L_B)`. Its 95%
interval and two-sided p-value come from a circular moving-block bootstrap: contiguous blocks of
**8** forecast starts are drawn with replacement, every district and horizon of a drawn start stays
with it, and both arms use the same draw. For each of 10,000 replicates (generator seed 0) the full
`ΔRMSE` is recomputed from the resampled `L` values. The p-value is twice the smaller share of
replicate `ΔRMSE` on either side of 0, capped at 1. Holm correction across P1 and P2. Block lengths
**4 and 12** are reported as sensitivity checks only; the primary result is block length 8 whatever
the other two show.

Every other pairwise difference in the final table gets the same interval, labelled as secondary
and uncorrected.

The development table (section 4) is tested as in `docs/PROTOCOL.md`: seeds averaged within
origin, exact paired sign-flip across the nine origins, Holm. It is secondary.

## 9. What may be claimed

Decided now, applied whatever the numbers are.

- "The SEIR component improves forecasts" only if P1 favours the SEIR arm with Holm-adjusted
  p < 0.05. Otherwise the paper says the gain was not confirmed, or reports the loss.
- "Beats persistence on later data" only if P2 favours the SEIR arm with Holm-adjusted p < 0.05.
- If a classical baseline beats the proposed model, the paper says so in the abstract.
- Every arm in section 5 is reported, including those that lose.
- If fewer than 52 test starts survive the missing-week rule, the result is reported as
  underpowered, not dropped.
- The development numbers (EXP-061, the paper's current 27.45 vs 28.54) are described as
  development evidence; only this run is described as a later, untouched test.

## 10. Not in this plan

Climate, seasonal and COVID inputs; the published encoders (STGAT, A3T-GCN, ASTGCN, AAGCN, LSTM);
the negative-binomial likelihood; sensitivity to the SEIR parameters. Any of these may be run later
as exploratory work on the development weeks, but not added to the confirmatory table.
