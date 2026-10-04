# SEIR head + adaptive graph audit — full result sheet

**Branch:** `fix/seir-adaptive-audit`, commit `5305bb3`. **Status: inconclusive, not adopted.**
Logged as EXP-059/060/061 in `docs/EXPERIMENT_LOG.md`; this file is the long-form companion with
every number, the defects found, and the fixes applied.

This is **not** a frozen result. Six of the thirty arms below were hand-picked by engineering
judgement (not pre-registered, not blind), as a quick check before committing to the full
46-config `audit()` grid. Treat the ranking as a lead, not a finding.

---

## 1. Why this audit happened

The user asked whether `adaptive graph + SEIR` combinations were implemented correctly before
running any of them. Reading `seirgnn2/models.py`, `train.py`, `core.py`, `diagnose_foi.py`
against the source papers found three real defects — not just "physics doesn't help":

| # | Defect | Why it matters |
|---|---|---|
| **S1** | The force-of-infection parameterisation (`lam_param="log"`/`"sigmoid"`) had no population or susceptible-fraction scale. New infections = λ·S·pop, but the net only ever saw one global bias — it had to learn an offset of roughly `-log(pop·S)` per district, blind, with population ranging 25× across districts (sd of log pop = 0.84). | Liu et al. 2025 (the SEIR-LSTM this project extends) write λ as explicitly per-capita (`~ a·b·h·A·I/N`); the repo's version was scale-blind by construction. |
| **S1b** | The repo's own claim — "λ is only r² ≈ 0.25 predictable, deficit is structural" — regressed oracle log λ on log cases and log S only. Population was never included. | See §3 below: this reframed the deficit from "random" to "partly a missing feature." |
| **S2** | The spatial-import term in the physics head centred its log-infected-neighbour value with `.mean()` and no `dim=` argument — a mean over the **whole batch**, not per window. Train batches are 32 windows; eval passes a whole split. A forecast's output depended on which other windows happened to share its minibatch, including later test weeks. | Correctness bug and a leakage-hygiene problem, confirmed to move predictions (see §4, regression test). |
| **A1** | The adaptive graph's node embeddings initialised at `randn * 0.05`. At that scale `softmax(ReLU(E1·E2ᵀ))` comes out **uniform to four decimal places** — "adaptive" was secretly computing a plain mean over all 25 districts, never a real graph, and nobody had checked. Adam's weight decay also pulls the embeddings back toward zero (= uniform) during training. | Graph WaveNet (Wu et al. 2019), which this construction is copied from, initialises embeddings at `randn` scale with no such collapse. |

Lower-severity findings, fixed or documented but not expected to move numbers much: E0 seeded
from one global scalar rather than per-district-window (S3); incubation rate ω = 0.1/day vs. the
measured dengue intrinsic incubation period of 5.9 days (Chan & Johansson 2012) (S4); permanent
immunity / single Colombo S₀ for all districts (S5); a stale docstring in `seir_sim.py` (S6); the
`adaptive`/`hybrid` graph trains two separate matrices per layer where Graph WaveNet shares one
(A2); the `hybrid` docstring's claim about matching Graph WaveNet's combination rule was false —
Graph WaveNet sums separate-weight diffusion terms, this repo's `hybrid` was a 0.5/0.5 average
through shared weights (A3); `gat` is a literal alias of `gcn`, not attention (A4, pre-existing
and already flagged in CLAUDE.md).

Sources: [Liu et al. 2025, PLoS Comp Biol](https://pmc.ncbi.nlm.nih.gov/articles/PMC12500091/) ·
[Graph WaveNet, Wu et al. 2019](https://ar5iv.labs.arxiv.org/html/1906.00121) ·
[Chan & Johansson 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3511440/)

---

## 2. Fixes applied (`seirgnn2/models.py`, `train.py`, `core.py`, `grids.py`, `sweep.py`)

All new behaviour is opt-in; every existing default is unchanged.

- **`lam_param="anchor"`** — λ = λ̂<sub>persist</sub> · exp(raw), where λ̂<sub>persist</sub> is the
  closed-form rate that would exactly reproduce last week's reported count (derived from the SEIR
  state and `rho`, `pop`). `raw = 0` now starts at persistence's own scale instead of an arbitrary
  global constant.
- **`lam_param="mass"`** — β·I/N mass-action, recomputed daily (`seir_sim.simulate_closed_loop`),
  matching Liu et al.'s stated form. The encoder predicts β only, which needs no per-district
  scale.
- **`state_fit="encoder"`** — the E₀ rescaling multiplier is now a function of the encoder's
  per-district-window representation (`nn.Linear(feat, 1)`, zero-initialised so it starts
  identical to the old global-scalar fit), instead of one number shared by the whole dataset.
- **S2 fix** — the spatial import term now centres per sample, over the window's own neighbours
  (`fixed @ I₀`, log, centred per row), never across the batch.
- **`adj_init="gwn"`** — node embeddings drawn `randn` (Graph WaveNet's own scale, 10 dimensions)
  instead of `randn * 0.05`.
- **`adj_shared=True`** — one adaptive matrix (`E1, E2` on the `Net`, not per `GraphLayer`) used
  by every layer, matching Graph WaveNet.
- **Weight-decay exemption** — under `adj_init="gwn"`, the adjacency embeddings get
  `weight_decay=0` in the optimiser's param groups (`train.py::_optimizer`), so Adam cannot pull
  them back toward the uniform collapse.
- **`backbone="uniform"`** — a fixed 1/N matrix, added as a mean-pooling control (not yet run in
  this quick pass — see §6).
- **Adjacency diagnostics** — every `adaptive`/`hybrid` run now logs `adj_entropy` (normalised row
  entropy, 1.0 = uniform) and its top-5 edges, so a learned graph can be inspected rather than
  assumed.
- **`origins="frozen9"`** — `docs/PROTOCOL.md`'s nine-origin split (0.50→0.90 step 0.05,
  `test_frac=0.05`) wired into `sweep.py`, verified byte-for-byte against
  `analysis/_build/protocol_check.py` (persistence 28.5410, every per-origin value matching to
  four decimals).

---

## 3. EXP-060 — did fixing S1 change the "structural deficit" framing?

Re-ran the existing `diagnose_foi.py` learnability regression (bisection-inverted oracle λ, origins
0.55 / 0.70 / 0.85, no training) with population and susceptible fraction added as regressors.

| regressor | r² (all cells) | r² (reachable cells only) |
|---|---|---|
| log cases[t-1] | 0.216 / 0.254 / 0.259 | 0.363 / 0.384 / 0.353 |
| + log pop | 0.218 / 0.257 / 0.260 | 0.512 / 0.515 / 0.444 |
| + log pop + log S | 0.224 / 0.266 / 0.273 | 0.589 / 0.660 / 0.674 |
| log pop alone | 0.058 / 0.066 / 0.074 | 0.001 / 0.004 / 0.005 |

(three numbers per cell = the three origins, in order)

39–41% of cells are pinned at λ=0 (target unreachable from below — E₀ alone already overshoots)
at every origin. That floor dominates the all-cell r², which is why adding population barely
moves it (0.216→0.218): population can't move a point that's already pinned. Restricted to
**reachable** cells, population lifts r² from ~0.37 to ~0.51, and adding S (which the network
never sees either) to ~0.59–0.67 — well above the originally-reported 0.25 figure.

**Conclusion:** the deficit was real but conflated two things. Part is genuinely a reach problem
(the λ=0 floor, S3, untouched by this fix). Part was a parameterisation gap — the free `log`
lambda asked the network to reproduce a `-log(pop·S)` offset it was never shown, which `anchor`
and `mass` now supply directly. `CLAUDE.md` and `seirgnn2/README.md` updated to point here instead
of restating the old headline figure as settled.

---

## 4. Regression check — did the fixes change any existing number?

Re-ran `seirgnn2/sweep.py screen` (135 runs, same config as the committed `screen.json`) after the
fixes, compared row-for-row by `(name, origin, seed)`:

**120 / 138 rows bit-for-bit identical.** The only arms that changed: `head=foi` (max |ΔRMSE| =
0.75) and `head=foi_res` (max |ΔRMSE| = 0.10) — exactly the two heads touched by the S2
batch-centring fix, and only by the amount that bug was actually worth. Every `direct`/`residual`
arm, every loss variant, every `graph=` arm (including the old, unfixed `adaptive`) reproduced
exactly. A dedicated unit test (`tests/test_forecast_does_not_depend_on_the_rest_of_the_batch`)
confirms the old code drifted (max |Δ| = 1.3e-3 between a window scored alone vs. in a batch) and
the new code does not (`torch.testing.assert_close`, default tolerance).

12 tests added in `tests/test_seir_adaptive_audit.py`, all passing; full suite (94 + 12) passes;
`ruff check`/`format --check` on `src tests tools` clean.

---

## 5. EXP-059 — the pre-fix picture, full backbone × head grid, frozen protocol

Before any fix, every backbone (gcn + 5 published architectures, DCRNN excluded per instruction)
crossed with every head (direct, residual, foi_res, foi), on `docs/PROTOCOL.md`'s frozen nine
origins. 657 rows, `seirgnn2/results/frozen9_all.json`.

**Persistence:** RMSE 28.5410, MAE 13.9386, val RMSE 27.0363 — exact match to
`analysis/_build/protocol_check.py`.

See the combined table in §7 (rows marked `old` and `pre-fix`) — this grid's numbers are included
there directly rather than duplicated. Headline: nothing beat persistence significantly; bare
`foi` broke on every backbone (+12 to +17 RMSE); `STGAT+direct` and `A3TGCN+direct` were
structurally broken (44–48 RMSE) independent of protocol, matching their behaviour on the old
3-origin numbers in `real.json`/`rescue.json`.

---

## 6. EXP-061 — six post-fix arms, hand-picked

Chosen, not pre-registered: `adaptive_gwn+residual` (the fixed graph, no physics — a control for
whether the graph itself moved), `adaptive_gwn+foi_res anchor` and `+foi_res anchor E0enc` (the
gated SEIR head under the two S1/S3 fixes), `adaptive_gwn+foi anchor` (the bare head that broke
worst pre-fix, under the S1 fix alone), `adaptive_gwn+foi_res mass` (the alternative λ
parameterisation), and `gcn+foi_res anchor` (same head fix, unchanged graph — separates "the fix
helped" from "the fix helped *because of the graph*"). All six: `loss="mse_z"`, `epochs=300`,
`origins="frozen9"`, matching every other arm in this sheet.

`adaptive_gwn` adjacency entropy across all 135 trained runs: **mean 0.469, range 0.315–0.618**
(1.0 = uniform). Confirms A1 is fixed — the graph is no longer frozen at exact uniform. Sampled
top-5 edges by weight, e.g. `Kalutara<-Kandy:0.918`, `Batticaloa<-Puttalam:0.812`,
`Jaffna<-Kegalle:0.722`, `Trincomalee<-Kurunegala:0.912`, `Anuradhapura<-Puttalam:0.896` — **these
are not neighbouring districts** on Sri Lanka's map. The graph is learning *some* consistent
structure (entropy well below uniform, stable top edges across runs) but it does not look like
geography; it may be tracking shared outbreak timing or population-scale similarity instead. Worth
investigating before trusting the graph's edges as interpretable.

---

## 7. Full result table — all 30 arms, origin unit (n=9), docs/PROTOCOL.md frozen nine origins

Persistence: **val RMSE 27.04, test RMSE 28.54, test MAE 13.94.** Sorted by val RMSE (the
project's selection rule, plan R6). `Δ` = arm − persistence, negative is better. `p_adj` is
Benjamini-Hochberg-adjusted, exact paired sign-flip test, 9 origins (seeds averaged within origin
first — the valid pairing unit; see `seirgnn2/stats.py`'s own docstring on why `origin_seed` must
never be quoted alone). `\*` = p_adj < 0.05.

| arm | val RMSE | Δval | p_adj(val) | test RMSE | Δtest | p_adj(test) | test MAE | Δmae | p_adj(mae) | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| ASTGCN+residual | 25.23 | -1.80 | 0.023\* | 28.65 | +0.11 | 0.984 | 13.98 | +0.04 | 0.957 | old |
| AAGCN+residual | 25.24 | -1.80 | 0.023\* | 28.61 | +0.07 | 0.984 | 13.87 | -0.07 | 0.957 | old |
| ASTGCN+direct | 25.35 | -1.68 | 0.023\* | 28.94 | +0.40 | 0.984 | 14.27 | +0.33 | 0.667 | old |
| **adaptive_gwn+residual** | **25.42** | **-1.62** | **0.023\*** | 27.94 | -0.60 | 0.694 | 13.61 | -0.33 | 0.352 | **NEW** |
| LSTM+direct | 25.45 | -1.59 | 0.023\* | 28.73 | +0.19 | 0.984 | 14.00 | +0.06 | 0.957 | old |
| AAGCN+direct | 25.46 | -1.58 | 0.032\* | 29.26 | +0.72 | 0.965 | 14.06 | +0.12 | 0.957 | old |
| **adaptive_gwn+foi_res anchor E0enc** | **25.53** | **-1.51** | **0.047\*** | **27.45** | **-1.10** | 0.576 | **13.29** | -0.65 | 0.181 | **NEW** |
| gcn+residual | 25.55 | -1.49 | 0.081 | 28.12 | -0.42 | 0.781 | 13.62 | -0.32 | 0.460 | old |
| gcn+direct | 25.64 | -1.40 | 0.158 | 28.17 | -0.37 | 0.984 | 13.66 | -0.28 | 0.547 | old |
| **adaptive_gwn+foi_res anchor** | 25.74 | -1.30 | 0.090 | **27.45** | **-1.09** | 0.576 | 13.31 | -0.63 | 0.176 | **NEW** |
| **gcn+foi_res anchor** | 25.83 | -1.20 | 0.158 | 27.62 | -0.92 | 0.576 | 13.39 | -0.55 | 0.176 | **NEW** |
| LSTM+residual | 25.93 | -1.11 | 0.158 | 27.73 | -0.81 | 0.576 | 13.43 | -0.51 | 0.215 | old |
| A3TGCN+residual | 26.32 | -0.71 | 0.290 | 28.24 | -0.30 | 0.965 | 13.76 | -0.18 | 0.667 | old |
| AAGCN+foi_res | 26.90 | -0.14 | 1.000 | 28.49 | -0.05 | 0.984 | 13.50 | -0.43 | 0.667 | pre-fix |
| ASTGCN+foi_res | 26.91 | -0.13 | 1.000 | 28.53 | -0.01 | 0.984 | 13.57 | -0.36 | 0.669 | pre-fix |
| gcn+foi_res | 26.92 | -0.12 | 1.000 | 28.43 | -0.11 | 0.984 | 13.51 | -0.43 | 0.667 | pre-fix |
| **adaptive_gwn+foi_res mass** | 26.93 | -0.11 | 1.000 | 28.39 | -0.16 | 0.984 | 13.52 | -0.42 | 0.667 | **NEW** |
| STGAT+residual | 27.01 | -0.02 | 1.000 | 28.73 | +0.19 | 0.984 | 14.04 | +0.10 | 0.722 | old |
| STGAT+foi_res | 27.02 | -0.01 | 1.000 | 28.60 | +0.06 | 0.984 | 13.54 | -0.40 | 0.667 | pre-fix |
| LSTM+foi_res | 27.08 | +0.04 | 1.000 | 28.47 | -0.07 | 0.984 | 13.53 | -0.41 | 0.667 | pre-fix |
| A3TGCN+foi_res | 27.09 | +0.05 | 1.000 | 28.47 | -0.07 | 0.984 | 13.52 | -0.42 | 0.667 | pre-fix |
| **adaptive_gwn+foi anchor** | 27.99 | +0.95 | 0.591 | 29.80 | +1.26 | 0.781 | 13.84 | -0.09 | 0.915 | **NEW** |
| STGAT+direct | 35.99 | +8.95 | 0.033\* | 44.71 | +16.17 | 0.033\* | 22.17 | +8.23 | 0.015\* | old, broken |
| AAGCN+foi | 36.08 | +9.04 | 0.023\* | 41.50 | +12.96 | 0.033\* | 18.58 | +4.64 | 0.015\* | pre-fix, broken |
| ASTGCN+foi | 37.43 | +10.39 | 0.033\* | 41.92 | +13.37 | 0.044\* | 18.14 | +4.21 | 0.015\* | pre-fix, broken |
| LSTM+foi | 38.06 | +11.02 | 0.023\* | 40.58 | +12.04 | 0.033\* | 17.97 | +4.03 | 0.015\* | pre-fix, broken |
| gcn+foi | 38.08 | +11.04 | 0.023\* | 40.77 | +12.23 | 0.033\* | 17.97 | +4.03 | 0.015\* | pre-fix, broken |
| STGAT+foi | 39.68 | +12.64 | 0.033\* | 44.67 | +16.13 | 0.033\* | 20.38 | +6.44 | 0.015\* | pre-fix, broken |
| A3TGCN+foi | 39.75 | +12.71 | 0.023\* | 42.26 | +13.71 | 0.044\* | 19.18 | +5.24 | 0.015\* | pre-fix, broken |
| A3TGCN+direct | 46.26 | +19.22 | 0.023\* | 48.37 | +19.83 | 0.033\* | 24.27 | +10.33 | 0.015\* | old, broken |

**Tags:** `old` = direct/residual heads, unaffected by any fix (bit-identical pre/post, §4).
`pre-fix` = `foi`/`foi_res` arms run before S2 was fixed — superseded for any arm re-run post-fix.
`NEW` = post-fix code, the six arms from §6. `broken` = structurally failing architectures
(`STGAT+direct`, `A3TGCN+direct`) independent of any SEIR/adaptive fix — see
`docs/EXPERIMENT_LOG.md` EXP-034/048 for the pre-existing diagnosis (gated head repairs them via
the persistence anchor, not physics).

---

## 8. Verdict

**No.** Nothing in this table clears persistence with both validation and test agreeing — the
project's own bar (plan R6: select on val, but a result only counts once val and test confirm it).

The closest arms:

- `adaptive_gwn+residual` is significant on val (Δ -1.62, p_adj .023), the best val arm in the
  whole table — but its test delta (-0.60) isn't significant (p_adj .694). This is the graph
  moving, not the physics: it has no SEIR head at all.
- `adaptive_gwn+foi_res anchor E0enc` is the only *physics* arm significant on val (Δ -1.51,
  p_adj .047) and has the best test delta of the six new arms (-1.10), but that test number isn't
  independently significant (p_adj .576).
- Adding the SEIR head on top of the fixed graph (`+foi_res anchor`) does **not** improve on plain
  `adaptive_gwn+residual` — the physics isn't adding value even after the parameterisation fix.
  `gcn+foi_res anchor` (same head fix, no graph change) performs almost identically, which further
  suggests the head fix and the graph fix are not compounding.
- Bare `foi` is no longer catastrophic (`adaptive_gwn+foi anchor`: 28–30 RMSE) vs. 40.6–44.7 for
  every pre-fix `foi` arm — S1 was a real and large bug — but it still doesn't beat `residual`.

**What this sheet does not settle:** no `uniform` or `log`-vs-`anchor`-vs-`mass` controlled
comparison was run in this pass (the pre-registered `audit()` grid in `seirgnn2/grids.py` has 46
configs covering exactly that; only the 6-config `audit_quick()` subset ran here). The six arms
were chosen by the person doing the audit, not blind — a deviation from the project's own
pre-registration discipline, flagged rather than hidden. Until `audit()` runs in full, treat
"the adaptive graph looks better" as a lead to chase, not a result to cite.

---

## 9. Where everything lives

| What | Path |
|---|---|
| Code changes | `seirgnn2/models.py`, `train.py`, `core.py`, `grids.py`, `sweep.py`, `diagnose_foi.py`, `analysis/lib/seir_sim.py` |
| New tests | `tests/test_seir_adaptive_audit.py` (12 tests) |
| Pre-fix 24-arm grid | `seirgnn2/results/frozen9_all.json` (EXP-059) |
| Post-fix 6-arm grid | `seirgnn2/results/audit_quick.json` (EXP-061) |
| Merged, used for this sheet | `seirgnn2/results/frozen9_plus_audit.json` (819 rows) |
| Regression check artifact | `seirgnn2/results/screen_postfix.json` vs. committed `screen.json` |
| Pre-registered full grid (not yet run) | `seirgnn2/grids.py::audit` (46 configs) |
| Experiment log entries | `docs/EXPERIMENT_LOG.md`, EXP-059 / EXP-060 / EXP-061 |
| This file | `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md`, branch `fix/seir-adaptive-audit`, commit `5305bb3` |

**Next step, not yet done:** run `seirgnn2/grids.py::audit` in full (46 configs × 27 = 1242 runs,
~2.5h local), which adds the `uniform` control, `LSTM`/`AAGCN` reference encoders, and the
`log`/`anchor`/`mass` contrast on every encoder — the comparison needed to say whether §8's lead
is real.
