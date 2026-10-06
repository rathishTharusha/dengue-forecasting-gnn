# Repository map (audit for the IEEE paper)

Audit date 2026-10-05. `git fetch --all --prune` run first. Nothing was checked out, merged, reset or pushed on existing branches. Only the new branch `paper/ieee-draft` was created, from **`origin/main`** (2800651).

**Important:** the local `main` (491d2f5, 2026-08-14) is **167 commits behind** `origin/main` (2026-10-05). It is stale and was not used. Every "ahead/behind" below is measured against `origin/main`.

**Existing paper material already in the repo (not touched):**
- `paper/` holds the Phase-2 short paper in ACM `sigconf` format ("Know When the Epidemic Comes", legacy benchmark array), including a `paper/refs.bib`. New files in this audit sit beside it.
- `full_paper/` holds an 8-page paper built from the Kaggle reproduction run (EXP-050): "Where Does Physics Help a Graph Network? A Leakage-Controlled Study of SEIR-Informed Spatio-Temporal GNNs for Dengue Forecasting", with `full_paper/overleaf/refs.bib`.
- `short_paper/` holds an overleaf copy of the short paper.

## Summary

- 22 remote branches (`origin/*`, excluding `origin/HEAD`), 4 local branches before this work (`main`, `praveen`, `praveen-02`, `fix/s8-seroprevalence`), plus the new `paper/ieee-draft`.
- **18 of 22 remote branches are fully merged into `origin/main`** (0 commits ahead). All the SEIR-GNN and corrected-data work is on main.
- **4 branches have content not on main:** `praveen` (local tip 13 ahead, remote tip 8 ahead), `origin/adaptive-graph` (7 ahead), `origin/Maleesha-Dev` (1 ahead), `praveen-02` (1 ahead, a `.gitignore` edit only).
- The unmerged branches hold the **original proposal-era Phase 2/3/4 work** (adaptive graph, spatial/physics loss, SEIR-SEI Phase 3, climate-informed force of infection proposal, WGAN-GP augmentation) on the **legacy benchmark array**. Main later removed that code (`c0b58e5`, tag `phase23-archive`) but kept the logged numbers (EXP-001 to EXP-014).

## All branches

Dates are last-commit dates. "Merged" = ancestor of `origin/main`. Purposes come from branch names and last-commit subjects; merged branches were not read file by file because their content is on main and was read there.

| Branch | Last commit | Author | Ahead / behind origin/main | Merged | Purpose |
|---|---|---|---|---|---|
| origin/main | 2026-10-05 | Janith Mahanama (merge of PR #12) | 0 / 0 | yes | Integration branch. Holds EXP-001 to EXP-061, `seirgnn2/`, `analysis/`, `full_paper/`. |
| main (local) | 2026-08-14 | Tharusha Perera | 0 / 167 | yes (stale) | Old README/contributing commit. **Stale, ignore.** |
| origin/fix/seir-adaptive-audit | 2026-10-04 | Jaybro-git | 0 / 1 | yes | Audit of SEIR head and adaptive graph; EXP-059/060/061; `docs/SEIR_ADAPTIVE_AUDIT_RESULTS.md`. Newest experiments. |
| origin/fix/s8-seroprevalence, fix/s8-seroprevalence (local) | 2026-10-03 | Praveen De Silva | 0 / 7 | yes | Stage S8 rerun against the real seroprevalence survey (EXP-058). |
| origin/phase2/review-and-fixes | 2026-10-03 | Praveen De Silva | 0 / 146 | yes | Phase-2 review and fixes, merged via PR #9. |
| origin/docs/full-paper-rewrite | 2026-09-25 | rathishTharusha | 0 / 36 | yes | One notebook that reproduces every finding; paper corrections. |
| origin/feat/kaggle-reproduction | 2026-09-25 | rathishTharusha | 0 / 34 | yes | Full-paper reproduction on Kaggle (EXP-050, 810 runs). |
| origin/exp/rescue-control | 2026-09-24 | rathishTharusha | 0 / 41 | yes | EXP-048: SEIR head's repair is the anchor, not the physics. |
| origin/exp/physics-gnn | 2026-09-24 | rathishTharusha | 0 / 45 | yes | EXP-046/047: physics-informed GNN screen and nine-origin check. |
| origin/exp/arch-improvements | 2026-09-23 | rathishTharusha | 0 / 50 | yes | EXP-045: architecture changes not adopted. |
| origin/exp/climate-lags | 2026-09-23 | rathishTharusha | 0 / 54 | yes | EXP-043/044: longer climate lags do not help. |
| origin/exp/gan-augmentation | 2026-09-23 | rathishTharusha | 0 / 58 | yes | EXP-042: GAN (TimeGAN) and simulator augmentation, not adopted. |
| origin/exp/literature-remedies | 2026-09-23 | rathishTharusha | 0 / 62 | yes | EXP-040: literature remedies, none adopted. |
| origin/exp/seir-gnn-v2 | 2026-09-23 | rathishTharusha | 0 / 67 | yes | `seirgnn2/` harness (leakage-controlled re-run). |
| origin/exp/seir-gnn | 2026-09-21 | rathishTharusha | 0 / 81 | yes | First SEIR-GNN stages S4 to S9 (defective S5, see EVIDENCE). |
| origin/feat/seir-gnn | 2026-09-14 | rathishTharusha | 0 / 99 | yes | Data rebuild ("553 WER reports match official PDFs"). |
| origin/feat/physics-informed-loss | 2026-09-11 | rathishTharusha | 0 / 133 | yes | Paper figures regenerated; physics-loss work. |
| origin/exp/beat-baseline | 2026-09-11 | rathishTharusha | 0 / 105 | yes | Attempts to beat the persistence floor, ensembles. |
| origin/feat/reproduction-and-eda | 2026-09-10 | rathishTharusha | 0 / 147 | yes | Reproduction of Weng et al., EDA, crosscheck workspace. |
| **praveen** (local) | 2026-09-10 | Praveen De Silva | **13 / 167** | **no** | Phase 2 + Phase 3 work: adaptive graph, physics loss, SEIR-SEI Phase 3, ADR 0003 to 0005 (climate-informed force of infection, proposed and then countered), raw CSVs for EXP-001 to EXP-005. 5 commits (Phase 3 + ADR 0005) exist only locally and on no remote. |
| **origin/praveen** | 2026-09-09 | Praveen De Silva | 8 / 167 | no | Same line, up to the data manifest and baselines (before Phase 3 commits). |
| **origin/adaptive-graph** | 2026-09-09 | Jaybro-git | 7 / 167 | no | Phase 2 rolling-origin harness, adaptive-graph GCN, **removal of two fabricated adjacency edges**, corrected-adjacency ablation (`results/phase2_runs.csv`). |
| **origin/Maleesha-Dev** | 2026-09-11 | Maleesha | 1 / 167 | no | "ran the phases": Phase 1 to 5 scripts, **WGAN-GP augmentation** (`gan.py`, `phase4_gan_results.csv`), master ablation table. Legacy array. |
| praveen-02 / origin/praveen-02 | 2026-10-03 | Praveen De Silva | 1 / 9 | no | One commit removing two `.gitignore` lines. No research content. |
| paper/ieee-draft (new) | now | this audit | from origin/main | n/a | This work. |

## What is only on unmerged branches (relevant to the paper)

| Content | Branch | Path | Protocol |
|---|---|---|---|
| Raw per-fold CSVs for EXP-003/004/005 (adaptive graph, physics-loss weight sweep, combined model) | `praveen` | `results/exp003_adaptive_gcn.csv`, `exp004_lambda_sweep.csv`, `exp005_combined_proposed.csv`, `baseline_rolling_origin.csv` | Legacy array, 3 folds (0.55/0.70/0.85), 3 seeds, window 3 to horizon 3 |
| Corrected-adjacency ablation | `praveen`, `origin/adaptive-graph` | `results/phase2_runs.csv` (192 rows) | Same, adjacency without two fabricated edges |
| Phase 3 SEIR-SEI GNN, Stage C negative result, ADR 0003/0004/0005 | `praveen` | `results/phase3_stageC.json`, `docs/decisions/0003..0005` | Legacy array, origins 0.70 and 0.85, 40 epochs |
| Climate-informed force of infection | `praveen` | ADR 0005, `scripts/smoke_phase3_climate.py`, `results/phase3_climate_smoke_*.json` | **Smoke test only. ADR status "Proposed, evidence against".** |
| WGAN-GP augmentation, master ablation (adaptive + physics + augmentation) | `origin/Maleesha-Dev` | `results/phase4_gan_results.csv`, `results/master_ablation_table.csv` | Legacy array, 3 folds, 3 seeds |

## Finding that affects every main-branch number

`origin/main` still uses the adjacency file with the two edges (Kandy to Ampara, Kegalle to Kalutara) that the `adaptive-graph` / `praveen` branches found to be one-directional and not real borders, verified against GADM 4.1 polygons (commit 6273b45). `analysis/lib/corrected_data.py` loads `notebooks/baseline/sri_lanka_adj_list.json`, so all `seirgnn2/` and EXP-050 graph results use the 141-edge version. The corrected file (139 edges) exists only on the unmerged branches. See question Q2 in PAPER_PLAN.md.
