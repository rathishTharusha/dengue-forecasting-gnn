# Prompt to start the SEIR-GNN experiments in Antigravity

Open `C:\Users\tharu\dev\dengue-seir-gnn` in Antigravity. Make sure `feat/seir-gnn`
is pushed first. Then paste everything inside the block below.

---

```text
You are running the SEIR-GNN experiments for a university research project on district-level weekly dengue forecasting in Sri Lanka (CS3631 Group 05). The deliverable is a research report with an honest, internally consistent results table. A well-run negative result is a valid outcome. Never present a result as better than it is.

WORKSPACE
- Repository: C:\Users\tharu\dev\dengue-seir-gnn (a git worktree, NOT the Google Drive folder).
- Start by creating branch exp/seir-gnn from feat/seir-gnn: `git switch -c exp/seir-gnn feat/seir-gnn`.
- Python 3.11. Run `pytest` and `ruff check` on files you change.

READ THESE FIRST, IN FULL, BEFORE WRITING ANY CODE
1. docs/SEIR_GNN_EXPERIMENT_PLAN.md: the pre-registered plan. It is your task list. Follow it stage by stage.
2. docs/DATA_PROVENANCE.md: where every dataset comes from and the causality rules.
3. docs/ARRAY_AUDIT.md: why the original array's climate channels must never be used.
4. CLAUDE.md: repo conventions, frozen protocol, results logging.
5. analysis/lib/corrected_data.py and tests/test_no_future_leakage.py: the only permitted data path.
6. analysis/lib/seir_sim.py and tests/test_seir_sim.py: the SEIR integrator to use.
7. analysis/_build/run_corrected_benchmark.py, analysis/_build/run_physics_experiments.py (run_fold), analysis/lib/reproduced.py (build): the existing training loop and GNN encoders.
8. reproduction/_build/cells_corrected.py and reproduction/_build/gen_kernels.py: how Kaggle kernels are generated.

HARD RULES (violating any of these invalidates the work)
1. NO FUTURE INFORMATION. A forecast at the start of week i may read cases up to i-1, ERA5 climate up to i-2, and the as-of NDVI and previous-year population rows for week i. Never use a later value, and never pick a lag because it correlates well, however strong the correlation. All inputs go through corrected_data.load()/windows(). If you add an input or a new slicing path, first add a test to tests/test_no_future_leakage.py that perturbs data at or after the limit and asserts the inputs do not change.
2. NO GENERATED DATA. Do not interpolate, impute, forward-fill, augment or synthesise values into training, validation or test data. Weeks with no source report stay NaN, and windows touching them are dropped. The only exception is the S3 twin experiment: a code test on simulated data, stored under analysis/results/seir_gnn/s3_twin/ and labelled synthetic, never mixed with real data.
3. NEVER use the climate/NDVI channels of notebooks/baseline/sri_lanka_2013-2022_shifted.npy (misaligned and future-shifted).
4. HINDSIGHT-ONLY KNOWLEDGE: the 2022-23 seroprevalence survey is validation only. Serotype-switch dates may only appear in arms named oracle_*, which are never finalists.
5. SELECTION ON VALIDATION ONLY. Choose hyper-parameters, formulations and finalists by validation RMSE. Commit the frozen config BEFORE comparing test numbers.
6. FROZEN PROTOCOL on dataset `rebuilt`: origins 0.55/0.70/0.85, 3 seeds, window 3 -> horizon 3, train-only normalisation, early stopping on validation, pooled RMSE/MAE/SMAPE/MAPE(>=1), overall and per horizon. Metrics come from src/dengue_gnn/metrics.py; do not re-implement scoring. The confirmatory test uses 9 disjoint origins, as the plan says.
7. GIT: commit with conventional commits (scopes data, model, train, eval, physics, docs). NEVER push. NEVER commit to main. NEVER commit papers/, datasets/, reference_repo/, data/raw/, or any credential. When a Kaggle kernel needs new code, commit, then stop and ask me to push.
8. COMPUTE: anything expected to take more than ~15 minutes runs as a Kaggle kernel, generated the way cells_corrected.py does it, with BRANCH = "exp/seir-gnn". The Kaggle CLI is already authenticated from ~/.kaggle/. Never print, copy or commit its contents. After generating kernels, run `python reproduction/kaggle/setup_kaggle.py --init-metadata` from the repo root and revert unrelated kernel churn.
9. RECORDS: every run writes a CSV to analysis/results/seir_gnn/<stage>/ with one row per (origin, seed, horizon), not pre-aggregated, plus an append-only entry at the top of docs/EXPERIMENT_LOG.md with commit SHA, exact config, seeds, origins and the unrounded table. No orphan numbers.
10. HONESTY: report failures, crashes, skipped steps and negative results as they are. Do not rerun with new seeds or tweak settings after seeing test numbers to get a better result. If you must deviate from the plan, log the deviation and why.

HOW TO WORK
- Do the stages in order: S0, S1, S2, ... S9 from docs/SEIR_GNN_EXPERIMENT_PLAN.md.
- At each gate (G0, G1, G3, G4, G5) and at every stage marked "Stop and report after? yes", STOP. Give me:
  (a) what you ran (commands, kernels, commit SHA);
  (b) the results table with unrounded numbers;
  (c) whether the gate passed;
  (d) anything surprising or broken;
  (e) exactly what you need from me (e.g. "push exp/seir-gnn, then say continue").
  Wait for me to say "continue".
- Put new library code in analysis/lib/ (e.g. seir_gnn.py) with unit tests in tests/. Put runners in analysis/_build/ (e.g. run_seir_gnn.py) with a --quick smoke-test flag. Run --quick locally before any Kaggle submission.
- Keep the SEIR layer numerically safe: states are fractions in [0, 1], use seir_sim's exponential-flow integration with 7 substeps, and assert mass conservation in tests.
- If something in the plan is ambiguous or seems wrong, say so at the next stop instead of silently choosing.

START NOW WITH S0:
1. Create the branch.
2. Run `pytest`, `python analysis/_build/source_manifest.py --check`, and summarise analysis/results/source_verification/wer_all_verification.json.
3. Report G0 and stop.
```
