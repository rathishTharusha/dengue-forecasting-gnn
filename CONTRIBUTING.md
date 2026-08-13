# Contributing

Six people, one small dataset, and one ablation table that has to be internally consistent.
These conventions exist so the final results section is defensible.

## Branching

`main` is protected and always runnable. Work happens on branches:

| Prefix | Use for | Example |
|---|---|---|
| `feat/` | new model component or code | `feat/adaptive-adjacency` |
| `exp/` | an experiment / sweep, results included | `exp/physics-loss-lambda-sweep` |
| `fix/` | bug in existing code | `fix/leaky-normalization` |
| `docs/` | docs, report text, references | `docs/related-work-section` |

```bash
git checkout main && git pull
git checkout -b feat/adaptive-adjacency
```

Open a PR into `main` when done. At least one teammate reviews. Squash-merge to keep history
readable.

## Commits

Conventional-commit style, imperative mood:

```
feat(model): add learned adaptive adjacency to GNN baseline
fix(data): use train-fold statistics only for z-normalization
exp(physics): sweep lambda_phys over {0.01, 0.1, 1.0}
docs(readme): record Phase-1 rolling-origin results
```

Scopes we use: `data`, `model`, `train`, `eval`, `physics`, `gan`, `graph`, `docs`, `ci`.

## Notebook hygiene

Notebooks are the primary workspace here, and they diff badly. Rules:

1. **One owner per notebook at a time.** Say so in the group chat before you start editing.
   Two people editing the same `.ipynb` produces a merge conflict you cannot resolve by hand.
2. **Restart & Run All before committing.** A notebook with out-of-order execution counts is
   not evidence of anything.
3. **Commit outputs for result notebooks** (they *are* the deliverable), but clear outputs on
   scratch/exploration notebooks — `jupyter nbconvert --clear-output --inplace nb.ipynb`.
4. **Do not paste `QUICK_TEST = True` numbers anywhere.** They are intentionally degraded.
5. Code that stabilizes and gets used twice moves out of the notebook into `src/dengue_gnn/`.

## Experiments

Every run whose numbers might end up in the report goes in
[`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md), with:

- the exact `CFG` (or a link to the committed config),
- git commit SHA of the code that produced it,
- seeds and fold definitions,
- the metric table, unrounded.

Metric tables go in `results/` as CSV, not only as a screenshot in a notebook.

**Protocol is frozen.** Rolling-origin CV, 3 chronological origins, 3 seeds, horizons 1–3,
metrics RMSE / MAE / SMAPE / masked MAPE (+ CRPS, PICP/MPIW, Moran's I once we produce
distributions). If you have a good reason to change it, change it *for every row of the
ablation table* and re-run the baseline — otherwise the comparison is invalid.

## Code

- Python 3.11, formatted and linted with `ruff`.
- `ruff check src tests tools && ruff format --check src tests tools` must pass — CI runs it.
- `pytest` for anything in `src/`. Metrics and windowing logic especially: a silent off-by-one
  in the horizon slicing invalidates every number downstream.

```bash
ruff check src tests tools
pytest -q
```

## Architecture decisions

Non-obvious modelling choices get a short ADR in [`docs/decisions/`](docs/decisions/) —
copy `docs/decisions/TEMPLATE.md`, number it sequentially. "Why residual-over-persistence?" is
exactly the kind of question a reviewer will ask, and `0001` is the answer.

## Before pushing

- [ ] Notebook restarted and run top-to-bottom (if you touched one)
- [ ] `ruff check` and `pytest` pass
- [ ] No data/checkpoints/secrets added — `git status` is clean of anything in `.gitignore`
- [ ] Experiment logged in `docs/EXPERIMENT_LOG.md` if it produced numbers
- [ ] `README.md` status table updated if you completed a phase
