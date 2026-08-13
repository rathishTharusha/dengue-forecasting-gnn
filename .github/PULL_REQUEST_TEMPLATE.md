## What & why

<!-- One paragraph. What changed, and what question it answers or problem it solves. -->

Related: <!-- issue #, ADR, or EXPERIMENT_LOG entry -->

## Type

- [ ] `feat` — new model component or code
- [ ] `exp` — experiment / sweep with results
- [ ] `fix` — bug in existing code
- [ ] `docs` — documentation or report text

## If this produced numbers

- [ ] Logged in `docs/EXPERIMENT_LOG.md` (config + commit SHA + seeds + folds)
- [ ] Metric CSV committed under `results/`
- [ ] Ran under the **frozen protocol** — rolling-origin CV, 3 origins × 3 seeds, horizons 1–3
- [ ] Not `QUICK_TEST` numbers

**Headline result:**

| Config | RMSE | MAE |
|---|---|---|
| | | |

Persistence floor is RMSE 44.8 / MAE 15.7. Say plainly whether this beats it.

## Checklist

- [ ] Notebook restarted and run top-to-bottom (if touched)
- [ ] `ruff check src tests tools` and `pytest` pass
- [ ] No data, checkpoints, or secrets added
- [ ] ADR added under `docs/decisions/` if a non-obvious choice was made
- [ ] `README.md` status table updated if a phase completed
