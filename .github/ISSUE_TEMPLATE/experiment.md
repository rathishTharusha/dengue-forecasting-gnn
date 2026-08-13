---
name: Experiment
about: Propose a run that should settle a specific question
title: "[EXP] "
labels: experiment
---

## Question

<!-- What will this run tell us that we don't already know? One sentence. -->

## Hypothesis

<!-- What you expect, and why. Stating it up front stops post-hoc rationalization. -->

## Setup

- **Phase / contribution:** baseline / adaptive graph / physics loss / GAN
- **Notebook or script:**
- **Config changes vs. the Phase-1 baseline:**
- **Protocol:** rolling-origin, 3 origins × 3 seeds, W=3 → H=3 *(deviations must be justified — they break cross-row comparison)*

## Success criterion

<!-- A number, decided before the run. e.g. "overall RMSE < 44.8 on the same folds." -->

## Compute estimate

<!-- Colab T4, ~N minutes per fold. Flag anything that won't fit the free tier. -->

## Definition of done

- [ ] Run completed under the frozen protocol
- [ ] Entry added to `docs/EXPERIMENT_LOG.md`
- [ ] Metric CSV in `results/`
- [ ] Verdict recorded: answered / inconclusive / superseded
