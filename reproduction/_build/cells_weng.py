"""Cell sources for the Weng et al. (2024) exact reproduction.

The notebooks reimplement nothing. They clone the authors' repository at a pinned
commit, build the versions their ``requirements.txt`` names under a standalone
Python 3.11, and call their own ``run_<model>`` function with their own segment
list -- which is what their ``if __name__ == "__main__"`` block does. The only
contribution here is the comparison against ``Models/results.txt``.

``build_probe()`` emits a cheap kernel that does the environment setup and a
1-epoch smoke test and stops, so a broken environment costs minutes rather than
the hours a full sweep would.
"""

import env_setup
from gen_kernels import code, md

#: The five architectures with released code, in Table I order.
MODELS = ["STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN"]

#: Targets transcribed from Models/results.txt, the authors' own raw output,
#: which matches Table I to the decimal.
TARGETS = {
    "STGAT": (25.38192804812382, 1.3723462830059112, 44.7822111235735, 2.2630818940691015),
    "A3TGCN": (34.02748489379883, 1.093589425086975, 58.52167510986328, 1.8782962560653687),
    "ASTGCN": (33.67945502916908, 19.85381349077559, 47.72269138467796, 18.11540119078704),
    "DCRNN": (45.97544632682798, 2.3412923164313133, 71.60834584214274, 2.559600671175839),
    "AAGCN": (41.930203246294084, 1.1113400017649204, 55.83017146663277, 1.9815054977092335),
}

#: Table I, shifted dataset, Cross Validated column (paper, as printed).
TABLE_I = {
    "STGAT": (25.38, 1.37, 44.78, 2.26),
    "A3TGCN": (34.03, 1.10, 58.52, 1.87),
    "ASTGCN": (33.68, 19.85, 47.72, 18.12),
    "DCRNN": (45.98, 2.34, 71.61, 2.56),
    "AAGCN": (41.93, 1.11, 55.83, 1.98),
}

_DEVIATIONS = """
## Three documented deviations, all forced

**1. Python version.** Kaggle's image is Python 3.12. The authors pin
`torch==2.1.2`, which has no cp312 wheel — torch added 3.12 support in 2.2.0 — so
their stack cannot install on the stock image. Rather than bump torch and lose
fidelity, these kernels use `uv` to fetch a standalone **Python 3.11** and build
the authors' exact stack inside it.

**2. pandas.** `torch_geometric_temporal==0.54.0` declares `pandas<=1.3.5`,
contradicting the authors' own `pandas~=2.2.0`; pip answers
`ResolutionImpossible`. → PGT installed with `--no-deps`, honouring the authors'
pandas pin. Only PGT's stale transitive pin is ignored, which the authors had
already overridden.

**3. torch_geometric.** The pinned pair `torch_geometric==2.5.3` +
`torch_geometric_temporal==0.54.0` **cannot import**:

```
File ".../torch_geometric_temporal/nn/attention/tsagcn.py", line 6
    from torch_geometric.utils.to_dense_adj import to_dense_adj
ModuleNotFoundError: No module named 'torch_geometric.utils.to_dense_adj'
```

PyG moved that module in 2.4. Tested: 2.4.0 is the newest release where all five
architectures import. PGT stays at the authors' `0.54.0`, because PGT defines the
model architectures — changing it risks changing the models.

Everything else — `torch==2.1.2`, `numpy~=1.26.2`, `pandas~=2.2.0`,
`scikit_learn==1.4.0`, `statsmodels==0.14.1` — is exactly what the authors named.
All three deviations must be restated wherever these numbers are cited. See
`reproduction/REPRODUCIBILITY_MATRIX.md`.
"""


_CODE_WALKTHROUGH = """
## What the authors' code actually does

Read this before the results — it is what the numbers mean. Everything below is
in `Models/evaluation.py` and `Models/gnn_models.py`; nothing is our paraphrase
of intent, only a description of the code that runs.

### Data (`load_data`, `create_dataset_single`)

```python
x = np.nan_to_num(np.load(data_file, allow_pickle=True))
mean, std = np.mean(x[..., -6]), np.std(x[..., -6])      # whole series
x = z_norm(torch.tensor(x), mean, std)
x = x[:int(x.shape[0] * subset), ...]                     # truncate AFTER
```

The array is `(459 weeks, 25 districts, 11 features)`; `x[..., -6]` is index 5,
weekly cases. Note the ordering: statistics are computed over the **entire**
series, including the test period, and only then is the segment truncated.
`inverse_z_norm` reuses those same statistics when scoring.

Windows are built as `window_size=3` past weeks → `predict_ahead=3` future weeks.
For the GNNs the authors pass `use_disease_only=True`, so the models see **only
the case channel** — the ten meteorological covariates are not used.

### Graph

`load_adjacency_matrix` builds a 25×25 binary matrix from
`sri_lanka_adj_list.json` with self-loops, giving 141 directed edges. Districts
are ordered by `sorted()` of the JSON keys.

### Training (`train_model`, `train_single_shot`, ...)

Adam at `lr=1e-4`, `weight_decay=5e-5`, `MSELoss`, `num_epochs=50`,
`batch_size=1`, no early stopping and no validation-based model selection — the
final epoch's weights are the ones scored. Module-level `torch.manual_seed(0)`,
`random.seed(0)` and `np.random.seed(0)` make it deterministic.

### Evaluation (`infer`) — the part that decides what Table I means

```python
rmse += RMSE(truth, pred)
mae  += MAE(truth, pred)
...
rmse /= n
```

Metrics accumulate **per batch** and are divided by the batch count. With
`batch_size=1` that is the mean of per-window RMSEs, not the RMSE of the pooled
predictions — the two differ substantially on a heavy-tailed target.

And in every `run_*`:

```python
y_pred, y_truth, _, _  = infer(model, "cpu", test, m, s, "Test")   # discarded
y_pred, y_truth, ma, rm = infer(model, "cpu", full, m, s, "Full")  # appended
maes += [ma]; rmses += [rm]
```

The held-out test score is computed and thrown away. The number that becomes
Table I's "Cross Validated" column comes from `full` — every window in the
segment, including the 70% the model trained on.

### Cross-validation

`run_*([0.6, 0.7, 0.8, 0.9, 1.0])` reinitializes the model per segment, trains on
the first 70% of that prefix, and reports mean ± population std across the five
segments.

**This notebook changes none of that.** It reproduces the numbers as the authors
produce them. Whether that protocol supports the paper's conclusions is a
separate question, examined in `crosscheck/FINDINGS.md`.
"""


def _env_cells() -> list:
    """The shared bootstrap: preamble, Python 3.11 stack, verification, clone."""
    return [
        md("## 1. Environment — the authors' pins, under a standalone Python 3.11"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. The authors' repository and data, verified by checksum"),
        code(env_setup.CLONE),
    ]


def build_probe() -> list:
    """A cheap kernel: build the environment, smoke-test, stop.

    Exists because the first full push failed on an environment error after
    queueing. Validating the environment separately costs minutes.
    """
    return [
        md(
            f"""
# Environment probe — Weng et al. (2024) reproduction

Builds the exact stack the five model kernels use and runs **one epoch on one
segment** for each of the authors' five architectures, then stops.

**The numbers this prints are meaningless.** The only question it answers is
whether the authors' code executes in this environment — worth answering in
minutes rather than discovering hours into a real sweep.

Run this first. When every architecture reports OK, push the model kernels.
{_DEVIATIONS}
"""
        ),
        md(_CODE_WALKTHROUGH),
        *_env_cells(),
        md(
            """
## 3. Smoke test — one epoch, one segment, all five architectures

`evaluation.num_epochs` is overridden to 1 here and **only** here. The model
kernels leave the authors' `num_epochs = 50` untouched.
"""
        ),
        code(
            '''
results = {}
for runner in ["run_stgat", "run_a3tgcn", "run_astgcn", "run_dcrnn", "run_aagcn"]:
    script = (
        "import sys\\n"
        "sys.path.insert(0, '.')\\n"
        "import evaluation\\n"
        "evaluation.num_epochs = 1        # smoke test only, NOT a reproduction\\n"
        f"evaluation.{runner}([0.6])\\n"
    )
    started = time.time()
    r = sh(str(PY311), "-u", "-c", script, cwd=str(SRC / "Models"), check=False, quiet=True)
    ok = r.returncode == 0
    if ok:
        lines = [ln for ln in r.stdout.splitlines() if "Average" in ln]
        detail = lines[-2] if len(lines) >= 2 else ""
    else:
        detail = (r.stdout.strip().splitlines() or ["?"])[-1][:160]
    results[runner] = ok
    print(f"{runner:12s} {'OK  ' if ok else 'FAIL'} {time.time() - started:6.1f}s  {detail}")

print()
if all(results.values()):
    print("ALL FIVE ARCHITECTURES RUN. Safe to push the model kernels.")
else:
    print("BROKEN:", [k for k, v in results.items() if not v])
    raise SystemExit("environment probe failed")
'''
        ),
    ]


def build(model: str) -> list:
    """Cells for one architecture's full reproduction kernel."""
    mae, mae_sd, rmse, rmse_sd = TARGETS[model]
    p_mae, p_mae_sd, p_rmse, p_rmse_sd = TABLE_I[model]
    runner = f"run_{model.lower()}"

    return [
        md(
            f"""
# Exact reproduction — Weng et al. (2024), **{model}**

**Paper:** Jiaqi Weng, David Qiu, Ethan Cruz, Malik Magdon-Ismail, Thilanka
Munasinghe, Jennifer C. Wei, Ashan Pathirana, *Graph Representation Learning for
Dengue Forecasting*, IEEE BigData 2024.

**Code:** [`MLOpenSourceOpenScience/disease_modeling_MLOS2`](https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2)
at `{env_setup.COMMIT[:7]}`.

A reproduction, not a reimplementation: it calls **their** `{runner}()` with
**their** segment list `[0.6, 0.7, 0.8, 0.9, 1.0]`, which is what their
`if __name__ == "__main__"` block does. No model code, training loop, metric or
data handling here is ours.

## Target

`Models/results.txt` is the authors' own raw output. For {model}:

| Source | MAE | RMSE |
|---|---|---|
| `results.txt` (raw) | {mae:.14f} ± {mae_sd:.14f} | {rmse:.14f} ± {rmse_sd:.14f} |
| Table I (as printed) | {p_mae} ± {p_mae_sd} | {p_rmse} ± {p_rmse_sd} |

Run the **environment probe** kernel before this one.
{_DEVIATIONS}
"""
        ),
        md(_CODE_WALKTHROUGH),
        *_env_cells(),
        md(
            """
## 3. Run the authors' code

`evaluation.py` seeds `torch`, `random` and `numpy` to 0 at import, so the run is
deterministic. Its hyperparameters are printed from their module rather than
restated here.

This is the long cell: 5 segments × 50 epochs on CPU. Expect 1–4 hours.
"""
        ),
        code(
            f'''
RUNNER = "{runner}"
MODEL = "{model}"

script = (
    "import sys\\n"
    "sys.path.insert(0, '.')\\n"
    "import evaluation\\n"
    "print(\\"authors' hyperparameters, read from their module:\\")\\n"
    "for _n in ('batch_size','window_size','in_channels','out_channels',"
    "'num_nodes','num_epochs','lr','decay','dropout','data_file'):\\n"
    "    print('  %-12s = %s' % (_n, getattr(evaluation, _n)))\\n"
    f"evaluation.{runner}({{SEGMENTS}})\\n"
)

started = time.time()
proc = subprocess.Popen(
    [str(PY311), "-u", "-c", script], cwd=str(SRC / "Models"),
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
)
lines = []
for line in proc.stdout:            # stream, so progress shows in the Kaggle log
    lines.append(line)
    if any(k in line for k in ("Average", "= ", "Test,", "Full,")):
        print(line, end="")
proc.wait()
elapsed = time.time() - started
captured = "".join(lines)

(WORK / (MODEL + "_stdout.txt")).write_text(captured, encoding="utf-8")
print(f"\\n{{RUNNER}} exited {{proc.returncode}} after {{elapsed / 60:.1f}} min")
if proc.returncode != 0:
    print(captured[-3000:])
    raise SystemExit("the authors' runner failed")
'''
        ),
        md("## 4. Compare against the authors' own results"),
        code(
            f"""
TARGET = dict(mae={mae!r}, mae_sd={mae_sd!r}, rmse={rmse!r}, rmse_sd={rmse_sd!r})
PAPER = dict(mae={p_mae!r}, rmse={p_rmse!r})
print(TARGET)
print(PAPER)
"""
        ),
        code(
            '''
# Their run_* functions end with two lines:
#   Average MAE: <x>, Standard Deviation: <y>
#   Average RMSE: <x>, Standard Deviation: <y>
def parse(text, metric):
    hits = re.findall(
        r"Average " + metric + r": ([\\d.eE+-]+), Standard Deviation: ([\\d.eE+-]+)", text
    )
    if not hits:
        raise ValueError("could not find 'Average " + metric + "' in the captured output")
    return float(hits[-1][0]), float(hits[-1][1])


got_mae, got_mae_sd = parse(captured, "MAE")
got_rmse, got_rmse_sd = parse(captured, "RMSE")

print("metric      this run          results.txt         abs diff     rel %    Table I")
for name, got, target, paper in [
    ("MAE", got_mae, TARGET["mae"], PAPER["mae"]),
    ("RMSE", got_rmse, TARGET["rmse"], PAPER["rmse"]),
]:
    diff = got - target
    print(f"{name:6s} {got:18.12f} {target:18.12f} {diff:12.2e} "
          f"{100 * diff / target:8.4f} {paper:9.2f}")

print()
print("sd     this run          results.txt")
print(f"{'MAE':6s} {got_mae_sd:18.12f} {TARGET['mae_sd']:18.12f}")
print(f"{'RMSE':6s} {got_rmse_sd:18.12f} {TARGET['rmse_sd']:18.12f}")
'''
        ),
        code(
            '''
import csv

row = {
    "model": MODEL, "runner": RUNNER, "segments": json.dumps(SEGMENTS),
    "commit": COMMIT, **{f"env_{k}": v for k, v in versions.items()},
    "minutes": round(elapsed / 60, 2),
    "mae": got_mae, "mae_sd": got_mae_sd, "rmse": got_rmse, "rmse_sd": got_rmse_sd,
    "target_mae": TARGET["mae"], "target_mae_sd": TARGET["mae_sd"],
    "target_rmse": TARGET["rmse"], "target_rmse_sd": TARGET["rmse_sd"],
    "paper_mae": PAPER["mae"], "paper_rmse": PAPER["rmse"],
    "mae_rel_error_pct": 100 * (got_mae - TARGET["mae"]) / TARGET["mae"],
    "rmse_rel_error_pct": 100 * (got_rmse - TARGET["rmse"]) / TARGET["rmse"],
}
out = WORK / ("weng2024_" + MODEL + "_reproduction.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
print("wrote", out)
for k, v in row.items():
    print(f"  {k:22s} {v}")
'''
        ),
        md(
            f"""
## 5. Reading the result

`evaluation.py` seeds all three RNGs to 0, so an exact-environment rerun should
land close to `results.txt`:

- **< 0.01%** — bit-level reproduction.
- **< 5%** — reproduced. Residual drift is expected: a different CPU, BLAS build
  and thread count from the authors' AMD Ryzen 5 PRO 4650U, and floating-point
  reduction order is not fixed by a seed.
- **> 5%** — investigate before citing. Check the version assertions in §1 and
  the checksums in §2 first.

{model}'s reported spread across segments is itself part of the target:
±{rmse_sd:.2f} RMSE.

**What this does not establish.** Only the five GNN rows on the shifted dataset
are reproducible at all — ARIMA, Random Forest and LSTM read a `MLSO2_Final.csv`
that is absent from the repository, XGBoost and ARNN have no script, and no
unshifted array is released. See `reproduction/REPRODUCIBILITY_MATRIX.md`.

Reproducing these numbers confirms the authors ran what they said they ran. It
does **not** validate the protocol behind them — `crosscheck/FINDINGS.md`
F1.1–F1.6 documents five issues with how Table I is computed, including that its
cross-validated column is measured largely on training data, and that naive
persistence beats every model in it.
"""
        ),
    ]
