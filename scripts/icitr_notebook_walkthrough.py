# Notebook cells deliberately share globals and introduce imports beside their explanations.
# Mathematical Unicode in Markdown is intentional.
# ruff: noqa: E402, F401, F821, I001, RUF003
# %% [markdown]
# # How the ICITR paper's results were obtained
#
# **Beyond Persistence? PAGE: Persistence-Anchored Gated Forecasting with
# Spatio-Temporal Graph Networks for Dengue**
#
# This is one local notebook for the latest paper, `full_paper/icitr/main.pdf`.
# It connects **data sources → weekly data → time splits → models → predictions →
# error scores → paper tables**. Each section explains what its code does before
# showing the calculation. Lower RMSE means fewer large forecast errors.
#
# ## Start here
#
# 1. Keep this notebook inside this project and open it in Jupyter or VS Code.
# 2. Select a Python environment containing `numpy`, `pandas`, `matplotlib`,
#    `torch` and `IPython`. Optional full training also needs `scikit-learn` and
#    `pymupdf`. No package installation or download happens automatically.
# 3. Choose **Run All**. The default checks existing evidence and prints the tables;
#    it does **not** train hundreds of networks.
# 4. To repeat training, read sections 13–14 and change `FULL_RETRAIN` below.
#    Fresh files go under `runs/icitr_notebook/`; paper files are never overwritten.
#
# **What a successful default run proves:** the calculations agree with the saved
# artifacts, and the available case data independently reproduce the naive baselines.
# **What it does not prove:** that every historical model was actually trained correctly,
# that local artifacts are independently authenticated, or that every cited paper is
# accurate. Logs, hashes and varying run times are evidence, not proof of execution.
# The original per-prediction archive for EXP-050 is absent from this checkout.
#
# ## Contents
#
# 1. Settings and evidence inventory
# 2. Where the data came from
# 3. Weekly data and missing weeks
# 4. Why there are different RMSE numbers: time splits and persistence
# 5. What PAGE changes
# 6. Table I: encoder and baseline comparison
# 7. Table II: ablation and paired comparisons
# 8. Other levers: loss, climate, graph and augmentation
# 9. Diagnostics: what can be recalculated and what is retained evidence
# 10. Table III: the amended 2024–2026 evaluation and its statistics
# 11. Tuning and Table IV: parameter counts and computational cost
# 12. Paper-number checks, coverage and limits
# 13. Optional: rerun the prospective training locally
# 14. Optional: rebuild inputs and rerun the 810-run development study
#
# The notebook intentionally reports the current paper, including known limitations.
# **EXP-062 is still marked pending in the paper:** date-joined seasonal and climate
# results need a recheck. This notebook does not silently replace those historical scores.

# %%
from pathlib import Path
import ast
import csv
import hashlib
import itertools
import json
import platform
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

FULL_RETRAIN = False
WORKERS = 1  # Reliable on Windows; the original EXP-050 notebook uses fork on Linux.
REPO_OVERRIDE = None  # Optional: r"C:\Users\ASUS\Desktop\dengue-forecasting"


def find_repo():
    if REPO_OVERRIDE:
        return Path(REPO_OVERRIDE).expanduser().resolve()
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "full_paper/icitr/main.tex").exists():
            return candidate
    raise FileNotFoundError(
        "Open this notebook inside the dengue-forecasting project, or set REPO_OVERRIDE."
    )


REPO = find_repo()
RUN_DIR = REPO / "runs/icitr_notebook"
RUN_DIR.mkdir(parents=True, exist_ok=True)
KAGGLE = REPO / "full_paper/outputs/kaggle_run"
PROSPECTIVE = REPO / "seirgnn2/results"
ENCODERS = ["AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN"]
CHECKS = []
pd.set_option("display.max_columns", 14)
pd.set_option("display.max_rows", 40)
pd.set_option("display.float_format", lambda value: f"{value:.4f}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_close(label, actual, expected, source, atol=1e-7):
    same_shape = np.asarray(actual).shape == np.asarray(expected).shape
    ok = same_shape and bool(np.allclose(actual, expected, atol=atol, rtol=0, equal_nan=True))
    CHECKS.append(
        {
            "check": label,
            "status": "PASS" if ok else "MISMATCH",
            "actual": str(actual),
            "reference": str(expected),
            "source": source,
        }
    )
    return ok


print("Project:", REPO)
print("Fresh outputs:", RUN_DIR)
print("Python:", platform.python_version(), "| platform:", platform.platform())
print(
    "Mode:",
    "historical audit + optional fresh training"
    if FULL_RETRAIN
    else "saved-result audit (no training)",
)

# %%
# SNAPSHOT_INSERTION_POINT

# %% [markdown]
# ## 1. Settings and evidence inventory
#
# The paper combines two experiments. **EXP-050** ran on Kaggle and compared many
# encoders on development weeks. **EXP-063** tuned adaptive models on development
# data, then tested them on later weeks. Those runs share neither all test weeks nor
# all model configurations. A result row is a score from one run, not one dengue case.
#
# `runs.jsonl` contains training records. `runs.json` additionally contains persistence
# and ensemble rows. Counting them separately prevents calling every row a trained NN.
# We record current file hashes for traceability. A hash detects a later byte change;
# it does not establish that an experiment is real or that the first download was authentic.

# %%
evidence_paths = [
    KAGGLE / "runs.json",
    KAGGLE / "runs.jsonl",
    KAGGLE / "notebook_output.txt",
    KAGGLE / "levers.csv",
    KAGGLE / "rescue.csv",
    KAGGLE / "results.json",
    KAGGLE / "residual_recovery.csv",
    KAGGLE / "residual_correlation.csv",
    KAGGLE / "seir_decoder_diagnosis.csv",
    PROSPECTIVE / "prospective_dev.json",
    PROSPECTIVE / "prospective_dev_classical.json",
    PROSPECTIVE / "prospective_frozen.json",
    PROSPECTIVE / "prospective_final.json",
    PROSPECTIVE / "prospective_stats.json",
    PROSPECTIVE / "prospective_dataset.json",
    REPO / "data/new_weeks/qc_report.json",
]
missing = [str(p.relative_to(REPO)) for p in evidence_paths if not p.exists()]
if missing:
    raise FileNotFoundError("Missing evidence files: " + ", ".join(missing))
inventory = pd.DataFrame(
    [
        {
            "file": p.relative_to(REPO).as_posix(),
            "bytes": p.stat().st_size,
            "sha256_now": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in evidence_paths
    ]
)
inventory.to_csv(RUN_DIR / "evidence_inventory.csv", index=False)
display(inventory[["file", "bytes"]])
runs = pd.DataFrame(read_json(KAGGLE / "runs.json"))
trained = pd.DataFrame(
    [json.loads(line) for line in (KAGGLE / "runs.jsonl").read_text().splitlines() if line]
)
keys = ["origin_set", "name", "origin", "seed"]
assert not runs.duplicated(keys).any(), "Duplicate result keys would double-count runs."
assert not trained.duplicated(keys).any(), "Duplicate training keys."
assert np.isfinite(runs[["RMSE", "val_RMSE"]].to_numpy()).all()
r3 = runs[runs.origin_set.eq("three")].copy()
r9 = runs[runs.origin_set.eq("nine")].copy()
display(
    pd.DataFrame(
        [
            {
                "artifact": "runs.jsonl",
                "rows": len(trained),
                "meaning": "recorded training runs (includes non-neural learners)",
            },
            {
                "artifact": "runs.json",
                "rows": len(runs),
                "meaning": "training + persistence + ensemble scores",
            },
            {
                "artifact": "three-origin rows",
                "rows": len(r3),
                "meaning": "Tables I and II use this group",
            },
            {
                "artifact": "nine-origin rows",
                "rows": len(r9),
                "meaning": "older confirmation group; differs from EXP-063 development",
            },
        ]
    )
)
print("Historical per-prediction archive present:", (KAGGLE / "predictions.pkl").exists())

# %% [markdown]
# ## 2. Where the data came from
#
# **The 2013–2024 case counts were parsed by the benchmark authors.** This project
# received their table, reordered and mapped its districts, kept missing weeks empty,
# and corrected one printed spreadsheet error. The local copy is
# `full_paper/data/dengue_cases_raw.csv`. This is a corrected and extended dataset;
# it is not a new collection of patient records.
#
# For 2024–2026 this project downloaded and parsed 127 official WER PDFs under the
# rules in `docs/NEW_WEEKS_RULES.md`. The final parse has 109 observed weeks,
# **8 additional corrected weeks**, and 10 missing weeks (109 + 8 + 10 = 127).
# The first scoring run exposed parser defects; run 2 uses amended rules.
#
# Weather comes from ERA5 via Open-Meteo; population comes from Sri Lanka's
# Department of Census and Statistics. URLs below are taken from project manifests;
# this notebook uses local files and does not contact the websites.

# %%
sources = pd.DataFrame(
    [
        {
            "input": "2013–2024 dengue counts",
            "source": "benchmark authors' parsed WER table",
            "local evidence": "full_paper/data/dengue_cases_raw.csv",
            "our work": "reorder, map, audit, correct one report",
        },
        {
            "input": "2024–2026 dengue counts",
            "source": "https://www.epid.gov.lk/weekly-epidemiological-report",
            "local evidence": "data/external/wer_new_manifest.csv",
            "our work": "download and parse; amendments documented",
        },
        {
            "input": "weather",
            "source": "https://archive-api.open-meteo.com/v1/archive",
            "local evidence": "data/raw/era5_openmeteo/",
            "our work": "aggregate daily data; apply causal lags",
        },
        {
            "input": "population",
            "source": "https://www.statistics.gov.lk/",
            "local evidence": "data/external/population_vintages.csv",
            "our work": "join official population with availability rules",
        },
        {
            "input": "benchmark and graph",
            "source": "https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2",
            "local evidence": "notebooks/baseline/",
            "our work": "audit array; correct two graph entries for prospective study",
        },
    ]
)
display(sources)
wer_manifest = pd.read_csv(REPO / "data/external/wer_new_manifest.csv")
qc = read_json(REPO / "data/new_weeks/qc_report.json")
display(
    pd.DataFrame([{k: qc[k] for k in ["expected_reports", "observed", "corrected", "missing"]}])
)
check_close(
    "new-report partition",
    qc["observed"] + qc["corrected"] + qc["missing"],
    len(wer_manifest),
    "qc_report.json and wer_new_manifest.csv",
    atol=0,
)
display(wer_manifest[["volume", "number", "url", "file"]].head(3))

# %% [markdown]
# This next cell checks the 127 downloaded PDFs against their recorded SHA-256 values,
# and checks that the parsed dataset still matches the EXP-063 run-2 checksum record.
# Missing original PDFs are reported as unavailable rather than treated as a pass.

# %%
pdf_checks = []
for row in wer_manifest.to_dict("records"):
    path = REPO / "data/raw/wer_new" / row["file"]
    status = "UNAVAILABLE"
    if path.exists():
        status = (
            "MATCH"
            if hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            else "MISMATCH"
        )
    pdf_checks.append({"file": row["file"], "status": status})
pdf_checks = pd.DataFrame(pdf_checks)
display(pdf_checks.status.value_counts().rename_axis("PDF checksum status").to_frame("files"))
check_close(
    "available new-report PDF checksum mismatches",
    int(pdf_checks.status.eq("MISMATCH").sum()),
    0,
    "downloaded PDFs vs wer_new_manifest.csv; unavailable files listed separately",
    atol=0,
)
dataset_record = read_json(PROSPECTIVE / "prospective_dataset.json")
dataset_checks = []
for name, expected in dataset_record["files"].items():
    path = REPO / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
    dataset_checks.append({"file": name, "status": "MATCH" if actual == expected else "MISMATCH"})
dataset_checks = pd.DataFrame(dataset_checks)
display(dataset_checks)
check_close(
    "prospective dataset checksum mismatches",
    int(dataset_checks.status.ne("MATCH").sum()),
    0,
    "parsed input files vs prospective_dataset.json",
    atol=0,
)

# %% [markdown]
# ## 3. Weekly data and missing weeks
#
# Rows are weeks and columns are districts. The available `data/corrected/` arrays
# are already derived data, not original reports. We use them to independently
# calculate persistence and inspect the split. Full raw rebuilding is optional below.
# Missing values stay missing; a forecast window touching a missing week is excluded.
# Kalmunai is excluded to follow the benchmark: the Ampara target is Ampara RDHS,
# not the whole administrative district including Kalmunai.

# %%
names = sorted(read_json(REPO / "notebooks/baseline/sri_lanka_adj_list.json"))
case_array = np.load(REPO / "data/corrected/rebuilt_cases.npy")
cases = case_array[..., 0] if case_array.ndim == 3 else case_array
index = pd.read_csv(REPO / "data/corrected/rebuilt_index.csv", parse_dates=["week_start"])
missing_week = np.isnan(cases).any(axis=1)
assert cases.shape == (len(index), len(names))
assert np.array_equal(missing_week, index.status.eq("missing")), "Missing flags disagree."
display(
    pd.DataFrame(
        [
            {
                "weeks": len(cases),
                "districts": len(names),
                "missing weeks": int(missing_week.sum()),
                "first date in current index": str(index.week_start.min().date()),
                "last date in current index": str(index.week_start.max().date()),
            }
        ]
    )
)
example = pd.DataFrame(cases[:5], columns=names, index=index.week_start.iloc[:5])
display(example[["Colombo", "Gampaha", "Ampara"]])
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(index.week_start, np.where(missing_week, np.nan, np.nansum(cases, axis=1)), lw=1)
ax.set(
    title="Development data: sum of the 25 modeled divisions",
    xlabel="Week",
    ylabel="Reported weekly cases",
)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### How the original benchmark was audited
#
# The old array lacks a date axis. The original audit matched each 25-district vector
# against the report table, then compared weather against ERA5 at several offsets.
# `30_audit.py` in the embedded source snapshots contains that calculation.
# Here we show the **retained audit log**, explicitly labeled; reading this log does
# not independently repeat PDF extraction or the ERA5 alignment test.

# %%
historical_log = (KAGGLE / "notebook_output.txt").read_text(encoding="utf-8", errors="replace")
audit_lines = [
    line
    for line in historical_log.splitlines()
    if any(
        term in line
        for term in [
            "rows 0-47",
            "rows from 2023",
            "array temperature",
            "'lag 12'",
            "array row 395",
            "row A (",
            "row B (",
            "integrity:",
        ]
    )
]
print("RETAINED EXP-050 LOG — not a new audit run")
print("\n".join(audit_lines))
corrections = read_json(REPO / "data/external/report_corrections.json")
display(
    Markdown(
        "The correction evidence is also recorded in `data/external/report_corrections.json`. "
        "The printed row sums to 7,165 over the modeled divisions; the corrected modeled total is 349. "
        "Including Kalmunai gives 351. These totals refer to different geographic coverage."
    )
)

# %% [markdown]
# ## 4. Why there are different RMSE numbers
#
# There are **three separate sets of evaluation weeks** in the current story:
#
# | Experiment | Model example | Persistence on the same evaluation | Used for |
# |---|---|---|---|
# | EXP-050, 3 origins × 3 seeds | AAGCN + PAGE | about 36.02 | Tables I, II, IV |
# | EXP-063, 9 purged development origins | adaptive PAGE + SEIR | about 28.54 | validation tuning, development comparison |
# | EXP-063, later weeks, 85 starts | adaptive PAGE + SEIR | about 34.75 | Table III |
#
# EXP-050 also has an older nine-origin confirmation group; it is **not** the purged
# EXP-063 group. Scores from different groups cannot establish which model is better.
#
# **RMSE calculation:** square each prediction error, average across all windows,
# districts and horizons, and take the square root. This gives one RMSE per run.
# Paper tables then average those run RMSEs. We do not average RMSEs of individual
# windows, and do not pool all runs before taking a square root.


# %%
def rmse(prediction, truth):
    return float(np.sqrt(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2)))


toy_truth = np.array([10, 20, 40])
toy_forecast = np.array([10, 20, 20])
print("Teaching example only (not experimental data):", rmse(toy_forecast, toy_truth))


def historical_folds():
    # This reproduces EXP-050's original unpurged partition of forecast starts.
    # Splitting start indices does not purge the overlapping three-week targets.
    ids = list(range(3, len(cases) - 3))
    output = []
    for origin in (0.55, 0.70, 0.85):
        cut = int(origin * len(ids))
        end = int(min(origin + 0.15, 1.0) * len(ids))
        parts = {"train": ids[: cut - 30], "val": ids[cut - 30 : cut], "test": ids[cut:end]}
        parts = {
            key: np.array([i for i in starts if not missing_week[i - 3 : i + 3].any()], dtype=int)
            for key, starts in parts.items()
        }
        output.append((origin, parts))
    return output


split_rows = []
baseline_rows = []
for origin, parts in historical_folds():
    for split, starts in parts.items():
        split_rows.append(
            {
                "origin": origin,
                "split": split,
                "windows": len(starts),
                "first target start": str(index.week_start.iloc[starts[0]].date()),
                "last target start": str(index.week_start.iloc[starts[-1]].date()),
            }
        )
        if split in ("val", "test"):
            # Original implementation scores torch.float32 tensors.
            truth = np.stack([cases[i : i + 3].T for i in starts]).astype(np.float32)
            prediction = np.repeat(cases[starts - 1, :, None], 3, axis=2).astype(np.float32)
            baseline_rows.append(
                {"origin": origin, "split": split, "recomputed RMSE": rmse(prediction, truth)}
            )
display(pd.DataFrame(split_rows))
baseline_check = pd.DataFrame(baseline_rows)
for row in baseline_check.to_dict("records"):
    metric = "val_RMSE" if row["split"] == "val" else "RMSE"
    saved = r3.loc[r3.name.eq("persistence") & r3.origin.eq(row["origin"]), metric].item()
    check_close(
        f"EXP-050 persistence {row['origin']} {row['split']}",
        row["recomputed RMSE"],
        saved,
        "derived cases → historical fold → predictions → RMSE",
        atol=2e-5,
    )
display(baseline_check)

# %% [markdown]
# **Historical split limitation:** EXP-050 separates forecast starts but does not
# purge the overlapping three-week target dates at partition boundaries. This notebook
# reproduces that protocol rather than calling it strictly leakage-free. EXP-063 uses
# purged partitions. Long climate lags also remove early training windows, and the
# reduced-training-data arms intentionally use less training data.
#
# Seeds repeat model training, not independent outbreaks. A 9-pair sign-flip test
# across 3 origins × 3 seeds describes training stability; it does not give nine
# independent time periods. The prospective block bootstrap addresses dependence
# across adjacent forecast starts, subject to its own block-length assumptions.

# %% [markdown]
# ## 5. What PAGE changes
#
# A **direct head** predicts the future case level. A **residual head** starts from
# the latest observed week and learns a correction. **PAGE** also learns a scalar
# gate that shrinks the departure toward persistence. The **PAGE + SEIR** variant
# generates that departure using a differentiable epidemic simulator.
#
# In standardized log-case space the gated head is
# $\hat z = p + \sigma(a)(u-p)$, where $p$ repeats the last observed week.
# At initialization $a=-2$, so $\sigma(a)\approx0.119$.
# The encoder is still a trained neural network; the persistence anchor is a fixed
# reference value, not a replacement for training. Table I uses squared error;
# NB likelihood is a separate ablation. PAGE is **not automatically an NB model**.

# %%
gate = 1 / (1 + np.exp(2))
print(f"Initial gate: {gate:.6f}")
print("Illustration in standardized log space: p=2, u=3 -> PAGE prediction", 2 + gate * (3 - 2))
head_names = pd.DataFrame(
    [
        {"saved head": "direct", "paper name": "direct", "meaning": "predict level"},
        {
            "saved head": "residual",
            "paper name": "residual",
            "meaning": "anchor + learned correction",
        },
        {"saved head": "gated", "paper name": "PAGE", "meaning": "p + sigmoid(a) * (u - p)"},
        {
            "saved head": "foi",
            "paper name": "pure SEIR decoder",
            "meaning": "whole forecast through simulator",
        },
        {
            "saved head": "foi_res",
            "paper name": "PAGE + SEIR",
            "meaning": "gated simulator departure",
        },
    ]
)
display(head_names)

# %% [markdown]
# ### What happens in one neural training run
#
# 1. Cut the fold's eligible windows: 3 observed weeks × 25 districts predict the
#    next 3 weeks. Fit the log-count mean and standard deviation on training history.
#    Transform each count with $z=(\log(1+y)-\mu)/s$.
# 2. Send those windows through the chosen encoder and head. Graph encoders mix
#    information across districts; the LSTM provides the non-graph comparison.
# 3. Update weights with Adam and cosine learning-rate decay. For the main head
#    comparison, minimize squared error in standardized log space.
# 4. After each epoch, convert forecasts back to counts and calculate validation
#    RMSE. Stop after 40 epochs without improvement, or at the 300-epoch budget.
#    Restore the weights with the lowest validation RMSE.
# 5. Score those weights on the test windows and save one row. Repeat for each
#    origin and seed, then average the saved RMSEs to obtain the table cells.
#
# The next cell **reads** the actual training defaults; it does not train. The
# configuration grid can override them: for example, NB/season arms use 400 epochs,
# and EXP-063 tunes learning rate and hidden width using validation only.

# %%
training_tree = ast.parse(MODULE_SOURCES["full_paper/kaggle/src/60_training.py"])
training_definition = next(
    node
    for node in training_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "run_fold"
)
training_defaults = {
    argument.arg: ast.literal_eval(value)
    for argument, value in zip(
        training_definition.args.kwonlyargs, training_definition.args.kw_defaults
    )
    if value is not None
}
display(
    pd.DataFrame(
        [
            {"setting": name, "EXP-050 default": training_defaults[name]}
            for name in [
                "hidden",
                "layers",
                "dropout",
                "lr",
                "weight_decay",
                "batch_size",
                "epochs",
                "patience",
            ]
        ]
    )
)

# %% [markdown]
# ## 6. Table I: encoder and baseline comparison
#
# Each cell below is **mean validation RMSE / mean test RMSE** from the stored
# per-run rows. We recompute the average rather than type the published numbers.
# Here `gated` is PAGE and `foi_res` is PAGE + SEIR. The same name “adaptive” in
# EXP-063 refers to another encoder, not to AAGCN.


# %%
def mean_score(frame, name, metric="val_RMSE"):
    values = frame.loc[frame.name.eq(name), metric]
    if values.empty:
        raise KeyError(f"No rows for {name!r}")
    return float(values.mean())


comparison = pd.DataFrame(
    {
        label: [
            f"{mean_score(r3, e + '+' + head):.2f} / {mean_score(r3, e + '+' + head, 'RMSE'):.2f}"
            for e in ENCODERS
        ]
        for head, label in [("direct", "Direct"), ("gated", "PAGE"), ("foi_res", "PAGE + SEIR")]
    },
    index=ENCODERS,
)
comparison.index.name = "Encoder"
display(comparison)
baseline_names = [
    "persistence",
    "graph=none",
    "graph=gcn",
    "R3 STID-style MLP",
    "R4b k-NN",
    "R4a NB-GLM",
    "K7 trees",
    "K6 trees + climate",
    "B",
]
baselines = pd.DataFrame(
    [
        {
            "saved name": name,
            "validation RMSE": mean_score(r3, name),
            "test RMSE": mean_score(r3, name, "RMSE"),
            "status": "DATE-JOIN RECHECK PENDING"
            if name in ["B", "K6 trees + climate", "R3 STID-style MLP", "R4a NB-GLM"]
            else "historical result",
        }
        for name in baseline_names
    ]
)
display(baselines)
comparison.to_csv(RUN_DIR / "table_I_encoders.csv")
baselines.to_csv(RUN_DIR / "table_I_baselines.csv", index=False)
print("One concrete run, before averaging:")
display(
    r3.loc[
        r3.name.eq("AAGCN+gated"),
        ["origin", "seed", "val_RMSE", "RMSE", "best_epoch", "epochs_ran"],
    ]
)

# %% [markdown]
# ## 7. Table II: ablation and paired comparisons
#
# Changing one component while keeping the encoder and evaluation fixed tests where
# a gain comes from. **Negative difference means the first model has lower error.**
# A “win” is one matched (origin, seed) pair with lower RMSE. Persistence has one
# deterministic score per origin, reused for each seed; it was not trained three times.
# We require a complete match and stop on missing pairs.


# %%
def paired_differences(frame, arm, control, metric="val_RMSE"):
    arm_rows = frame.loc[frame.name.eq(arm), ["origin", "seed", metric]]
    ref_rows = frame.loc[frame.name.eq(control), ["origin", "seed", metric]]
    if arm_rows.empty or ref_rows.empty:
        raise ValueError(f"Missing arm/control: {arm}, {control}")
    join_keys = ["origin"] if ref_rows.seed.lt(0).all() else ["origin", "seed"]
    matched = arm_rows.merge(
        ref_rows[[*join_keys, metric]],
        on=join_keys,
        how="left",
        suffixes=("_arm", "_control"),
        validate="many_to_one",
    )
    if len(matched) != len(arm_rows) or matched[f"{metric}_control"].isna().any():
        raise ValueError("Incomplete pairs; refusing to drop missing matches.")
    matched = matched.sort_values(["origin", "seed"])
    return (matched[f"{metric}_arm"] - matched[f"{metric}_control"]).to_numpy()


def sign_flip(values):
    values = np.asarray(values, dtype=float)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(values))), dtype=float)
    return float(np.mean(np.abs(signs @ values / len(values)) >= abs(values.mean()) - 1e-12))


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(order))
    ceiling = 1.0
    for rank_index in range(len(order) - 1, -1, -1):
        pos = order[rank_index]
        ceiling = min(ceiling, pvalues[pos] * len(order) / (rank_index + 1))
        adjusted[pos] = ceiling
    return adjusted


ablation = pd.DataFrame(
    {
        label: [mean_score(r3, e + "+" + head) for e in ENCODERS]
        for head, label in [
            ("direct", "Direct"),
            ("residual", "Residual"),
            ("gated", "PAGE"),
            ("foi_res", "PAGE + SEIR"),
        ]
    },
    index=ENCODERS,
)
display(ablation)
paired_rows = []
for encoder in ENCODERS:
    for control in [encoder + "+direct", "persistence"]:
        for metric in ["val_RMSE", "RMSE"]:
            delta = paired_differences(r3, encoder + "+gated", control, metric)
            assert len(delta) == 9
            paired_rows.append(
                {
                    "encoder": encoder,
                    "control": control,
                    "metric": metric,
                    "mean difference": delta.mean(),
                    "wins": f"{(delta < 0).sum()}/9",
                    "p_raw": sign_flip(delta),
                }
            )
paired = pd.DataFrame(paired_rows)
paired["p_BH_24_comparisons"] = bh_adjust(paired.p_raw)
display(paired)
rescue_rows = []
for encoder in ENCODERS:
    dv = paired_differences(r3, encoder + "+foi_res", encoder + "+gated")
    dt = paired_differences(r3, encoder + "+foi_res", encoder + "+gated", "RMSE")
    rescue_rows.append(
        {
            "encoder": encoder,
            "SEIR - PAGE (val)": dv.mean(),
            "val wins": int((dv < 0).sum()),
            "p_raw": sign_flip(dv),
            "SEIR - PAGE (test)": dt.mean(),
            "test wins": int((dt < 0).sum()),
        }
    )
rescue = pd.DataFrame(rescue_rows)
failing = rescue.encoder.isin(["STGAT", "A3TGCN", "DCRNN"])
rescue.loc[failing, "p_BH_failing_3"] = bh_adjust(rescue.loc[failing, "p_raw"])
rescue["passes_physics_rule"] = (
    failing
    & rescue["SEIR - PAGE (val)"].lt(0)
    & rescue["val wins"].ge(7)
    & rescue.p_BH_failing_3.lt(0.05)
)
display(rescue)
ablation.to_csv(RUN_DIR / "table_II_ablation.csv")
paired.to_csv(RUN_DIR / "paired_PAGE.csv", index=False)

# %% [markdown]
# The adjustment family matters: the PAGE-versus-direct table uses 24 comparisons;
# the physics-credit test adjusts over the three failing encoders. We keep both
# families explicit. Failure to detect a difference does **not** establish equivalence
# or prove that epidemic physics can never help.

# %% [markdown]
# ## 8. Other levers: loss, climate, graph and augmentation
#
# The stored lever table tells us each arm and its matched control. We recalculate
# every difference, win count, exact sign-flip p-value and BH adjustment from the run
# rows, then compare to that CSV. The NB result belongs to model B with seasonal
# features; it is **not** the squared-error PAGE result in Table I.

# %%
stored_levers = pd.read_csv(KAGGLE / "levers.csv")
lever_rows = []
for row in stored_levers.to_dict("records"):
    dv = paired_differences(r3, row["arm"], row["control"])
    dt = paired_differences(r3, row["arm"], row["control"], "RMSE")
    lever_rows.append(
        {
            "lever": row["lever"],
            "arm": row["arm"],
            "control": row["control"],
            "val delta": dv.mean(),
            "test delta": dt.mean(),
            "wins": f"{(dv < 0).sum()}/{len(dv)}",
            "p": sign_flip(dv),
        }
    )
levers = pd.DataFrame(lever_rows)
levers["p_adj"] = bh_adjust(levers.p)
for metric in ["val delta", "test delta", "p", "p_adj"]:
    check_close(
        "lever recalculation: " + metric,
        levers[metric].to_numpy(),
        stored_levers[metric].to_numpy(),
        "runs.json → matched pairs → levers.csv",
        atol=1e-8,
    )
assert levers.wins.tolist() == stored_levers.wins.tolist()
display(levers[["lever", "val delta", "test delta", "wins", "p_adj"]])
levers.to_csv(RUN_DIR / "recomputed_levers.csv", index=False)
display(
    Markdown(
        "**Reading the NB row:** validation improves by about 0.76 RMSE, but test worsens "
        "by about 1.61. These date-joined configurations remain subject to EXP-062. "
        "Model B follows the historical adoption rule; it is not simply the smallest "
        "validation number or the test winner in Table I."
    )
)

# %% [markdown]
# ## 9. Diagnostics and retained evidence
#
# These CSVs were computed during the original study. Without `predictions.pkl` we
# can inspect and summarize them, but cannot independently regenerate the residual
# regressions and correlations from predictions in the default run.
# The SEIR-floor diagnostic needs states and simulator inversion; its implementation
# is included in the optional complete development run.
#
# Residual recovery uses the **NB + seasonal-feature configuration**, not the direct
# squared-error rows in Table I. The 0.91–0.97 correlation range refers to selected
# working models (AAGCN, ASTGCN, LSTM, SEIR-LSTM and k-NN), not every model.
# High residual correlation suggests shared errors; it does not prove a data ceiling.

# %%
recovery = pd.read_csv(KAGGLE / "residual_recovery.csv", index_col=0)
corr = pd.read_csv(KAGGLE / "residual_correlation.csv", index_col=0)
seir_diagnosis = pd.read_csv(KAGGLE / "seir_decoder_diagnosis.csv")
display(Markdown("**Retained diagnostic: validation residual variance recovered by a linear fit**"))
display(recovery)
selected_models = ["AAGCN", "ASTGCN", "LSTM", "SEIR-LSTM", "k-NN"]
pair_corr = [corr.loc[a, b] for a, b in itertools.combinations(selected_models, 2)]
print("Selected working-model pairwise correlation range:", min(pair_corr), "to", max(pair_corr))
display(Markdown("**Retained diagnostic: simulator floor and required force of infection**"))
display(seir_diagnosis)

# %% [markdown]
# ### Persistence predictability: clarify what r² means here
#
# The implementation in `80_analysis.py` calculates **squared pooled Pearson
# correlation on raw counts**. Some older descriptions call it log-case r²; that
# wording does not match the code. This is a pooled association across districts
# and time, not an out-of-sample R² score and not a causal effect of weather.
# The available case array allows a new calculation of the lag-one figure.
# The weather figure below is labeled as retained, because the current date-aligned
# weather array differs from the one used for EXP-050 (EXP-062 is pending).

# %%
valid = np.isfinite(cases[:-1]) & np.isfinite(cases[1:])
lag_one_r2 = float(np.corrcoef(cases[:-1][valid], cases[1:][valid])[0, 1] ** 2)
historical_summary = read_json(KAGGLE / "results.json")
check_close(
    "raw-count lag-one pooled Pearson r²",
    lag_one_r2,
    historical_summary["r2_lag1"],
    "derived cases → paired raw counts → corrcoef squared",
    atol=1e-10,
)
print("Recomputed raw-count lag-one r²:", lag_one_r2)
print(
    "Retained best climate r² (old alignment; pending recheck):",
    historical_summary["r2_best_climate"],
)

# %% [markdown]
# ## 10. Table III: amended prospective evaluation, 2024–2026
#
# The neural encoder here is adaptive (Graph WaveNet-style). All three heads use
# fixed settings chosen on development validation. Each has three seeds; persistence,
# seasonal naive and AR(3) are deterministic and have one row each.
# The table below averages **per-seed metrics**, matching the paper's table builder.
# MAE/SMAPE/MAPE are secondary; no significance claim is made for their differences.
# The evaluation is amended run 2, not an untouched prospective trial.

# %%
final = read_json(PROSPECTIVE / "prospective_final.json")
final_rows = pd.DataFrame(final["rows"])
saved_stats = read_json(PROSPECTIVE / "prospective_stats.json")
assert len(final["test_starts"]) == final["n_test"]
prospective_table = final_rows.groupby("name", sort=False)[
    ["RMSE", "MAE", "SMAPE", "MAPE", "RMSE_h1", "RMSE_h2", "RMSE_h3"]
].mean()
display(prospective_table)
print(
    "Test starts:",
    final["n_test"],
    "| train windows:",
    final["n_train"],
    "| validation windows:",
    final["n_val"],
)
display(final_rows.groupby("name", sort=False)[["RMSE_2024", "RMSE_2025", "RMSE_2026"]].mean())
prospective_table.to_csv(RUN_DIR / "table_III_prospective.csv")

# %% [markdown]
# ### Independently recalculate the prospective persistence forecast
#
# This uses the current corrected data loader, constructs the actual final split,
# and checks the start indices against the stored result. We then build persistence
# predictions from the case array and calculate its metrics. No neural-network result
# is needed for this calculation. The same cell checks that train, validation and
# test partitions share no target weeks.

# %%
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "analysis/lib"))
sys.path.insert(0, str(REPO / "seirgnn2"))
import corrected_data as cd
import core as prospective_core
import classical
from dengue_gnn.metrics import score

combined_data, last_dev = cd.load_with_new_weeks()
final_fold = prospective_core.build_final_fold(combined_data.cases, combined_data.missing, last_dev)
test_starts = final_fold.idx["test"]
check_close(
    "prospective test starts",
    test_starts,
    final["test_starts"],
    "current loader/core vs prospective_final.json",
    atol=0,
)
target_sets = {
    key: {i + h for i in starts for h in range(3)} for key, starts in final_fold.idx.items()
}
for a, b in itertools.combinations(target_sets, 2):
    assert not target_sets[a].intersection(target_sets[b]), f"Shared target weeks: {a}, {b}"
prospective_truth = np.stack([combined_data.cases[i : i + 3].T for i in test_starts])
prospective_persistence = classical.persistence(combined_data.cases, test_starts)
baseline_metrics = score(prospective_persistence, prospective_truth)
for metric in ["RMSE", "MAE", "SMAPE", "MAPE"]:
    check_close(
        "prospective persistence " + metric,
        baseline_metrics[metric],
        prospective_table.loc["Persistence", metric],
        "case values → persistence → metric",
        atol=1e-9,
    )
display(pd.DataFrame([baseline_metrics], index=["Recomputed persistence"]))

# %% [markdown]
# ### Recompute the block bootstrap, rather than copy its p-values
#
# `prospective_final.json` retains one mean squared loss for each forecast start,
# averaged over districts, horizons and seeds. We call the original bootstrap on
# those arrays (10,000 replicates, block length 8), then apply Holm across two primary
# tests. All sensitivity and secondary comparisons are also checked.
#
# **Small estimand difference:** the table reports mean(seed RMSE). The bootstrap
# uses sqrt(mean(seed MSE)). Those operations are not identical, so the bootstrap
# difference is calculated from its loss arrays, not by subtracting rounded table cells.
# A two-sided non-significant test does not demonstrate that two models are equivalent.

# %%
losses = {name: np.asarray(values, dtype=float) for name, values in final["losses"].items()}
assert all(
    len(values) == final["n_test"] and np.isfinite(values).all() for values in losses.values()
)
persistence_losses = classical.per_start_loss(prospective_persistence, prospective_truth)
check_close(
    "prospective per-start persistence losses",
    persistence_losses,
    losses["Persistence"],
    "case values → per-start MSE → prospective_final.json",
    atol=1e-8,
)
primary_pairs = [
    ("P1", "Adaptive SEIR-GNN", "Adaptive gated non-SEIR"),
    ("P2", "Adaptive SEIR-GNN", "Persistence"),
]
primary_results = [
    dict(test=tag, a=a, b=b, **classical.block_bootstrap(losses[a], losses[b], 8, 10000))
    for tag, a, b in primary_pairs
]
for row, padj in zip(primary_results, classical.holm([row["p"] for row in primary_results])):
    row["p_holm"] = float(padj)
for category in ["primary", "sensitivity", "secondary"]:
    for stored in saved_stats[category]:
        current = classical.block_bootstrap(
            losses[stored["a"]], losses[stored["b"]], stored["block"], stored["reps"]
        )
        for metric in ["delta", "lo", "hi", "p"]:
            check_close(
                f"bootstrap {category}: {stored['a']} vs {stored['b']} block={stored['block']} {metric}",
                current[metric],
                stored[metric],
                "stored per-start losses → deterministic bootstrap",
                atol=1e-12,
            )
for row, stored in zip(primary_results, saved_stats["primary"]):
    check_close(
        row["test"] + " Holm p",
        row["p_holm"],
        stored["p_holm"],
        "Holm across P1 and P2",
        atol=1e-12,
    )
display(pd.DataFrame(primary_results)[["test", "a", "b", "delta", "lo", "hi", "p", "p_holm"]])
display(
    Markdown(
        "P1 asks whether the SEIR branch improves on PAGE. P2 asks whether the SEIR variant "
        "beats persistence. Neither primary test establishes an improvement at the 5% threshold. "
        "The simpler PAGE headline was chosen after observing these results."
    )
)

# %% [markdown]
# ## 11. Tuning and Table IV: counts and computational cost
#
# Tuning tried 2 learning rates × 2 hidden sizes × 9 purged origins × 3 seeds =
# 108 runs per neural arm. We recalculate validation means and the selected settings.
# The old 27.34 versus 28.54 numbers belong to this development evaluation;
# they are not Table I's AAGCN + PAGE versus persistence scores.

# %%
dev = pd.DataFrame(read_json(PROSPECTIVE / "prospective_dev.json"))
frozen = read_json(PROSPECTIVE / "prospective_frozen.json")
tuning = (
    dev.loc[~dev.name.eq("persistence")]
    .groupby("name", sort=False)
    .agg(
        runs=("val_RMSE", "size"), validation_RMSE=("val_RMSE", "mean"), test_RMSE=("RMSE", "mean")
    )
)
assert tuning.runs.eq(27).all()
for name, row in tuning.iterrows():
    check_close(
        "grid mean: " + name,
        row.validation_RMSE,
        frozen["dev_val_RMSE"][name],
        "prospective_dev.json → mean validation RMSE",
        atol=1e-10,
    )
display(tuning)
display(pd.DataFrame(frozen["arms"]).T.rename_axis("neural arm"))
for arm, setting in frozen["arms"].items():
    group = tuning.loc[tuning.index.str.startswith(arm + " | ")]
    selected_tag = f"{arm} | lr={setting['lr']:g} hidden={setting['hidden']}"
    assert selected_tag == group.validation_RMSE.idxmin(), (
        "Frozen setting disagrees with validation selection."
    )
development_comparison = pd.DataFrame(
    [
        {
            "model": "Adaptive SEIR-GNN (selected development setting)",
            "test RMSE": mean_score(dev, "Adaptive SEIR-GNN | lr=0.003 hidden=64", "RMSE"),
        },
        {
            "model": "Persistence on these nine origins",
            "test RMSE": mean_score(dev, "persistence", "RMSE"),
        },
    ]
)
display(development_comparison)

classical_dev = pd.DataFrame(read_json(PROSPECTIVE / "prospective_dev_classical.json"))
assert len(classical_dev) == 54 and not classical_dev.duplicated(["name", "origin"]).any()
classical_tuning = classical_dev.groupby("name", sort=False).agg(
    origins=("val_RMSE", "size"),
    validation_RMSE=("val_RMSE", "mean"),
    test_RMSE=("RMSE", "mean"),
)
assert classical_tuning.origins.eq(9).all()
for name, row in classical_tuning.loc[classical_tuning.index.str.startswith("AR(3)")].iterrows():
    check_close(
        "classical grid mean: " + name,
        row.validation_RMSE,
        frozen["dev_val_RMSE"][name],
        "prospective_dev_classical.json → mean validation RMSE",
        atol=1e-10,
    )
selected_ar = f"AR(3) ridge | alpha={frozen['ar_alpha']:g}"
assert (
    selected_ar
    == classical_tuning.loc[classical_tuning.index.str.startswith("AR(3)")].validation_RMSE.idxmin()
)
display(classical_tuning)
print("AR(3) ridge alpha selected using validation only:", frozen["ar_alpha"])

# %% [markdown]
# The next cell executes the **model definitions**, not training, and counts their
# parameters. It uses the embedded source snapshots and the graph used by the table
# builder. Timing is a historical measurement read from `runs.jsonl`, not a local
# speed claim. Seconds per epoch = total elapsed seconds / total epochs across the
# nine runs; seconds per run = mean elapsed seconds. Windows training can take much
# longer than the reported four-worker Kaggle session.

# %%
import torch

model_ns = {"__name__": "icitr_parameter_counts"}
exec(
    compile(
        MODULE_SOURCES["full_paper/kaggle/src/40_architectures.py"], "40_architectures.py", "exec"
    ),
    model_ns,
)
adjacency = read_json(REPO / "notebooks/baseline/sri_lanka_adj_list.json")
graph_names = list(adjacency)
dense = np.eye(len(graph_names), dtype=np.float32)
for district, neighbors in adjacency.items():
    for neighbor in neighbors:
        if neighbor in graph_names:
            dense[graph_names.index(district), graph_names.index(neighbor)] = 1
source_node, destination_node = np.nonzero(dense)
model_ns["EDGE_INDEX"] = torch.tensor(np.stack([source_node, destination_node]), dtype=torch.long)
exec(
    compile(MODULE_SOURCES["full_paper/kaggle/src/50_models.py"], "50_models.py", "exec"), model_ns
)
cost_rows = []
for encoder in ENCODERS:
    row = {"encoder": encoder}
    for head, label in [("direct", "direct"), ("gated", "PAGE"), ("foi_res", "+SEIR")]:
        net = model_ns["Net"](3, 25, backbone=encoder, head=head)
        row[label + " parameters"] = sum(parameter.numel() for parameter in net.parameters())
        times = trained.loc[trained.origin_set.eq("three") & trained.name.eq(encoder + "+" + head)]
        assert len(times) == 9
        row[label + " s/epoch"] = times.elapsed.sum() / times.epochs_ran.sum()
        if head == "gated":
            row["PAGE s/run"] = times.elapsed.mean()
    cost_rows.append(row)
cost_table = pd.DataFrame(cost_rows).set_index("encoder")
display(cost_table)
check_close(
    "one extra gate parameter",
    cost_table["PAGE parameters"] - cost_table["direct parameters"],
    np.ones(len(ENCODERS)),
    "count parameters from code",
    atol=0,
)
cost_table.to_csv(RUN_DIR / "table_IV_cost.csv")
print("Retained complete-session wall time (hours):", historical_summary["hours"])

# %% [markdown]
# ## 12. Check the paper's numbers and report the limits
#
# We compare the numeric cells of all **four current LaTeX tables** against their
# newly calculated values, including the prospective seed ranges. This detects stale
# generated tables. The older `verify_paper_numbers.py` is also included and executed
# with its report redirected to this notebook's output folder.
# Its “85 checks” use a predefined list of expected prose values; they do **not**
# automatically discover every number in an edited paper. Reading a diagnostic CSV
# or log is correctly treated as retained evidence, not a fresh experiment.


# %%
def latex_table_values(filename, label):
    text = (REPO / "full_paper/icitr/tables" / filename).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith(label + " &"):
            # Ignore formatting commands; retain numbers inside the data cells.
            cells = [
                re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}", "", cell) for cell in line.split("&")[1:]
            ]
            return [
                [
                    float(value.replace(",", ""))
                    for value in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", cell)
                ]
                for cell in cells
            ]
    raise KeyError(f"No row {label!r} in {filename}")


table_checks = []
for encoder in ENCODERS:
    expected = [
        [mean_score(r3, encoder + "+" + head), mean_score(r3, encoder + "+" + head, "RMSE")]
        for head in ["direct", "gated", "foi_res"]
    ]
    actual = latex_table_values("encoders.tex", encoder)
    assert len(actual) == len(expected)
    ok = all(
        [
            check_close(
                f"Table I {encoder} cell {i}", cell, ref, "encoders.tex vs runs.json", atol=0.00501
            )
            for i, (cell, ref) in enumerate(zip(actual, expected))
        ]
    )
    table_checks.append({"table": "I", "row": encoder, "match": ok})
for encoder in ENCODERS:
    actual = latex_table_values("rescue.tex", encoder)
    sr = rescue.loc[rescue.encoder.eq(encoder)].iloc[0]
    expected = [
        [mean_score(r3, encoder + "+" + h)] for h in ["direct", "residual", "gated", "foi_res"]
    ]
    expected += [
        [sr["SEIR - PAGE (val)"], sr["val wins"], 9],
        [sr["SEIR - PAGE (test)"], sr["test wins"], 9],
    ]
    assert len(actual) == len(expected)
    ok = all(
        [
            check_close(
                f"Table II {encoder} cell {i}", cell, ref, "rescue.tex vs paired runs", atol=0.00501
            )
            for i, (cell, ref) in enumerate(zip(actual, expected))
        ]
    )
    table_checks.append({"table": "II", "row": encoder, "match": ok})
table_baselines = {
    "Persistence": "persistence",
    "No message passing (res.)": "graph=none",
    "GCN (res.)": "graph=gcn",
    r"STID-style MLP~\cite{shao2022stid}": "R3 STID-style MLP",
    "$k$-NN analogues": "R4b k-NN",
    "NB-GLM": "R4a NB-GLM",
    "Boosted trees": "K7 trees",
    "Boosted trees + climate": "K6 trees + climate",
    "$B$: AAGCN, NB, season": "B",
}
for label, name in table_baselines.items():
    actual = latex_table_values("encoders.tex", label)
    expected = [[mean_score(r3, name), mean_score(r3, name, "RMSE")]]
    ok = check_close(
        f"Table I baseline {label}", actual, expected, "encoders.tex vs runs.json", atol=0.00501
    )
    table_checks.append({"table": "I", "row": label, "match": ok})
prospective_labels = {
    "Persistence": "Persistence",
    r"\model{} (adaptive graph)": "Adaptive gated non-SEIR",
    r"\model{} + SEIR branch": "Adaptive SEIR-GNN",
    "Residual head (no gate)": "Adaptive residual",
    "AR(3) ridge": "AR(3) ridge",
    "Seasonal naive": "Seasonal naive",
}
for label, name in prospective_labels.items():
    actual = latex_table_values("prospective.tex", label)
    values = final_rows[final_rows.name.eq(name)]
    row = prospective_table.loc[name]
    rmse_cell = [row.RMSE]
    if len(values) > 1:
        rmse_cell += [values.RMSE.min(), values.RMSE.max()]
    expected = [
        rmse_cell,
        [row.MAE],
        [row.SMAPE],
        [row.MAPE],
        [row.RMSE_h1, row.RMSE_h2, row.RMSE_h3],
    ]
    tolerances = [0.00501, 0.00501, 0.05001, 0.05001, 0.05001]
    assert len(actual) == len(expected)
    ok = all(
        [
            check_close(
                f"Table III {name} cell {i}",
                cell,
                ref,
                "prospective.tex vs saved seed metrics",
                atol=tol,
            )
            for i, (cell, ref, tol) in enumerate(zip(actual, expected, tolerances))
        ]
    )
    table_checks.append({"table": "III", "row": name, "match": ok})
for encoder, row in cost_table.iterrows():
    actual = latex_table_values("compute.tex", encoder)
    expected = [
        [row["direct parameters"]],
        [row["PAGE parameters"] - row["direct parameters"]],
        [row["+SEIR parameters"] - row["direct parameters"]],
        [row["direct s/epoch"]],
        [row["PAGE s/epoch"]],
        [row["+SEIR s/epoch"]],
        [row["PAGE s/run"]],
    ]
    tolerances = [0, 0, 0, 0.00501, 0.00501, 0.00501, 0.50001]
    assert len(actual) == len(expected)
    ok = all(
        [
            check_close(
                f"Table IV {encoder} cell {i}",
                cell,
                ref,
                "compute.tex vs code and training log",
                atol=tol,
            )
            for i, (cell, ref, tol) in enumerate(zip(actual, expected, tolerances))
        ]
    )
    table_checks.append({"table": "IV", "row": encoder, "match": ok})
print("Table checks:")
display(pd.DataFrame(table_checks))

# %%
verification_source = MODULE_SOURCES["scripts/verify_paper_numbers.py"]
# Redirect the single report assignment; keep all numerical calculations unchanged.
tree = ast.parse(verification_source)
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "OUT" for target in node.targets
    ):
        node.value = ast.Call(
            func=ast.Name(id="Path", ctx=ast.Load()),
            args=[ast.Constant(str(RUN_DIR / "NUMBER_SOURCES_rechecked.txt"))],
            keywords=[],
        )
ast.fix_missing_locations(tree)
verification_ns = {
    "__file__": str(REPO / "scripts/verify_paper_numbers.py"),
    "__name__": "icitr_saved_number_checks",
}
exec(compile(tree, "embedded_verify_paper_numbers.py", "exec"), verification_ns)
print(
    "Predefined prose checks:",
    verification_ns["n_ok"],
    "match;",
    verification_ns["n_bad"],
    "mismatch",
)
check_close(
    "predefined prose-value mismatches",
    verification_ns["n_bad"],
    0,
    "embedded verifier: fixed expected-value list",
    atol=0,
)

# %% [markdown]
# ### Coverage and open issues
#
# | Paper result | Default notebook action | What is still needed for stronger verification |
# |---|---|---|
# | Table I comparison | recalculate run means; independently recalculate persistence | fresh training or original predictions for neural RMSE |
# | Table II ablation and PAGE paired tests | recalculate every matched difference, win and sign-flip test | fresh training to verify per-run neural scores |
# | Loss/graph/climate/augmentation levers | recalculate from run rows, compare saved CSV | date-joined results need EXP-062 |
# | Benchmark data audit | rebuild case table; recalculate vector matches, future rows, weather alignment and persistence | original correction PDF extraction and complete source bundle |
# | Lag-one r² | calculate raw-count pooled Pearson r² from cases | external source validation of the case table |
# | Best climate r² | show retained EXP-050 value | rerun with documented alignment; EXP-062 |
# | SEIR-floor diagnostic | display retained simulator-inversion CSV | section 14 reruns diagnostic from inputs |
# | Residual recovery/correlation | summarize retained CSVs, identify configurations | original prediction archive or fresh training |
# | Table III metrics | recalculate per-seed means; persistence from cases | fresh adaptive model training for neural metrics |
# | Prospective primary/secondary/sensitivity tests | recompute from the saved per-start loss arrays | prediction-level reconstruction for neural losses |
# | Tuning | recalculate neural/classical validation grids and selection | fresh grid training for independent run verification |
# | Table IV | parameters from code, elapsed time from logs | fresh timing is hardware-dependent |
# | 2017 outbreak/death figures; seroprevalence prior | external cited figures, not training results | check the actual publications; not authenticated here |
#
# **Known interpretation limits:** no significant difference is not equivalence;
# shared residuals are not proof of a data ceiling; EXP-050 target boundaries are
# unpurged; the amended prospective trial must be described as amended. The paper's
# wording “are worse on 0/9 runs” for RevIN/STID/TimeGAN conflicts with the intended
# “improve on 0/9 runs.” The notebook reports the actual win counts and does not edit
# the paper. References and the anonymized code-link placeholder remain outside this audit.
# The paper's “every arm” seasonal-naive margin also omits AR(3): across all five
# competitors the actual range is 26.1–27.7 RMSE; 26.7–27.7 covers the three neural
# arms and persistence only. These are annotation issues, not fabricated score rows.

# %%
check_table = pd.DataFrame(CHECKS)
check_table.to_csv(RUN_DIR / "calculation_checks.csv", index=False)
display(check_table.groupby("status").size().to_frame("checks"))
failures = check_table[check_table.status.ne("PASS")]
if not failures.empty:
    display(failures)
print("Open the CSV files in:", RUN_DIR)
print(
    "These are independently recalculated summaries and baseline checks; this default run did not train models."
)
