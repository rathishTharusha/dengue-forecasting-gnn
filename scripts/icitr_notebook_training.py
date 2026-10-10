# Percent-format notebook continuation; names come from earlier walkthrough cells.
# Mathematical Unicode in Markdown is intentional.
# ruff: noqa: F821
# %% [markdown]
# ## 14. Optional: repeat the complete development study locally
#
# The original 810-run program is embedded in this notebook. This section writes
# those implementation snapshots to a private workspace and executes them in order,
# retaining new predictions and run logs. It does not call the old notebook, use old
# scores as new training results, or replace any paper artifacts.
#
# There are two input routes:
#
# - **`derived` (default):** use the locally available corrected arrays. This allows
#   real training now, but uses the current corrected calendar/weather/population.
#   Scores for date-joined configurations can change; this is not a byte-identical
#   reproduction of the old source dataset. The historical 141-entry graph is used.
# - **`original`:** rebuild from the complete 58-file `SOURCES.csv` bundle used by the
#   original notebook. The bundle is absent in this checkout. The preflight reports
#   missing inputs clearly; this notebook does not download or substitute PDFs.
#   Both routes execute the embedded current parser, including its calendar fix;
#   neither recreates the old date-joined scores byte for byte.
#
# The historical Kaggle run used four Linux workers and took about three hours.
# On Windows the original program falls back to one worker, so a complete run may
# take substantially longer. All 810 runs, early stopping and epoch budgets are
# preserved. It includes the older nine-origin confirmation, not EXP-063's purged
# nine-origin tuning. Every fresh result is labeled separately from the paper result.

# %%
EXP050_INPUTS = "derived"  # "derived" or "original"; only used with FULL_RETRAIN=True.
EXP050_SOURCE_BUNDLE = REPO / "full_paper/kaggle/dataset"
# To resume interrupted development training, paste the prior exp050 workspace path here.
EXP050_RESUME_WORKSPACE = None


def original_bundle_preflight(bundle):
    bundle = Path(bundle)
    manifest = bundle / "SOURCES.csv"
    if not manifest.exists():
        return pd.DataFrame(
            [{"file": "SOURCES.csv", "status": "MISSING; original-source rebuild unavailable"}]
        )
    with manifest.open(encoding="utf-8") as handle:
        entries = list(csv.DictReader(handle))
    manifest_fields = {"file", "sha256", "bytes", "source", "manifest_match"}
    if not entries or not manifest_fields.issubset(entries[0]):
        return pd.DataFrame([{"file": "SOURCES.csv", "status": "INVALID MANIFEST"}])
    census_spelling = {"Batticaloa": "Baticaloa", "Moneragala": "Monaragala"}
    separate_a1 = {"Anuradhapura", "Moneragala", "Polonnaruwa"}
    required = {
        "graph/sri_lanka_adj_list.json",
        "graph/gadm41_LKA_1.json",
        "graph/disease_config.json",
        "benchmark/sri_lanka_2013-2022_shifted.npy",
        "cases/output_Dengue Fever.csv",
        "cases/vol_48_no_02-english_1.pdf",
        "population/Mid-year_population_by_district_and_sex_2024.pdf",
    }
    for name in names:
        census_name = census_spelling.get(name, name)
        suffix = "_A1.pdf" if name in separate_a1 else ".pdf"
        required.add("population/census2012/" + census_name + suffix)
        required.add("climate/" + name + ".json")
    recorded = [row["file"] for row in entries]
    complete = (
        len(entries) == 58
        and len(set(recorded)) == 58
        and required.issubset(recorded)
        and sum(name.startswith("wheels/") and name.endswith(".whl") for name in recorded) == 1
    )
    rows = [
        {
            "file": "SOURCES.csv: complete 58-file inventory",
            "status": "MATCH" if complete else "INCOMPLETE INVENTORY",
        }
    ]
    for row in entries:
        path = (bundle / row["file"]).resolve()
        if not path.is_relative_to(bundle.resolve()):
            rows.append({"file": row["file"], "status": "OUTSIDE SOURCE BUNDLE"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        rows.append(
            {
                "file": row["file"],
                "status": "MATCH" if digest == row["sha256"] else "MISSING OR CHANGED",
            }
        )
    return pd.DataFrame(rows)


display(original_bundle_preflight(EXP050_SOURCE_BUNDLE))
if not FULL_RETRAIN:
    print(
        "The 810-run development training is disabled. Set FULL_RETRAIN=True to enable optional training."
    )

# %% [markdown]
# ### Pipeline order and exact implementation
#
# | Source snapshot | What executes |
# |---|---|
# | `20_data.py` | window cutting, training-only normalization, graph construction |
# | `40_architectures.py` | five ST-GNN definitions and LSTM |
# | `50_models.py` | forecast heads, SEIR simulator and likelihoods |
# | `55_diagnosis.py` | zero-transmission floor and simulator inversion |
# | `60_training.py` | Adam, cosine decay, early stopping, augmentation and baselines |
# | `70_experiments.py` | all configurations and 810 training jobs |
# | `80_analysis.py` | paired tests, lever table, residual analyses and plots |
# | `90_outputs.py` | fresh CSVs, JSONs and summary |
#
# The derived route skips only raw PDF/API preprocessing and the original array
# audit. It uses the retained historical array-floor diagnostic solely as context
# for the final summary, clearly labeled in its manifest. It is **not** used as a
# training input or freshly computed diagnostic. The original-source route runs
# the raw audit as well. Both routes write fresh `predictions.pkl` for independent
# recomputation of learned-model errors and diagnostics.
# Checkpoint writes are made atomic for reliable local resume; model calculations
# and training budgets are preserved. A resume rejects changed inputs or environments.

# %%
if FULL_RETRAIN:
    import os
    import pickle
    import shutil
    import tempfile
    from datetime import datetime, timezone

    import sklearn
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,  # noqa: F401 -- fail before costly fits
    )

    current_environment = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }

    if EXP050_INPUTS not in {"derived", "original"}:
        raise ValueError("EXP050_INPUTS must be 'derived' or 'original'.")
    if EXP050_INPUTS == "original":
        try:
            import pymupdf  # noqa: F401 -- prevent the legacy setup's pip fallback
        except ImportError as exc:
            raise ImportError(
                "Install pymupdf in this notebook's environment before the original-source route; no installation is performed here."
            ) from exc
        preflight = original_bundle_preflight(EXP050_SOURCE_BUNDLE)
        if not preflight.status.eq("MATCH").all():
            raise FileNotFoundError(
                "Original source bundle is not ready. Read the preflight above, or use the explicitly labeled derived route."
            )
    if EXP050_RESUME_WORKSPACE:
        exp050_root = Path(EXP050_RESUME_WORKSPACE).resolve()
        exp050_root.relative_to(
            RUN_DIR.resolve()
        )  # Restrict resume to this notebook's output tree.
        if not (exp050_root / "reproduction_manifest.json").exists():
            raise ValueError("Resume needs an existing notebook reproduction workspace.")
        previous_manifest = read_json(exp050_root / "reproduction_manifest.json")
        if previous_manifest["input_route"] != EXP050_INPUTS:
            raise ValueError("Refusing to mix input routes in a resumed run.")
        if any(previous_manifest.get(key) != value for key, value in current_environment.items()):
            raise ValueError(
                "The Python/library environment changed since this run. Use the prior environment or start a fresh workspace."
            )
    else:
        exp050_root = Path(tempfile.mkdtemp(prefix="exp050_retrain_", dir=RUN_DIR))
    exp050_out = exp050_root / "outputs"
    exp050_out.mkdir(exist_ok=True)
    exp050_sources = {
        name: text
        for name, text in MODULE_SOURCES.items()
        if name.startswith("full_paper/kaggle/src/")
    }
    for name, text in exp050_sources.items():
        target = exp050_root / "src" / Path(name).name
        target.parent.mkdir(exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise ValueError("The embedded implementation differs from the resumed workspace.")
        target.write_bytes(text.encode("utf-8"))

    asset_paths = [
        "data/corrected/rebuilt_cases.npy",
        "data/corrected/rebuilt_index.csv",
        "data/corrected/rebuilt_climate_era5.npy",
        "data/corrected/rebuilt_population.npy",
    ]
    if EXP050_INPUTS == "original":
        with (EXP050_SOURCE_BUNDLE / "SOURCES.csv").open(encoding="utf-8") as handle:
            original_entries = list(csv.DictReader(handle))
        current_asset_hashes = {
            "original_bundle/" + relative: hashlib.sha256(
                (EXP050_SOURCE_BUNDLE / relative).read_bytes()
            ).hexdigest()
            for relative in ["SOURCES.csv", *[row["file"] for row in original_entries]]
        }
    else:
        current_asset_hashes = {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in asset_paths
        }
    if EXP050_RESUME_WORKSPACE and previous_manifest["asset_sha256"] != current_asset_hashes:
        raise ValueError("Input hashes changed since the resumed run; start a fresh workspace.")
    if EXP050_INPUTS == "derived":
        for relative in asset_paths:
            target = exp050_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            original = REPO / relative
            if target.exists() and target.read_bytes() != original.read_bytes():
                raise ValueError("Input data changed since the resumed run; start a new workspace.")
            shutil.copy2(original, target)
    exp050_manifest = {
        "purpose": "Fresh local 810-run study; historical paper outputs are read-only",
        "input_route": EXP050_INPUTS,
        "created_utc": previous_manifest["created_utc"]
        if EXP050_RESUME_WORKSPACE
        else datetime.now(timezone.utc).isoformat(),
        "last_started_utc": datetime.now(timezone.utc).isoformat(),
        **current_environment,
        "workers_requested": WORKERS,
        "source_sha256": {name: SOURCE_SHA256[name] for name in exp050_sources},
        "asset_sha256": current_asset_hashes,
        "graph_source": "git 3878e70^:notebooks/baseline/sri_lanka_adj_list.json"
        if EXP050_INPUTS == "derived"
        else "original_bundle/graph/sri_lanka_adj_list.json",
        "retained_context_only": ["array_floor from historical results.json"]
        if EXP050_INPUTS == "derived"
        else [],
        "interpretation": "Derived route reruns current corrected inputs; not all historical date-joined scores are expected to match."
        if EXP050_INPUTS == "derived"
        else "Raw-source reconstruction using the current parser/calendar fix; historical date-joined scores can differ. See bundle hashes.",
        "runner_adaptations": [
            "local paths and workers",
            "atomic prediction checkpoints",
            "derived-route provenance labels",
        ],
    }
    # The original program flushes score rows every job but prediction checkpoints every 25.
    # On resume, rerun only score rows whose predictions were not checkpointed.
    score_log = exp050_out / "runs.jsonl"
    prediction_checkpoint = exp050_out / "predictions.pkl"
    if score_log.exists():
        checkpoint = (
            pickle.loads(prediction_checkpoint.read_bytes())
            if prediction_checkpoint.exists()
            else {}
        )
        logged = [json.loads(line) for line in score_log.read_text().splitlines() if line]
        retained = [row for row in logged if tuple(row[key] for key in keys) in checkpoint]
        if len(retained) != len(logged):
            backup = score_log.with_name("runs_before_checkpoint_reconciliation.jsonl")
            backup_number = 1
            while backup.exists():
                backup = score_log.with_name(
                    f"runs_before_checkpoint_reconciliation_{backup_number}.jsonl"
                )
                backup_number += 1
            shutil.copy2(score_log, backup)
            score_log.write_text(
                "".join(json.dumps(row) + "\n" for row in retained), encoding="utf-8"
            )
            exp050_manifest["rerun_jobs_without_prediction_checkpoint"] = len(logged) - len(
                retained
            )
            print(
                "Resume: rerunning",
                len(logged) - len(retained),
                "jobs without saved predictions; prior log was backed up.",
            )
    (exp050_root / "reproduction_manifest.json").write_text(
        json.dumps(exp050_manifest, indent=2), encoding="utf-8"
    )

    runner_prefix = """import os, sys, time, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(1)
PROFILE, QUICK = "full", False
NOTEBOOK_START = time.time()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
INPUT_ROUTE = "original"
def save_checkpoint(path, predictions):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(predictions))
    temporary.replace(path)

def execute_source(name):
    path = ROOT / "src" / name
    source = path.read_text(encoding="utf-8")
    if name == "70_experiments.py":
        # Only checkpoint I/O changes: training, seeds and epoch budgets are unchanged.
        source = source.replace("PRED_FILE.write_bytes(pickle.dumps(PREDS))",
                                "save_checkpoint(PRED_FILE, PREDS)")
    if INPUT_ROUTE == "derived" and name == "90_outputs.py":
        source = source.replace("weeks rebuilt from {len(SOURCES)} verified source files",
                                "weeks loaded from corrected arrays (raw PDFs not re-parsed)")
        source = source.replace("  benchmark array: persistence",
                                "  retained historical benchmark context: persistence")
    exec(compile(source, str(path), "exec"), globals())
"""
    if EXP050_INPUTS == "derived":
        runner_data = """
INPUT_ROUTE = "derived"
GRAPH = HISTORICAL_GRAPH
NAMES = sorted(GRAPH)
N = len(NAMES)
CASES = np.load(ROOT / "data/corrected/rebuilt_cases.npy")[..., 0].astype(np.float64)
MISSING = np.isnan(CASES).any(axis=1)
T = len(CASES)
idx = pd.read_csv(ROOT / "data/corrected/rebuilt_index.csv", parse_dates=["week_start"])
WEEK_START = idx.week_start
YEARS = idx.year.to_numpy()
CLIMATE = np.load(ROOT / "data/corrected/rebuilt_climate_era5.npy").astype(np.float64)
POPULATION = np.load(ROOT / "data/corrected/rebuilt_population.npy").astype(np.float64)
source20 = (ROOT / "src/20_data.py").read_text(encoding="utf-8")
start = source20.index('LAGS = {"cases": 1, "climate": 2}')
exec(compile(source20[start:], "derived_window_and_graph_code.py", "exec"), globals())
SOURCES = []  # No raw-source verification is claimed on this route.
ARRAY_FLOOR = RETAINED_ARRAY_FLOOR
print("DERIVED-INPUT RERUN: corrected arrays; original PDFs not re-parsed; array floor is retained context.", flush=True)
"""
        settings = f"WORKERS = {int(WORKERS)}\nHISTORICAL_GRAPH = {HISTORICAL_GRAPH!r}\nRETAINED_ARRAY_FLOOR = {list(historical_summary['array_floor'].values())!r}\n"
        runner = runner_prefix + settings + runner_data
    else:
        source00 = MODULE_SOURCES["full_paper/kaggle/src/00_setup.py"]
        source00 = source00.replace("WORKERS = 4", f"WORKERS = {int(WORKERS)}")
        source00 = source00.replace(
            "SRC = _find_sources()", f"SRC = Path({str(EXP050_SOURCE_BUNDLE.resolve())!r})"
        )
        source00 = source00.replace(
            'OUT = Path("/kaggle/working/outputs") if Path("/kaggle/working").exists() else Path.cwd() / "outputs"',
            f"OUT = Path({str(exp050_out.resolve())!r})",
        )
        runner = (
            runner_prefix + f"exec(compile({source00!r}, 'original_setup.py', 'exec'), globals())\n"
        )
        runner += "execute_source('20_data.py')\nexecute_source('30_audit.py')\n"
    runner += "\n".join(
        f"execute_source({name!r})"
        for name in [
            "40_architectures.py",
            "50_models.py",
            "55_diagnosis.py",
            "60_training.py",
            "70_experiments.py",
            "80_analysis.py",
            "90_outputs.py",
        ]
    )
    runner += "\n"
    if EXP050_INPUTS == "derived":
        runner += """
summary_path = OUT / "results.json"
derived_summary = json.loads(summary_path.read_text(encoding="utf-8"))
derived_summary["input_route"] = "derived corrected arrays, not raw-source reconstruction"
derived_summary["source_verification_performed"] = False
derived_summary["retained_array_floor_context"] = derived_summary.pop("array_floor", None)
summary_path.write_text(json.dumps(derived_summary, indent=2), encoding="utf-8")
print("DERIVED-INPUT RESULT: no original source files were verified; array-floor values above are retained context only.", flush=True)
"""
    runner_path = exp050_root / "run_development.py"
    runner_path.write_text(runner, encoding="utf-8")
    compile(runner, str(runner_path), "exec")
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    log_path = exp050_root / "training.log"
    print("Fresh development workspace:", exp050_root)
    print("Live log:", log_path)
    print("On Windows the original process uses one worker; allow several hours.")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(runner_path)],
            cwd=exp050_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                if any(word in line for word in ["runs", "finished", "DERIVED-INPUT", "SUMMARY"]):
                    print(line.rstrip(), flush=True)
            status = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if status != 0:
        print(log_path.read_text(encoding="utf-8")[-6000:])
        raise subprocess.CalledProcessError(status, [sys.executable, str(runner_path)])
    fresh_trained = pd.DataFrame(
        [json.loads(line) for line in (exp050_out / "runs.jsonl").read_text().splitlines() if line]
    )
    assert len(fresh_trained) == 810 and not fresh_trained.duplicated(keys).any()
    fresh_runs = pd.DataFrame(read_json(exp050_out / "runs.json"))
    fresh_three = fresh_runs[fresh_runs.origin_set.eq("three")]
    fresh_comparison = pd.DataFrame(
        [
            {
                "configuration": encoder + "+" + head,
                "historical test RMSE": mean_score(r3, encoder + "+" + head, "RMSE"),
                "fresh test RMSE": mean_score(fresh_three, encoder + "+" + head, "RMSE"),
            }
            for encoder in ENCODERS
            for head in ["direct", "gated", "foi_res"]
        ]
    )
    fresh_comparison["difference"] = (
        fresh_comparison["fresh test RMSE"] - fresh_comparison["historical test RMSE"]
    )
    display(fresh_comparison)
    fresh_comparison.to_csv(exp050_root / "fresh_vs_historical.csv", index=False)
    print("Fresh prediction archive:", exp050_out / "predictions.pkl")
else:
    print(
        "Skipped the 810-run development study. All earlier displayed model scores remain historical."
    )

# %% [markdown]
# ## Finish: what to keep with your review
#
# The notebook itself contains this explanation, calculations, implementation
# snapshots and the displayed outputs from the default audit. The project provides
# the local data and retained experiment artifacts. Generated CSVs and the
# rechecked number-source report are under `runs/icitr_notebook/`.
#
# If you enable full training, keep each fresh workspace's manifest, run logs and
# predictions. Those distinguish a new local experiment from a recalculation of
# old scores. Differences should be explained before changing paper values.
# The original source PDFs required for the complete raw rebuild are not all
# available in this checkout; the notebook reports that limitation openly.

# %%
if not failures.empty:
    raise AssertionError(
        f"{len(failures)} calculation checks disagreed. Inspect calculation_checks.csv before trusting the report."
    )
print("Notebook calculations completed. Historical paper files were not edited.")
print("Audit report:", RUN_DIR / "calculation_checks.csv")
print("Optional fresh training enabled:", FULL_RETRAIN)
