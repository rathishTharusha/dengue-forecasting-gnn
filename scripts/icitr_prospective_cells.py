"""Build the optional, isolated EXP-063 retraining section of the paper notebook.

This file is a notebook-generation helper, not a second experiment implementation.
The emitted cells run snapshots of the repository's original command-line pipeline.
"""

from __future__ import annotations

from textwrap import dedent

PROSPECTIVE_SOURCE_PATHS = (
    "seirgnn2/prospective.py",
    "seirgnn2/classical.py",
    "seirgnn2/core.py",
    "seirgnn2/sweep.py",
    "seirgnn2/train.py",
    "seirgnn2/models.py",
    "seirgnn2/backbones.py",
    "seirgnn2/augment.py",
    "seirgnn2/knn.py",
    "seirgnn2/gbm.py",
    "analysis/lib/corrected_data.py",
    "analysis/lib/count_loss.py",
    "analysis/lib/physics_loss.py",
    "analysis/lib/seir_sim.py",
    "analysis/_build/population_vintages.py",
    "src/dengue_gnn/__init__.py",
    "src/dengue_gnn/metrics.py",
)

PROSPECTIVE_ASSET_PATHS = (
    "data/corrected/rebuilt_index.csv",
    "data/corrected/rebuilt_cases.npy",
    "data/corrected/rebuilt_climate_era5.npy",
    "data/corrected/rebuilt_climate_era5.json",
    "data/corrected/rebuilt_population.npy",
    "data/corrected/rebuilt_population_sources.csv",
    "data/external/modis_ndvi_weekly_by_district.csv",
    "data/external/population_vintages.csv",
    "data/new_weeks/cases.npy",
    "data/new_weeks/index.csv",
    "notebooks/baseline/sri_lanka_adj_list.json",
    "seirgnn2/results/prospective_frozen.json",
)


def _markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip()}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": dedent(source).strip(),
        "execution_count": None,
        "outputs": [],
    }


def prospective_cells() -> list[dict]:
    """Return cells expecting REPO, RUN_DIR, FULL_RETRAIN and MODULE_SOURCES.

    MODULE_SOURCES maps repository-relative Python paths to their source text.
    It must contain PROSPECTIVE_SOURCE_PATHS; other embedded sources are ignored.
    FULL_RETRAIN=False executes without importing torch or creating a workspace.
    """
    return [
        _markdown(
            """
            ## 13. Optional: train EXP-063 again from the corrected data

            Set `FULL_RETRAIN = True` in the configuration cell, restart the kernel,
            and run this notebook from the top to enable this section. It runs the
            **original development grid (324 neural fits), validation-only selection,
            final evaluation (9 neural fits plus 3 classical baselines), and statistics**.
            The historical CPU run took about 19 minutes for development tuning;
            runtime on this computer may differ. There is no shortened-epoch substitute.

            The code uses the implementation snapshots embedded in this notebook and
            copies the necessary data into a fresh directory below `RUN_DIR`.
            Python, NumPy, pandas, PyTorch, and the Git executable must already be
            installed locally. No data download or package installation is performed.
            The adaptive models used here do not require PyG.

            A rerun of validation selection is saved for comparison. The final models
            always use the **historically frozen settings**: learning rate 0.003 and
            hidden width 64 for every neural arm, AR(3) alpha 0.01. Different library
            versions or numeric kernels can produce slightly different neural results;
            differences are reported rather than silently replacing paper results.

            This reproduces the **amended run 2** on the existing corrected new-week
            files. It does not re-extract the source PDFs or recreate run 1's defective
            dataset. The amendments were made after run 1 was scored. Rerunning today
            also cannot recreate the historical fact of preregistration or an untouched
            test period. The isolated Git commits below satisfy the original program's
            file-consistency guard; they are clearly recorded as reproduction commits.
            """
        ),
        _code(
            """
            # This cell prepares a private copy only when full retraining is enabled.
            import hashlib
            import json
            import os
            import platform
            import shutil
            import subprocess
            import sys
            import tempfile
            from datetime import datetime, timezone
            from pathlib import Path

            PROSPECTIVE_WORKERS = max(1, int(WORKERS))
            exp063_root = None

            if FULL_RETRAIN:
                import numpy as np
                import pandas as pd
                import torch

                if shutil.which("git") is None:
                    raise RuntimeError("Git must be installed before using FULL_RETRAIN.")
                required_sources = __SOURCE_PATHS__
                required_assets = __ASSET_PATHS__
                absent_sources = [p for p in required_sources if p not in MODULE_SOURCES]
                absent_assets = [p for p in required_assets if not (REPO / p).is_file()]
                if absent_sources or absent_assets:
                    raise FileNotFoundError(
                        f"Missing embedded sources: {absent_sources}; missing local assets: {absent_assets}"
                    )

                RUN_DIR.mkdir(parents=True, exist_ok=True)
                exp063_root = Path(tempfile.mkdtemp(prefix="exp063_retrain_", dir=RUN_DIR)).resolve()
                for relative in required_sources:
                    target = exp063_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(MODULE_SOURCES[relative].encode("utf-8"))
                for relative in required_assets:
                    target = exp063_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(REPO / relative, target)

                exp063_env = os.environ.copy()
                exp063_env.update(
                    PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
                )
                exp063_env.pop("PYTHONPATH", None)
                for git_variable in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"):
                    exp063_env.pop(git_variable, None)

                def exp063_git(*arguments):
                    completed = subprocess.run(
                        ["git", "-c", "user.name=Notebook reproduction",
                         "-c", "user.email=notebook-reproduction@localhost",
                         "-c", "commit.gpgsign=false", *arguments],
                        cwd=exp063_root, env=exp063_env, check=True,
                        capture_output=True, text=True, encoding="utf-8",
                    )
                    return completed.stdout.strip()

                # Only this fresh private repository is initialized or committed.
                exp063_git("init", "--quiet")
                exp063_git("add", "--", *required_sources)
                exp063_git("commit", "--quiet", "-m", "Notebook reproduction: embedded source snapshot")

                exp063_results = exp063_root / "seirgnn2/results"
                exp063_logs = exp063_root / "logs"
                exp063_logs.mkdir()
                exp063_historical_frozen_bytes = (
                    exp063_results / "prospective_frozen.json"
                ).read_bytes()
                exp063_historical_frozen = json.loads(exp063_historical_frozen_bytes)

                def exp063_sha256(path):
                    return hashlib.sha256(path.read_bytes()).hexdigest()

                exp063_manifest = {
                    "purpose": "Reproduction of amended EXP-063 run 2; not a new prospective test",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source_repository": str(REPO.resolve()),
                    "workspace": str(exp063_root),
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
                    "workers": PROSPECTIVE_WORKERS,
                    "source_sha256": {p: exp063_sha256(exp063_root / p) for p in required_sources},
                    "asset_sha256": {p: exp063_sha256(exp063_root / p) for p in required_assets},
                    "historical_selected_at_commit": exp063_historical_frozen["selected_at_commit"],
                    "local_source_commit": exp063_git("rev-parse", "HEAD"),
                    "commands": [],
                }

                def exp063_write_manifest():
                    (exp063_root / "reproduction_manifest.json").write_text(
                        json.dumps(exp063_manifest, indent=2), encoding="utf-8"
                    )

                def exp063_run_command(stage, command):
                    log_path = exp063_logs / f"{stage}.log"
                    print(f"Running {stage}; full output: {log_path}", flush=True)
                    exp063_manifest["commands"].append({"stage": stage, "argv": command})
                    exp063_write_manifest()
                    with log_path.open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(
                            command, cwd=exp063_root, env=exp063_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                        )
                        completed_jobs = 0
                        try:
                            for line in process.stdout:
                                log.write(line)
                                log.flush()
                                if line.startswith("["):
                                    completed_jobs += 1
                                    if completed_jobs % 27 == 0 or "FAILED" in line:
                                        print(line.strip(), flush=True)
                            return_code = process.wait()
                        except BaseException:
                            process.terminate()
                            process.wait()
                            raise
                    if return_code:
                        print(log_path.read_text(encoding="utf-8")[-6000:])
                        raise subprocess.CalledProcessError(return_code, command)
                    print(f"Completed {stage}", flush=True)

                def exp063_cli(step):
                    exp063_run_command(
                        step, [sys.executable, "-u", "seirgnn2/prospective.py", step,
                               "--workers", str(PROSPECTIVE_WORKERS)]
                    )

                exp063_write_manifest()
                # Fail before training if a captured import dependency is missing.
                exp063_run_command(
                    "imports",
                    [sys.executable, "-u", "-c",
                     "import sys; sys.path.insert(0, 'seirgnn2'); "
                     "import prospective, train; "
                     "sys.path.insert(0, 'analysis/_build'); import population_vintages"],
                )
                print(f"Private workspace ready: {exp063_root}")
            else:
                print("Full retraining is disabled. The saved-result audit above is the default run.")
            """.replace("__SOURCE_PATHS__", repr(PROSPECTIVE_SOURCE_PATHS)).replace(
                "__ASSET_PATHS__", repr(PROSPECTIVE_ASSET_PATHS)
            )
        ),
        _markdown(
            """
            ### Development: tuning and selection

            Each neural arm receives the same four settings on nine purged origins
            and seeds 0, 1, 2: `3 × 4 × 9 × 3 = 324` fits. The original pipeline
            also evaluates persistence, seasonal naive and five AR(3) penalties.
            Selection sees mean **validation RMSE only**. The checks below reject
            an incomplete grid before selection, because a failed worker must not
            silently reduce the tuning budget.

            `prospective_frozen_reselected.json` records this computer's new choice.
            For the final evaluation, the original paper's frozen file is restored
            and committed inside the private workspace. A different new choice is
            reported, but does not change the reproduced final experiment.
            """
        ),
        _code(
            """
            if FULL_RETRAIN:
                exp063_cli("dev")
                dev_rows = pd.DataFrame(json.loads(
                    (exp063_results / "prospective_dev.json").read_text(encoding="utf-8")
                ))
                classical_rows = pd.DataFrame(json.loads(
                    (exp063_results / "prospective_dev_classical.json").read_text(encoding="utf-8")
                ))
                expected_origins = {round(0.50 + 0.05 * k, 2) for k in range(9)}
                expected_arms = set(exp063_historical_frozen["arms"])
                expected_names = {
                    f"{arm} | lr={lr:g} hidden={hidden}"
                    for arm in expected_arms for lr in (0.001, 0.003) for hidden in (32, 64)
                }
                neural_rows = dev_rows[dev_rows["seed"] >= 0]
                actual_names = set(neural_rows["name"])
                assert actual_names == expected_names, "The development grid is incomplete."
                assert len(neural_rows) == 324
                assert not neural_rows.duplicated(["name", "origin", "seed"]).any()
                assert np.isfinite(neural_rows[["val_RMSE", "RMSE"]].to_numpy()).all()
                expected_pairs = {(origin, seed) for origin in expected_origins for seed in (0, 1, 2)}
                for name, rows in neural_rows.groupby("name"):
                    assert set(zip(rows["origin"], rows["seed"])) == expected_pairs, name
                persistence_rows = dev_rows[dev_rows["name"] == "persistence"]
                assert set(persistence_rows["origin"]) == expected_origins and len(persistence_rows) == 9
                expected_classical = {"Seasonal naive"} | {
                    f"AR(3) ridge | alpha={alpha:g}" for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
                }
                assert set(classical_rows["name"]) == expected_classical and len(classical_rows) == 54
                assert not classical_rows.duplicated(["name", "origin"]).any()
                assert np.isfinite(classical_rows[["val_RMSE", "RMSE"]].to_numpy()).all()
                for name, rows in classical_rows.groupby("name"):
                    assert set(rows["origin"]) == expected_origins, name

                exp063_cli("select")
                reselected_path = exp063_results / "prospective_frozen_reselected.json"
                shutil.copy2(exp063_results / "prospective_frozen.json", reselected_path)
                reselected = json.loads(reselected_path.read_text(encoding="utf-8"))
                settings_match = (
                    reselected["arms"] == exp063_historical_frozen["arms"]
                    and reselected["ar_alpha"] == exp063_historical_frozen["ar_alpha"]
                )
                exp063_manifest["rerun_selection_matches_historical_settings"] = settings_match
                exp063_manifest["rerun_selected_settings"] = {
                    "arms": reselected["arms"], "ar_alpha": reselected["ar_alpha"]
                }
                print("New validation selection matches historical settings:", settings_match)
                selection_table = pd.DataFrame({
                    "historical validation RMSE": exp063_historical_frozen["dev_val_RMSE"],
                    "rerun validation RMSE": reselected["dev_val_RMSE"],
                })
                print(selection_table.round(6).to_string())
                selection_table.to_csv(exp063_root / "development_selection_comparison.csv")

                (exp063_results / "prospective_frozen.json").write_bytes(exp063_historical_frozen_bytes)
                exp063_git("add", "--", "seirgnn2/results/prospective_frozen.json")
                exp063_git("commit", "--quiet", "-m", "Notebook reproduction: historical frozen settings")
                exp063_manifest["local_frozen_commit"] = exp063_git("rev-parse", "HEAD")
                exp063_manifest["final_uses_historical_settings"] = True
                exp063_write_manifest()
            else:
                print("Skipped the 324-fit development grid and selection.")
            """
        ),
        _markdown(
            """
            ### Final evaluation and uncertainty

            The private copy of the original `final` command trains each of the three
            neural arms once per seed, using the purged final split and validation
            early stopping. It evaluates all six arms on identical surviving test
            starts. The `stats` command then performs the paired circular block
            bootstrap (10,000 replicates, primary block length 8) and Holm adjustment
            over the two primary comparisons. Block lengths 4 and 12 are sensitivity
            analyses. All fresh output stays in the private workspace.
            """
        ),
        _code(
            r"""
            if FULL_RETRAIN:
                exp063_cli("final")
                exp063_cli("stats")
                exp063_fresh = json.loads(
                    (exp063_results / "prospective_final.json").read_text(encoding="utf-8")
                )
                exp063_fresh_stats = json.loads(
                    (exp063_results / "prospective_stats.json").read_text(encoding="utf-8")
                )
                fresh_rows = pd.DataFrame(exp063_fresh["rows"])
                expected_final_keys = {
                    (arm, seed) for arm in exp063_historical_frozen["arms"] for seed in (0, 1, 2)
                } | {(arm, -1) for arm in ("Persistence", "Seasonal naive", "AR(3) ridge")}
                assert len(fresh_rows) == 12
                assert set(zip(fresh_rows["name"], fresh_rows["seed"])) == expected_final_keys
                assert not fresh_rows.duplicated(["name", "seed"]).any()
                assert np.isfinite(fresh_rows[["RMSE", "MAE", "SMAPE", "MAPE"]].to_numpy()).all()
                assert exp063_fresh["frozen"] == exp063_historical_frozen
                assert set(exp063_fresh["losses"]) == set(fresh_rows["name"])
                assert all(len(losses) == exp063_fresh["n_test"] for losses in exp063_fresh["losses"].values())

                historical_final = json.loads(
                    (REPO / "seirgnn2/results/prospective_final.json").read_text(encoding="utf-8")
                )
                assert exp063_fresh["test_starts"] == historical_final["test_starts"], (
                    "Local data produce different test starts; compare the captured asset hashes."
                )
                historical_rows = pd.DataFrame(historical_final["rows"])
                metrics = [column for column in fresh_rows.columns if column not in ("name", "seed")]
                rerun_means = fresh_rows.groupby("name")[metrics].mean()
                historical_means = historical_rows.groupby("name")[metrics].mean()
                comparison = pd.DataFrame({
                    "historical RMSE": historical_means["RMSE"],
                    "rerun RMSE": rerun_means["RMSE"],
                    "RMSE difference": rerun_means["RMSE"] - historical_means["RMSE"],
                    "historical MAE": historical_means["MAE"],
                    "rerun MAE": rerun_means["MAE"],
                }).sort_values("historical RMSE")
                print(comparison.round(6).to_string())
                print("\nFresh primary comparisons:")
                print(pd.DataFrame(exp063_fresh_stats["primary"]).round(6).to_string(index=False))
                fresh_rows.to_csv(exp063_root / "final_metrics_per_seed.csv", index=False)
                comparison.to_csv(exp063_root / "final_metrics_comparison.csv")
                exp063_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
                exp063_manifest["n_test"] = exp063_fresh["n_test"]
                exp063_manifest["outputs"] = {
                    "development": "seirgnn2/results/prospective_dev.json",
                    "development_classical": "seirgnn2/results/prospective_dev_classical.json",
                    "reselected_settings": "seirgnn2/results/prospective_frozen_reselected.json",
                    "historical_settings_used": "seirgnn2/results/prospective_frozen.json",
                    "final": "seirgnn2/results/prospective_final.json",
                    "statistics": "seirgnn2/results/prospective_stats.json",
                }
                exp063_write_manifest()
                print(f"\nReproduction complete. Outputs and logs: {exp063_root}")
            else:
                print("Skipped fresh final-model training and its bootstrap; saved results remain available above.")
            """
        ),
    ]
