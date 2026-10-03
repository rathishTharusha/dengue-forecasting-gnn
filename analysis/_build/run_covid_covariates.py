"""Exploratory COVID covariate ablation; independent of the spatial runner.

Frozen v2 STGAT, 9 origins x 3 seeds, 400 epochs, validation early stopping.
Arms: unchanged v2; policy multiplier on force of infection; policy + mobility.
Latest external row is i-2. Missing external measurements disable their term,
not the case window; this is an explicit model-level neutral fallback, not
imputation into source data. Archived values are not verified historical vintages.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "analysis/lib", REPO / "analysis/_build", REPO / "src"):
    sys.path.insert(0, str(p))
import run_s5_seir_gnn_v2 as v2
import seir_sim

OriginalModel = v2.SEIRGNNv2
ACTIVE_ARM = "base"
MODEL = None
ARMS = ("base", "policy", "policy_mobility")


def external_inputs(frame, dates):
    """Fixed physical scaling, no fitting to test data; explicit missing gate."""
    frame = frame.set_index("week_start").reindex(pd.DatetimeIndex(dates))
    raw = frame[["stringency_index", "mobility_workplaces"]].to_numpy(float)
    values = raw / np.array([100.0, -100.0])
    inputs = np.zeros_like(values, dtype=np.float32)
    available = np.zeros_like(values, dtype=bool)
    # At origin i the most recent permitted *completed* external week is i-2.
    for i in range(2, len(values)):
        available[i] = np.isfinite(values[i - 2])
        inputs[i, available[i]] = values[i - 2, available[i]]
    return inputs, available


class CovidModel(OriginalModel):
    def __init__(self, arch_name, in_dim, *args, **kwargs):
        super().__init__(arch_name, 1, *args, **kwargs)
        global MODEL
        MODEL = self
        self.arm = ACTIVE_ARM
        # Zero initial effect; an unseen exposure cannot obtain a random effect.
        if self.arm != "base":
            self.covid_weight = torch.nn.Parameter(torch.zeros(2))
        self.last_prediction = None

    def forward(self, x, st0):
        h = self.backbone(self.in_proj(x[..., :1]).squeeze(-1), self.edge_index)
        z = self.head_fc(h)
        beta = torch.nn.functional.softplus(self.beta0) * torch.exp(
            z.clamp(-self.mod_clamp, self.mod_clamp))
        if self.arm != "base":
            exposure = x[:, :, -1, 1:3]
            weights = self.covid_weight
            if self.arm == "policy":
                weights = weights * weights.new_tensor([1.0, 0.0])
            shift = (exposure * weights).sum(-1).clamp(-1.5, 1.5)
            beta = beta * torch.exp(shift.unsqueeze(-1))
        lam = beta * st0[..., 2].unsqueeze(-1)
        omega, gamma = self.rates()
        _, inc = seir_sim.simulate_weeks(st0, lam, omega, gamma, substeps=7)
        return inc

    def to_cases(self, inc, pop):
        result = super().to_cases(inc, pop)
        self.last_prediction = result.detach().cpu().numpy()
        return result


def run_job(job):
    global ACTIVE_ARM
    torch.set_num_threads(1)
    ACTIVE_ARM = job["arm"]
    v2.SEIRGNNv2 = CovidModel
    rec = v2.run_job(job)
    model = MODEL
    idx = job["test_idx"]
    truth = np.stack([job["cases"][i:i + v2.HORIZON].T for i in idx])
    pred = model.last_prediction
    pers = np.repeat(job["cases"][idx - 1, :, None], v2.HORIZON, axis=2)
    rec.update(bias=float((pred - truth).mean()),
               pers_RMSE=float(np.sqrt(np.mean((pers - truth) ** 2))),
               pers_MAE=float(np.abs(pers - truth).mean()),
               train_policy_windows=int(job["available"][job["train_idx"], 0].sum()),
               train_mobility_windows=int(job["available"][job["train_idx"], 1].sum()),
               test_policy_windows=int(job["available"][idx, 0].sum()),
               test_mobility_windows=int(job["available"][idx, 1].sum()))
    if job["arm"] != "base":
        rec["policy_weight"], rec["mobility_weight"] = model.covid_weight.detach().tolist()
    stem = f"{job['arm']}_o{job['origin']}_s{job['seed']}"
    np.savez_compressed(Path(job["out_dir"]) / f"{stem}.npz", origin_index=idx,
                        prediction=pred, truth=truth, persistence=pers)
    return rec


def paired_p(delta):
    delta = np.asarray(delta)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(delta))))
    return float((np.abs((signs * delta).mean(1)) >= abs(delta.mean()) - 1e-12).mean())


def summarize(records, out):
    df = pd.DataFrame(records)
    df.to_csv(out / "runs.csv", index=False)
    base = df[df.arm.eq("base")].set_index(["origin", "seed"])
    rows = []
    for arm in ARMS:
        g = df[df.arm.eq(arm)].set_index(["origin", "seed"])
        if len(g) != 27:
            continue
        delta = (g.test_RMSE - base.test_RMSE).groupby(level="origin").mean()
        rows.append(dict(arm=arm, val_RMSE=g.val_RMSE.mean(),
                         test_RMSE=g.test_RMSE.mean(), test_MAE=g.test_MAE.mean(),
                         bias=g.bias.mean(), delta_vs_base=delta.mean(),
                         origins_won=int((delta < 0).sum()),
                         p_vs_base=paired_p(delta.to_numpy())))
    table = pd.DataFrame(rows)
    if len(table) == 3:
        tests = table.index[table.arm.ne("base")].tolist()
        ordered = sorted(tests, key=lambda k: table.loc[k, "p_vs_base"])
        running = 0.0
        for rank, k in enumerate(ordered):
            running = max(running, min(1.0, (2 - rank) * table.loc[k, "p_vs_base"]))
            table.loc[k, "p_holm"] = running
    table.to_csv(out / "summary.csv", index=False)
    per_origin = df.groupby(["arm", "origin"])[
        ["test_RMSE", "test_MAE", "bias", "pers_RMSE", "train_policy_windows",
         "train_mobility_windows", "test_policy_windows", "test_mobility_windows"]].mean()
    per_origin.to_csv(out / "per_origin.csv")
    report = "# COVID covariate experiment\n\nExploratory, retrospective evaluation; "
    report += "archived input vintages are unverified. Means are across origins and seeds.\n\n"
    report += "```text\n" + table.to_string(index=False) + "\n```\n\n"
    report += "Per-origin scores and input coverage:\n\n```text\n"
    report += per_origin.to_string() + "\n```\n\n"
    report += "The first lockdown cannot be learned at an origin with no policy exposure in training. "
    report += "Missing external observations have neutral model effects. "
    report += "P-values are sign flips clustered by origin; Holm adjusts two versus-base comparisons. "
    report += "These are exploratory tests after previous model searches, not independent confirmation.\n"
    (out / "report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO / "analysis/results/covid_covariates")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = v2.cd.load()
    cases, adj, artifact, missing, _ = v2.rcb.prepare("rebuilt")
    source = REPO / "data/external/covid_response_weekly.csv"
    frame = pd.read_csv(source, parse_dates=["week_start"])
    external, available = external_inputs(frame, data.week_start)
    folds = v2.build_folds(cases, missing, "9origin")
    edge_index = np.stack(np.nonzero(adj))
    # Freeze config and provenance before training; no test-led tuning.
    metadata = dict(commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        torch=torch.__version__, arms=ARMS, seeds=[0, 1, 2], lag_weeks=2,
        protocol="9origin", epochs=400, exploratory=True,
        vintage_verified=False, missing_external="neutral multiplier, model-level gate")
    metadata["smoke"] = args.smoke
    config_path = args.out / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != json.loads(json.dumps(metadata)):
            raise SystemExit("Output config differs; use a new --out directory rather than mix runs")
    else:
        config_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    jobs = []
    for fold in folds:
        f = v2.features_train_only(data, "cases", fold)
        ex = np.broadcast_to(external[:, None, None, :], (*f.shape[:3], 2))
        features = np.concatenate([f, ex], axis=-1).copy()
        for arm in ARMS:
            for seed in (0, 1, 2):
                jobs.append(dict(arm=arm, cfg=dict(v2.ARMS["v2"]), arch="STGAT",
                    seed=seed, dataset="rebuilt", artifact=artifact, coupling="implicit",
                    origin=fold.origin, train_idx=np.asarray(fold.train_index),
                    val_idx=np.asarray(fold.val_index), test_idx=np.asarray(fold.test_index),
                    cases=cases, features=features, population=data.population,
                    edge_index=edge_index, adj_dense=adj, available=available,
                    out_dir=str(args.out)))
    if args.smoke:
        jobs = jobs[:1]
        jobs[0]["cfg"]["epochs"] = 2
    records_path = args.out / "runs.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    done = {(r["arm"], r["origin"], r["seed"]) for r in records}
    jobs = [j for j in jobs if (j["arm"], j["origin"], j["seed"]) not in done]
    print(f"{len(jobs)} remaining jobs; {args.workers} CPU workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_job, j) for j in jobs]):
            records.append(future.result())  # fail loudly; never hide a missing arm
            records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            pd.DataFrame(records).to_csv(args.out / "runs.csv", index=False)
            print(f"Saved {len(records)} runs", flush=True)
    if not args.smoke:
        summarize(records, args.out)


if __name__ == "__main__":
    main()
