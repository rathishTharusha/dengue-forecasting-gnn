"""Build the paper tables that the reproduction notebook does not write.

``build_paper_assets.py`` builds every table that comes from the Kaggle run (EXP-050).
Three tables the Phase-3 brief asks for need other sources, so they are built here,
from files only, into ``full_paper/overleaf/tables_extra.tex``:

* ``tab:tuning``      validation RMSE of the EXP-063 development grid
                      (``seirgnn2/results/prospective_frozen.json``)
* ``tab:compute``     parameter counts from the notebook's own model code
                      (``full_paper/kaggle/src/40_architectures.py``, ``50_models.py``) and the
                      per-run wall time the Kaggle run recorded (``runs.jsonl``, ``elapsed``)
* ``tab:prospective`` the amended prospective evaluation, run 2 of EXP-063
                      (``prospective_final.json``, ``prospective_stats.json``)

Needs torch (for the parameter counts), so it is not part of CI.

    python scripts/build_paper_extra_tables.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "full_paper" / "outputs" / "kaggle_run"
SRC = REPO / "full_paper" / "kaggle" / "src"
RES = REPO / "seirgnn2" / "results"
OUT = REPO / "full_paper" / "overleaf" / "tables_extra.tex"
ENC = ["AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN"]

# Paper names for the EXP-063 arms. The gated non-SEIR arm is PAGE on the adaptive encoder.
ARMS = {
    "Persistence": "Persistence",
    "Adaptive gated non-SEIR": r"\model{} (adaptive graph)",
    "Adaptive SEIR-GNN": r"\model{} + SEIR branch",
    "Adaptive residual": "Residual head (no gate)",
    "AR(3) ridge": "AR(3) ridge",
    "Seasonal naive": "Seasonal naive",
}


def model_namespace() -> dict:
    """Execute the notebook's architecture and model cells, with the benchmark graph."""
    ns: dict = {"__name__": "paper_tables"}
    exec((SRC / "40_architectures.py").read_text(encoding="utf-8"), ns)
    adj = json.loads(
        (REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text(encoding="utf-8")
    )
    names = list(adj)
    a = np.eye(len(names), dtype=np.float32)
    for d, nbrs in adj.items():
        for o in nbrs:
            if o in names:
                a[names.index(d), names.index(o)] = 1.0
    src, dst = np.nonzero(a)
    ns["EDGE_INDEX"] = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    exec((SRC / "50_models.py").read_text(encoding="utf-8"), ns)
    return ns


def params(ns: dict, encoder: str, head: str) -> int:
    net = ns["Net"](3, 25, backbone=encoder, head=head)
    return sum(p.numel() for p in net.parameters())


def tuning_table() -> str:
    frozen = json.loads((RES / "prospective_frozen.json").read_text(encoding="utf-8"))
    val = frozen["dev_val_RMSE"]
    arms = ["Adaptive gated non-SEIR", "Adaptive SEIR-GNN", "Adaptive residual"]
    settings = [(1e-3, 32), (1e-3, 64), (3e-3, 32), (3e-3, 64)]
    best = {arm: min(val[f"{arm} | lr={lr:g} hidden={h}"] for lr, h in settings) for arm in arms}
    lines = []
    for lr, h in settings:
        cells = []
        for arm in arms:
            v = val[f"{arm} | lr={lr:g} hidden={h}"]
            cells.append(rf"\textbf{{{v:.2f}}}" if v == best[arm] else f"{v:.2f}")
        lr_tex = r"$1\times10^{-3}$" if lr == 1e-3 else r"$3\times10^{-3}$"
        lines.append(f"{lr_tex} & {h} & " + " & ".join(cells) + r" \\")
    alphas = sorted(k for k in val if k.startswith("AR(3)"))
    ar = ", ".join(f"{k.split('alpha=')[1]}: {val[k]:.2f}" for k in alphas)
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Hyperparameter tuning: mean validation RMSE over the nine purged development origins"
            r" $\times$ 3 seeds (108 runs per arm). Bold: the setting each arm selects. The same grid and budget"
            r" are given to every learned arm; test data play no part. AR(3) ridge, by $\alpha$: "
            + ar
            + ".}",
            r"\label{tab:tuning}",
            r"\small",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"learning rate & hidden $d$ & \model{} & \model{}+SEIR & residual \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def compute_table(ns: dict) -> str:
    rows = [
        json.loads(line)
        for line in (RUN / "runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    by = defaultdict(list)
    for r in rows:
        if r.get("origin_set") == "three" and "epochs_ran" in r:
            by[r["name"]].append(r)

    def sec_per_epoch(name: str) -> float:
        v = by[name]
        return sum(r["elapsed"] for r in v) / sum(r["epochs_ran"] for r in v)

    def run_time(name: str) -> float:
        return float(np.mean([r["elapsed"] for r in by[name]]))

    lines = []
    for e in ENC:
        p_direct, p_gated, p_seir = (
            params(ns, e, "direct"),
            params(ns, e, "gated"),
            params(ns, e, "foi_res"),
        )
        lines.append(
            f"{e} & {p_direct:,} & +{p_gated - p_direct} & +{p_seir - p_direct} & "
            f"{sec_per_epoch(e + '+direct'):.2f} & {sec_per_epoch(e + '+gated'):.2f} & "
            f"{sec_per_epoch(e + '+foi_res'):.2f} & {run_time(e + '+gated'):.0f}" + r" \\"
        )
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Computational cost. Parameters of the full model behind the direct head, and the"
            r" parameters each head adds. Seconds per epoch and seconds per training run (with early"
            r" stopping) are means over the 9 runs of the Kaggle run, each run single-threaded on a 4-core"
            r" Kaggle CPU. \model{} adds one parameter and no measurable time; the SEIR branch adds five"
            r" parameters and about $2\times$ the time per epoch on the light encoders.}",
            r"\label{tab:compute}",
            r"\small",
            r"\setlength{\tabcolsep}{2.2pt}",
            r"\begin{tabular}{lrrrrrrr}",
            r"\toprule",
            r" & \multicolumn{3}{c}{parameters} & \multicolumn{3}{c}{s / epoch} & s / run \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
            r"encoder & direct & \model{} & +SEIR & direct & \model{} & +SEIR & \model{} \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test on the mean of paired differences."""
    obs = abs(d.mean())
    signs = np.array(np.meshgrid(*[[1.0, -1.0]] * len(d))).reshape(len(d), -1).T
    return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-12))


def bh(p: list[float]) -> list[float]:
    p = np.asarray(p)
    order = np.argsort(p)
    adj = p[order] * len(p) / np.arange(1, len(p) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()


def paired_table() -> str:
    """PAGE against the published direct head and against persistence, per encoder.

    Pairs are matched (origin, seed) units of the Kaggle run (3 origins x 3 seeds); persistence
    has no seed, so its value at the origin is paired with each seed. BH across the table.
    """
    rows = json.loads((RUN / "runs.json").read_text(encoding="utf-8"))
    r3 = {(r["name"], r["origin"], r["seed"]): r for r in rows if r["origin_set"] == "three"}
    pers = {
        r["origin"]: r for r in rows if r["origin_set"] == "three" and r["name"] == "persistence"
    }
    units = [(o, s) for o in (0.55, 0.7, 0.85) for s in (0, 1, 2)]
    cells, pvals = [], []
    for e in ENC:
        for ref in ("direct", "persistence"):
            for metric in ("val_RMSE", "RMSE"):
                d = np.array(
                    [
                        r3[(f"{e}+gated", o, s)][metric]
                        - (pers[o] if ref == "persistence" else r3[(f"{e}+{ref}", o, s)])[metric]
                        for o, s in units
                    ]
                )
                cells.append((e, ref, metric, d.mean(), int((d < 0).sum())))
                pvals.append(sign_flip_p(d))
    padj = bh(pvals)
    lines = []
    for i, e in enumerate(ENC):
        parts = []
        for j in range(4):
            _, _, _, mu, wins = cells[4 * i + j]
            p = padj[4 * i + j]
            star = "$^{*}$" if p < 0.05 else ""
            parts.append(f"${mu:+.2f}$ {wins}/9{star}")
        lines.append(f"{e} & " + " & ".join(parts) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{\model{} against the published direct head and against persistence, on matched"
            r" (origin, seed) runs of Table~\ref{tab:encoders}. Mean RMSE difference (negative favours \model{})"
            r" and wins out of 9. $^{*}$: $p_{\text{adj}}<0.05$, exact sign-flip test, BH across the table"
            r" (attainable floor 0.004).}",
            r"\label{tab:paired}",
            r"\small",
            r"\setlength{\tabcolsep}{2pt}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r" & \multicolumn{2}{c}{vs direct (published)} & \multicolumn{2}{c}{vs persistence} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"encoder & val & test & val & test \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def prospective_table() -> str:
    final = json.loads((RES / "prospective_final.json").read_text(encoding="utf-8"))
    stats = json.loads((RES / "prospective_stats.json").read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in final["rows"]:
        by[r["name"]].append(r)
    best = {
        m: min(float(np.mean([r[m] for r in v])) for v in by.values())
        for m in ("RMSE", "MAE", "SMAPE", "MAPE")
    }

    def cell(v: list, m: str, nd: int) -> str:
        x = float(np.mean([r[m] for r in v]))
        s = f"{x:.{nd}f}"
        return rf"\textbf{{{s}}}" if round(x, 6) == round(best[m], 6) else s

    lines = []
    for key, label in ARMS.items():
        v = by[key]
        rng = ""
        if len(v) > 1:
            lo, hi = min(r["RMSE"] for r in v), max(r["RMSE"] for r in v)
            rng = f" [{lo:.2f}, {hi:.2f}]"
        lines.append(
            f"{label} & {cell(v, 'RMSE', 2)}{rng} & {cell(v, 'MAE', 2)} & {cell(v, 'SMAPE', 1)} & "
            f"{cell(v, 'MAPE', 1)} & "
            + "/".join(f"{np.mean([r[f'RMSE_h{h}'] for r in v]):.1f}" for h in (1, 2, 3))
            + r" \\"
        )
    p = {t["test"]: t for t in stats["primary"]}

    def ci(t: dict) -> str:
        return f"${t['delta']:+.2f}$ $[{t['lo']:+.2f}, {t['hi']:+.2f}]$, $p_{{\\text{{Holm}}}}={t['p_holm']:.2f}$"

    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Prospective evaluation on weeks no model or setting had seen: "
            + str(len(final["test_starts"]))
            + r" forecast starts, 2024-W11 to 2026-W32. Settings frozen on development data before the new weeks"
            r" were parsed. Neural arms: mean over 3 seeds, seed range of RMSE in brackets. Bold: best per metric."
            r" Pre-registered tests (block bootstrap over forecast starts, Holm over two):"
            r" P1, SEIR branch vs \model{}: " + ci(p["P1"]) + r";"
            r" P2, \model{}+SEIR vs persistence: " + ci(p["P2"]) + r".}",
            r"\label{tab:prospective}",
            r"\small",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"arm & RMSE & MAE & SMAPE & MAPE & RMSE $h{=}1/2/3$ \\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ]
    )


def main() -> None:
    ns = model_namespace()
    body = "\n\n".join([paired_table(), tuning_table(), compute_table(ns), prospective_table()])
    OUT.write_text(
        "% Generated by scripts/build_paper_extra_tables.py -- do not edit by hand.\n"
        "% Sources: seirgnn2/results/prospective_*.json (EXP-063) and the Kaggle run (EXP-050).\n\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
