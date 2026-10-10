"""Recompute every number in the ICITR paper from the raw result files, and say where it comes from.

Writes full_paper/icitr/NUMBER_SOURCES.txt: for each number, the value the paper prints, the
value recomputed here, OK or MISMATCH, the file it comes from and how it is computed. Then a
list of where the raw data itself came from.

Nothing is typed in by hand except the value the paper prints (to compare against). Every
recomputed value is read from:

  full_paper/outputs/kaggle_run/    the Kaggle notebook run (EXP-050): 810 training runs
  seirgnn2/results/prospective_*    the 2024-2026 prospective test (EXP-063)
  data/new_weeks/qc_report.json     the parse of the 127 new reports

    python scripts/verify_paper_numbers.py
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "full_paper" / "outputs" / "kaggle_run"
RES = REPO / "seirgnn2" / "results"
OUT = REPO / "full_paper" / "icitr" / "NUMBER_SOURCES.txt"
ENC = ["AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN"]
FAILING = ["STGAT", "A3TGCN", "DCRNN"]
WORKING = ["AAGCN", "ASTGCN", "LSTM"]

lines: list[str] = []
n_ok = n_bad = 0


def say(text: str = "") -> None:
    lines.append(text)


def check(paper: str, value: float, where: str, source: str, how: str, decimals: int = 2) -> None:
    """Compare the printed value with the recomputed one at the paper's precision."""
    global n_ok, n_bad
    shown = f"{value:.{decimals}f}"
    ok = abs(float(paper) - value) <= 0.5 * 10 ** (-decimals) + 1e-9
    n_ok += ok
    n_bad += not ok
    say(f"  [{'OK' if ok else 'MISMATCH'}] {paper:>9}  (recomputed {shown})  -- {where}")
    say(f"             from: {source}")
    say(f"             how:  {how}")


def note(text: str, where: str, source: str, how: str) -> None:
    say(f"  [INFO] {text}  -- {where}")
    say(f"             from: {source}")
    say(f"             how:  {how}")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


# ----------------------------------------------------------------------------- inputs
runs = json.loads((RUN / "runs.json").read_text(encoding="utf-8"))
R3 = [r for r in runs if r["origin_set"] == "three"]
U = [(o, s) for o in (0.55, 0.7, 0.85) for s in (0, 1, 2)]
BY = {(r["name"], r["origin"], r["seed"]): r for r in R3}
PERS = {r["origin"]: r for r in R3 if r["name"] == "persistence"}
NB_OUT = (RUN / "notebook_output.txt").read_text(encoding="utf-8", errors="replace")


def mean(name: str, metric: str = "val_RMSE") -> float:
    v = [r[metric] for r in R3 if r["name"] == name]
    if not v:
        raise KeyError(name)
    return float(np.mean(v))


def diffs(a: str, b: str, metric: str) -> np.ndarray:
    return np.array(
        [
            BY[(a, o, s)][metric] - (PERS[o] if b == "persistence" else BY[(b, o, s)])[metric]
            for o, s in U
        ]
    )


def sign_flip(d: np.ndarray) -> float:
    flips = np.array(list(itertools.product((-1, 1), repeat=len(d))), float)
    return float(np.mean(np.abs(flips @ d / len(d)) >= abs(d.mean()) - 1e-12))


def read_csv(name: str) -> list[dict]:
    with (RUN / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nb_line(pattern: str) -> re.Match:
    m = re.search(pattern, NB_OUT)
    if m is None:
        raise SystemExit(f"notebook_output.txt has no line matching {pattern!r}")
    return m


LEV = {r["lever"]: r for r in read_csv("levers.csv")}
RESCUE = {r["encoder"]: r for r in read_csv("rescue.csv")}
RECOVER = {
    r[""]: float(r["validation residual variance a linear model recovers"])
    for r in read_csv("residual_recovery.csv")
}
SEIRDX = read_csv("seir_decoder_diagnosis.csv")
CORR = read_csv("residual_correlation.csv")
FINAL = json.loads((RES / "prospective_final.json").read_text(encoding="utf-8"))
STATS = json.loads((RES / "prospective_stats.json").read_text(encoding="utf-8"))
FROZEN = json.loads((RES / "prospective_frozen.json").read_text(encoding="utf-8"))
DEV = json.loads((RES / "prospective_dev.json").read_text(encoding="utf-8"))
QC = json.loads((REPO / "data" / "new_weeks" / "qc_report.json").read_text(encoding="utf-8"))

KAG = "full_paper/outputs/kaggle_run/runs.json (EXP-050 Kaggle run)"
PRO = "seirgnn2/results/prospective_final.json (EXP-063)"
PST = "seirgnn2/results/prospective_stats.json (EXP-063)"


def pro_mean(arm: str, metric: str) -> float:
    return float(np.mean([r[metric] for r in FINAL["rows"] if r["name"] == arm]))


def stat(group: str, a: str, b: str, test: str | None = None) -> dict:
    for t in STATS[group]:
        if t["a"] == a and t["b"] == b and (test is None or t.get("test") == test):
            return t
    raise KeyError((group, a, b))


SEIR, PAGE, RESID, PERSIST = (
    "Adaptive SEIR-GNN",
    "Adaptive gated non-SEIR",
    "Adaptive residual",
    "Persistence",
)

# ----------------------------------------------------------------------------- header
say("WHERE EVERY NUMBER IN THE ICITR PAPER COMES FROM")
say("=" * 78)
say("Generated by scripts/verify_paper_numbers.py -- do not edit by hand. Rerun it to check.")
say(f"Repository commit when generated: {git('rev-parse', '--short', 'HEAD')}")
say("")
say("How to read this file")
say("  [OK]        the number printed in the paper equals the value recomputed from the raw file")
say("  [MISMATCH]  it does not -- fix the paper")
say("  [INFO]      a count or fact that is read, not computed")
say("  'from'      the raw file the value is read from")
say("  'how'       what is computed from it")
say("")
say("The raw result files and the commits that added them")
for rel in (
    "full_paper/outputs/kaggle_run/runs.json",
    "seirgnn2/results/prospective_final.json",
    "seirgnn2/results/prospective_stats.json",
    "data/new_weeks/qc_report.json",
):
    say(f"  {rel:45s} added in {git('log', '-1', '--format=%h %ad %s', '--date=short', '--', rel)}")
say("")
say("Two experiments produce the numbers, on different test weeks, so numbers from one cannot be")
say("compared with numbers from the other:")
say(
    "  EXP-050  development study, 3 rolling origins (0.55/0.70/0.85) x 3 seeds = 9 runs per model,"
)
say(
    "           run once on Kaggle (CPU) by full_paper/kaggle/dengue_physics_gnn.ipynb. Tables I, II, IV."
)
say("  EXP-063  prospective test on 2024-W11..2026-W32, settings frozen before those weeks were")
say("           parsed; run locally by seirgnn2/prospective.py. Table III.")

# ----------------------------------------------------------------------------- 1. headline
say("")
say("1. ABSTRACT AND INTRODUCTION")
say("-" * 78)
m = nb_line(
    r"cases\(t-1\) -> cases\(t\): r2 = ([0-9.]+);\s+best climate variable \(lags 2-8\): r2 = ([0-9.]+)"
)
check(
    "0.89",
    float(m.group(1)),
    "lag-1 r2 (abstract, intro)",
    "notebook_output.txt (EXP-050)",
    "r2 of log cases at t on log cases at t-1, all districts, development weeks",
)
check(
    "0.03",
    float(m.group(2)),
    "best climate covariate r2 (intro)",
    "notebook_output.txt (EXP-050)",
    "best single ERA5 variable at causal lags 2-8 weeks",
)
for e in FAILING:
    gap = mean(f"{e}+direct") - mean("persistence")
    check(
        {"STGAT": "5", "A3TGCN": "11", "DCRNN": "17"}[e],
        gap,
        f"{e} direct minus persistence, validation (intro: '5--17')",
        KAG,
        "mean validation RMSE of the published (direct) head minus persistence",
        0,
    )
say("  (the intro's '5--17' is the range of these three)")
check(
    "34.75",
    pro_mean(PERSIST, "RMSE"),
    "persistence RMSE on 2024-2026 (abstract, intro)",
    PRO,
    "mean over the 85 test starts",
)
check(
    "35.43",
    pro_mean(PAGE, "RMSE"),
    "PAGE RMSE on 2024-2026 (abstract, intro)",
    PRO,
    "mean over 3 seeds of the 'Adaptive gated non-SEIR' arm (= PAGE)",
)
p1 = stat("primary", SEIR, PAGE, "P1")
check(
    "0.67",
    p1["p_holm"],
    "SEIR branch vs PAGE, p (abstract, intro)",
    PST,
    "pre-registered test P1, block bootstrap, Holm",
)
check("-0.03", p1["delta"], "SEIR branch vs PAGE, RMSE difference", PST, "P1 delta")
note(
    f"{len(FINAL['test_starts'])} forecast starts",
    "'85 forecast starts'",
    PRO,
    "length of test_starts",
)
note(
    f"{QC['expected_reports']} reports expected; {QC['observed']} observed, {QC['corrected']} corrected, {QC['missing']} missing",
    "'127 new reports', '109 observed, 8 corrected, 10 missing'",
    "data/new_weeks/qc_report.json",
    "counts written by the parser",
)
m = nb_line(r"rebuilt series: (\d+) weeks.*?(\d+) weeks with no report")
note(
    f"{m.group(1)} weeks, {m.group(2)} with no report",
    "'559 weeks', 'seven weeks missing'",
    "notebook_output.txt",
    "printed by the data cell",
)
note(
    "186,000 cases and 440 deaths in 2017",
    "intro, background",
    "cited paper: Tissera et al. 2020 (full_paper/references/Tissera_2020_Severe_Dengue_Sri_Lanka_2017.pdf)",
    "quoted from the paper, not computed -- open the PDF to check",
)

# ----------------------------------------------------------------------------- 2. audit
say("")
say("2. DATASET AUDIT (Section III)")
say("-" * 78)
m = nb_line(r"rows 0-47 are dated (\S+) \.\. (\S+); rows 51 onward run (\S+) \.\. (\S+)")
note(
    f"rows 0-47 = {m.group(1)}..{m.group(2)}; rows 51+ = {m.group(3)}..{m.group(4)}",
    "'December 2022 to November 2023'",
    "notebook_output.txt",
    "each array row matched to a report by its 25 counts",
)
m = nb_line(r"benchmark fold 0.55: (\d+) rows from 2023 in TRAINING")
note(
    f"{m.group(1)} rows from 2023 in training",
    "'45 rows from 2023'",
    "notebook_output.txt",
    "rows dated 2023 inside each benchmark training split",
)
m = nb_line(r"r = ([0-9.]+)\); on the dates the CASE rows carry, r = ([0-9.]+)")
check(
    "0.95",
    float(m.group(1)),
    "array temperature vs ERA5, own order",
    "notebook_output.txt",
    "Pearson r",
)
check(
    "0.45",
    float(m.group(2)),
    "array temperature vs ERA5, case dates",
    "notebook_output.txt",
    "Pearson r",
)
m = nb_line(r"best at k = \+12 \(r = ([0-9.]+)\); at the documented k = -12, r = ([0-9.]+)")
check(
    "0.68",
    float(m.group(1)),
    "'lag-12' rain vs ERA5 12 weeks later",
    "notebook_output.txt",
    "Pearson r",
)
check(
    "0.19",
    float(m.group(2)),
    "'lag-12' rain vs ERA5 12 weeks earlier",
    "notebook_output.txt",
    "Pearson r",
)
m = nb_line(r"array row 395: districts sum ([0-9,]+).*corrected week sums to (\d+)")
note(
    f"{m.group(1)} printed vs {m.group(2)} true",
    "'7,165 cases ... truly report 349'",
    "notebook_output.txt",
    "row A sum vs cumulative-row difference, Vol 48 No 02 PDF",
)
m = nb_line(r"(\d+)/24 cells = sum of the two before")
note(
    f"{m.group(1)} of 24 cells",
    "'15 of 24 cells'",
    "notebook_output.txt",
    "cells equal to the sum of the two before them",
)
m = nb_line(r"graph: 25 districts, (\d+) directed edges plus self-loops")
note(
    f"{m.group(1)} edges + 25 self-loops = {int(m.group(1)) + 25}",
    "'141 directed entries'",
    "notebook_output.txt",
    "benchmark adjacency list",
)
adj = json.loads(
    (REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json").read_text(encoding="utf-8")
)
n139 = sum(1 for d, nb in adj.items() for o in nb if o in adj and o != d) + len(adj)
note(
    f"{n139} entries",
    "'corrected version ... (139)'",
    "notebooks/baseline/sri_lanka_adj_list.json (commit 3878e70)",
    "edges + self-loops",
)

# ----------------------------------------------------------------------------- 3. Table I
say("")
say("3. TABLE I (comparative analysis) -- mean validation / test RMSE over 9 runs")
say("-" * 78)
say(f"  source for every row: {KAG}")
say(
    "  how: mean of val_RMSE and of RMSE (test) over the 9 (origin, seed) runs of that configuration"
)
for e in ENC:
    cells = [
        f"{mean(f'{e}+{h}'):.2f} / {mean(f'{e}+{h}', 'RMSE'):.2f}"
        for h in ("direct", "gated", "foi_res")
    ]
    say(f"    {e:7s} direct {cells[0]}   PAGE {cells[1]}   PAGE+SEIR {cells[2]}")
for label, name in (
    ("Persistence", "persistence"),
    ("No message passing", "graph=none"),
    ("GCN", "graph=gcn"),
    ("STID-style MLP", "R3 STID-style MLP"),
    ("k-NN analogues", "R4b k-NN"),
    ("NB-GLM", "R4a NB-GLM"),
    ("Boosted trees", "K7 trees"),
    ("Boosted trees + climate", "K6 trees + climate"),
    ("B", "B"),
):
    say(f"    {label:24s} {mean(name):.2f} / {mean(name, 'RMSE'):.2f}")
say("  (in runs.json, 'gated' is PAGE and 'foi_res' is PAGE + SEIR branch)")

# ----------------------------------------------------------------------------- 4. Results text
say("")
say("4. RESULTS, SECTION VI-A (comparative analysis)")
say("-" * 78)
ps = []
for e in ENC:
    for ref in ("direct", "persistence"):
        for metric in ("val_RMSE", "RMSE"):
            d = diffs(f"{e}+gated", ref if ref == "persistence" else f"{e}+direct", metric)
            ps.append(sign_flip(d))
for e, pv, pt in (
    ("STGAT", "5.95", "25.54"),
    ("A3TGCN", "10.95", "23.24"),
    ("DCRNN", "16.29", "34.50"),
):
    dv, dt = (
        diffs(f"{e}+gated", f"{e}+direct", "val_RMSE"),
        diffs(f"{e}+gated", f"{e}+direct", "RMSE"),
    )
    check(
        pv,
        -dv.mean(),
        f"PAGE repairs {e}, validation ({int((dv < 0).sum())}/9 runs)",
        KAG,
        "mean over 9 runs of (direct - PAGE)",
    )
    check(
        pt,
        -dt.mean(),
        f"PAGE repairs {e}, test ({int((dt < 0).sum())}/9 runs)",
        KAG,
        "mean over 9 runs of (direct - PAGE)",
    )
p = sign_flip(diffs("STGAT+gated", "STGAT+direct", "val_RMSE"))
check(
    "0.004",
    p,
    "p for a 9/9 repair (raw; BH-adjusted stays 0.004)",
    KAG,
    "exact sign-flip test over 2^9 sign patterns",
    3,
)
check(
    "0.14",
    max(abs(diffs(f"{e}+gated", f"{e}+direct", "val_RMSE").mean()) for e in WORKING),
    "largest validation change on the working encoders",
    KAG,
    "max over AAGCN/ASTGCN/LSTM of |PAGE - direct|",
)
d = diffs("AAGCN+gated", "AAGCN+direct", "RMSE")
check(
    "-1.62",
    d.mean(),
    f"AAGCN PAGE vs direct, test ({int((d < 0).sum())}/9)",
    KAG,
    "mean of (PAGE - direct)",
)
for e, v, t in (("AAGCN", "-1.29", "-2.48"), ("ASTGCN", "-1.10", "-1.26")):
    dv, dt = (
        diffs(f"{e}+gated", "persistence", "val_RMSE"),
        diffs(f"{e}+gated", "persistence", "RMSE"),
    )
    check(
        v,
        dv.mean(),
        f"{e}+PAGE vs persistence, validation ({int((dv < 0).sum())}/9)",
        KAG,
        "mean of (PAGE - persistence), same test weeks",
    )
    check(
        t,
        dt.mean(),
        f"{e}+PAGE vs persistence, test ({int((dt < 0).sum())}/9)",
        KAG,
        "mean of (PAGE - persistence), same test weeks",
    )
check(
    "16.57",
    min(mean(f"{e}+gated") for e in ENC),
    "PAGE validation range, low",
    KAG,
    "min over encoders",
)
check(
    "18.35",
    max(mean(f"{e}+gated") for e in ENC),
    "PAGE validation range, high",
    KAG,
    "max over encoders",
)
check("33.53", mean("AAGCN+gated", "RMSE"), "lowest test RMSE (AAGCN+PAGE)", KAG, "mean test RMSE")
lowest = min({r["name"] for r in R3}, key=lambda n: mean(n, "RMSE"))
note(
    f"lowest mean test RMSE among all 3-origin configurations: '{lowest}'",
    "'lowest test RMSE of any configuration'",
    KAG,
    "argmin over configurations",
)
check(
    "37.54", mean("B", "RMSE"), "B on test", KAG, "mean test RMSE of B (AAGCN, direct, NB, season)"
)
check(
    "0.48",
    RECOVER["STGAT"],
    "STGAT residual variance recovered ('48%')",
    "full_paper/outputs/kaggle_run/residual_recovery.csv",
    "linear fit on training residuals, R^2 on validation",
)
check(
    "0.60",
    RECOVER["A3TGCN"],
    "A3TGCN residual variance recovered ('60%')",
    "residual_recovery.csv",
    "same",
)
check(
    "0.03",
    min(RECOVER[e] for e in ("AAGCN", "ASTGCN", "LSTM")),
    "working encoders, low ('3%')",
    "residual_recovery.csv",
    "same",
    2,
)
check(
    "0.07",
    max(RECOVER[e] for e in ("AAGCN", "ASTGCN", "LSTM")),
    "working encoders, high ('7%')",
    "residual_recovery.csv",
    "same",
    2,
)
g = LEV["graph (GCN vs no message passing)"]
check(
    "-0.06",
    float(g["val delta"]),
    f"graph adds 0.06 ({g['wins']})",
    "full_paper/outputs/kaggle_run/levers.csv",
    "mean validation (GCN - no message passing)",
)
check("16.52", mean("R4b k-NN"), "k-NN validation", KAG, "mean")
check("34.34", mean("R4b k-NN", "RMSE"), "k-NN test", KAG, "mean")

say("")
say("5. RESULTS, SECTION VI-B (ablation, Table II)")
say("-" * 78)
check(
    "17.47",
    min(mean(f"{e}+residual") for e in FAILING),
    "residual head, failing encoders, low",
    KAG,
    "mean validation RMSE",
)
check(
    "17.85",
    max(mean(f"{e}+residual") for e in FAILING),
    "residual head, failing encoders, high",
    KAG,
    "mean validation RMSE",
)
check("16.98", mean("STGAT+gated"), "STGAT with gate", KAG, "mean validation RMSE")
check(
    "0.21",
    mean("A3TGCN+gated") - mean("A3TGCN+residual"),
    "gate costs A3TGCN",
    KAG,
    "PAGE - residual, validation",
)
check(
    "0.83",
    mean("DCRNN+gated") - mean("DCRNN+residual"),
    "gate costs DCRNN",
    KAG,
    "PAGE - residual, validation",
)
t = stat("secondary", PAGE, RESID)
check(
    "-0.31",
    t["delta"],
    "gate vs no gate on 2024-2026",
    PST,
    "secondary comparison, block bootstrap",
)
check("-0.77", t["lo"], "its 95% CI, low", PST, "bootstrap interval")
check("-0.08", t["hi"], "its 95% CI, high", PST, "bootstrap interval")
for e, v, w in (("STGAT", "0.66", "1/9"), ("A3TGCN", "-0.00", "6/9")):
    r = RESCUE[e]
    check(
        v,
        float(r["SEIR - gated (val)"]),
        f"SEIR branch vs PAGE on {e} ({r['wins']}; paper says {w})",
        "full_paper/outputs/kaggle_run/rescue.csv",
        "mean validation (PAGE+SEIR - PAGE)",
    )
note(
    f"DCRNN: SEIR branch wins {RESCUE['DCRNN']['wins']}",
    "'helps DCRNN on only 3/9 runs'",
    "rescue.csv",
    "wins column",
)
w = [float(RESCUE[e]["SEIR - gated (val)"]) for e in WORKING]
check("0.75", min(w), "SEIR branch cost on working encoders, low", "rescue.csv", "PAGE+SEIR - PAGE")
check(
    "0.86", max(w), "SEIR branch cost on working encoders, high", "rescue.csv", "PAGE+SEIR - PAGE"
)
check(
    "24.01",
    min(mean(f"{e}+foi") for e in ENC),
    "pure SEIR decoder, low",
    KAG,
    "mean validation RMSE of 'foi' head",
)
check(
    "25.75",
    max(mean(f"{e}+foi") for e in ENC),
    "pure SEIR decoder, high",
    KAG,
    "mean validation RMSE of 'foi' head",
)
fl = [float(r["targets below the lambda=0 floor (%)"]) for r in SEIRDX]
r2 = [float(r["r2 of log lambda* on log cases[t-1]"]) for r in SEIRDX]
check(
    "14",
    min(fl),
    "targets below the SEIR floor, low (%)",
    "full_paper/outputs/kaggle_run/seir_decoder_diagnosis.csv",
    "training targets the simulator cannot reach, per origin",
    0,
)
check(
    "16", max(fl), "targets below the SEIR floor, high (%)", "seir_decoder_diagnosis.csv", "same", 0
)
check(
    "0.22",
    min(r2),
    "force-of-infection r2, low",
    "seir_decoder_diagnosis.csv",
    "r2 of the required log force of infection on log cases",
    2,
)
check("0.26", max(r2), "force-of-infection r2, high", "seir_decoder_diagnosis.csv", "same", 2)
nb = LEV["NB likelihood (vs squared error)"]
check(
    "-0.76",
    float(nb["val delta"]),
    f"NB likelihood, validation ({nb['wins']})",
    "levers.csv",
    "B minus B with squared error",
)
check(
    "0.014",
    float(nb["p_adj"]),
    "its BH-adjusted p",
    "levers.csv",
    "sign-flip p, BH across the lever table",
    3,
)
check("1.61", float(nb["test delta"]), "NB likelihood, test", "levers.csv", "same, on test")
arch = [float(r["val delta"]) for r in read_csv("levers.csv") if r["group"] == "graph/architecture"]
best_arch = -min(arch)
note(
    f"largest gain {best_arch:.3f} -> {'OK' if best_arch <= 0.15 else 'MISMATCH'} (bound, not a value)",
    "'no architecture change improves validation by more than 0.15'",
    "levers.csv",
    "most negative val delta in the architecture group",
)
for name, v in (("RevIN", "1.47"), ("STID-style MLP", "0.84"), ("TimeGAN augmentation", "0.60")):
    r = LEV[name]
    check(
        v,
        float(r["val delta"]),
        f"{name} ({r['wins']})",
        "levers.csv",
        "arm minus its control, validation",
    )

say("")
say("6. RESULTS, SECTION VI-C (prospective 2024-2026, Table III)")
say("-" * 78)
p2 = stat("primary", SEIR, PERSIST, "P2")
check("0.62", p2["p_holm"], "PAGE+SEIR vs persistence, p (Holm)", PST, "pre-registered test P2")
check("-0.41", p1["lo"], "P1 interval, low", PST, "bootstrap")
check("0.12", p1["hi"], "P1 interval, high", PST, "bootstrap")
check("11.35", pro_mean(PAGE, "MAE"), "PAGE MAE", PRO, "mean over seeds")
check("11.78", pro_mean(PERSIST, "MAE"), "persistence MAE", PRO, "")
check("45.7", pro_mean(PAGE, "SMAPE"), "PAGE SMAPE", PRO, "mean over seeds", 1)
check("50.3", pro_mean(PERSIST, "SMAPE"), "persistence SMAPE", PRO, "", 1)
gaps = [stat("secondary", a, "Seasonal naive")["delta"] for a in (SEIR, PAGE, RESID)]
check(
    "26.7",
    -max(gaps),
    "beats seasonal naive by, low",
    PST,
    "secondary comparisons, learned arms and persistence",
    1,
)
check(
    "27.7",
    -min([*gaps, stat("secondary", PERSIST, "Seasonal naive")["delta"]]),
    "beats seasonal naive by, high",
    PST,
    "same",
    1,
)
for arm, vals in ((PERSIST, ("14.72", "20.05", "160.51")), (PAGE, ("14.32", "21.43", "162.75"))):
    for y, v in zip((2024, 2025, 2026), vals):
        check(
            v,
            pro_mean(arm, f"RMSE_{y}"),
            f"{'PAGE' if arm == PAGE else 'persistence'} RMSE in {y}",
            PRO,
            "mean over seeds",
        )
chosen = [r for r in DEV if r["name"] == "Adaptive SEIR-GNN | lr=0.003 hidden=64"]
check(
    "27.34",
    float(np.mean([r["RMSE"] for r in chosen])),
    "SEIR arm, development test (9 origins)",
    "seirgnn2/results/prospective_dev.json",
    "mean over 27 runs, chosen setting",
)
check(
    "28.54",
    float(np.mean([r["RMSE"] for r in DEV if r["name"] == "persistence"])),
    "persistence, same development test",
    "prospective_dev.json",
    "mean over 9 origins",
)

say("")
say("7. RESULTS, SECTION VI-D (tuning and cost, Table IV)")
say("-" * 78)
note(
    f"chosen: {FROZEN['arms']}",
    "'all arms chose 3e-3 and 64'",
    "seirgnn2/results/prospective_frozen.json",
    "validation-only choice",
)
per_arm = defaultdict(list)
for k, v in FROZEN["dev_val_RMSE"].items():
    per_arm[k.split(" | ")[0]].append(v)
check(
    "0.57",
    max(max(v) - min(v) for a, v in per_arm.items() if not a.startswith("AR")),
    "largest validation spread over the grid",
    "prospective_frozen.json",
    "max over arms of (worst - best setting)",
)
note(
    f"{len(chosen)} runs per setting x 4 settings = {4 * len(chosen)} per arm",
    "'108 runs per arm'",
    "prospective_dev.json",
    "count",
)
rows = [json.loads(x) for x in (RUN / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x]
three = [r for r in rows if r.get("origin_set") == "three" and "epochs_ran" in r]
spe = {
    n: sum(r["elapsed"] for r in three if r["name"] == n)
    / sum(r["epochs_ran"] for r in three if r["name"] == n)
    for n in ("AAGCN+direct", "AAGCN+foi_res")
}  # Table IV's definition: total time / total epochs
check(
    "2.2",
    spe["AAGCN+foi_res"] / spe["AAGCN+direct"],
    "SEIR branch time per epoch, AAGCN (x)",
    "full_paper/outputs/kaggle_run/runs.jsonl",
    "seconds per epoch (total time / total epochs), PAGE+SEIR / direct",
    1,
)
check(
    "19",
    float(
        np.mean(
            [
                r["elapsed"]
                for r in rows
                if r["name"] == "AAGCN+gated" and r["origin_set"] == "three"
            ]
        )
    ),
    "AAGCN+PAGE seconds per run",
    "runs.jsonl",
    "mean elapsed",
    0,
)
spec = importlib.util.spec_from_file_location(
    "extra", REPO / "scripts" / "build_paper_extra_tables.py"
)
extra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extra)
check(
    "2873",
    extra.params(extra.model_namespace(), "AAGCN", "direct"),
    "AAGCN parameters",
    "full_paper/kaggle/src (model code)",
    "count of trainable parameters",
    0,
)
m = nb_line(r"810/810 runs\s+([0-9.]+) min elapsed")
check(
    "3.0",
    float(m.group(1)) / 60,
    "main study wall time (hours)",
    "notebook_output.txt",
    "last progress line, 4 parallel workers",
    1,
)

say("")
say("8. DISCUSSION")
say("-" * 78)
names = ["AAGCN", "ASTGCN", "LSTM", "SEIR-LSTM", "k-NN"]
cm = {r[""]: r for r in CORR}
vals = [float(cm[a][b]) for a, b in itertools.combinations(names, 2)]
check(
    "0.91",
    min(vals),
    "residual correlation, low",
    "full_paper/outputs/kaggle_run/residual_correlation.csv",
    "pairwise, working encoders + SEIR-LSTM + k-NN",
)
check("0.97", max(vals), "residual correlation, high", "residual_correlation.csv", "same")

# ----------------------------------------------------------------------------- data sources
say("")
say("9. WHERE THE RAW DATA CAME FROM")
say("-" * 78)
say("Development weeks (2013-W26 .. 2024-W10), as listed by the Kaggle notebook's first cell:")
start = NB_OUT.index("source files verified")
for line in NB_OUT[start:].splitlines()[1:11]:
    say("  " + line.strip())
say("")
say("  ERA5 weather: Open-Meteo historical archive (https://archive-api.open-meteo.com), one")
say("  interior point per district; census and mid-year population: Department of Census and")
say("  Statistics, Sri Lanka (https://www.statistics.gov.lk). All 58 files: SOURCES.csv in the")
say("  Kaggle dataset, built by full_paper/kaggle/fetch_sources.py.")
say("")
say("  IMPORTANT: 'cases/output_Dengue Fever.csv' is the table of parsed Weekly Epidemiological")
say("  Report rows that the benchmark authors (Weng et al.) sent us. It has no public URL.")
say("  We did NOT parse the 2013-2024 reports ourselves. We re-dated, mapped and corrected it.")
say("")
man = REPO / "data" / "external" / "wer_new_manifest.csv"
with man.open(encoding="utf-8") as f:
    wer = list(csv.DictReader(f))
say(
    f"New weeks (2024-W11 .. 2026-W32): {len(wer)} Weekly Epidemiological Report PDFs, downloaded by us"
)
say("  from https://www.epid.gov.lk (Epidemiology Unit, Ministry of Health, Sri Lanka).")
say(f"  File names, URLs and SHA-256: {man.relative_to(REPO).as_posix()}")
say("  Parsed by analysis/_build/parse_new_weeks.py under docs/NEW_WEEKS_RULES.md (rules fixed")
say("  before any count was read).")
say("")
say("Seroprevalence 0.682 (SEIR initial immunity): Jeewandara et al., PLOS ONE 10(12): e0144799,")
say("  2015 -- suburban Colombo 2013-14, 1152/1689 seropositive. Quoted, not computed.")

# ----------------------------------------------------------------------------- summary
say("")
say("=" * 78)
say(f"SUMMARY: {n_ok} numbers match the raw files, {n_bad} do not.")
say("Not checked by this script: Tables I-IV themselves (they are generated by")
say("scripts/build_icitr_tables.py from the same files, so they cannot differ), quoted figures")
say("from other papers, and the EXP-062 re-check (pending).")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{n_ok} OK, {n_bad} MISMATCH -> {OUT.relative_to(REPO)}")
