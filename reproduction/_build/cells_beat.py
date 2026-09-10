"""Cell sources for the beat-the-floor kernels.

Each kernel runs one architecture over disjoint rolling-origin blocks, with a
deterministic head and again with a Gaussian head, and scores every
retransformation and combination arm against persistence on the identical
windows.

One kernel per architecture, for the same reason the physics sweep splits:
Kaggle caps a session at 12 hours and a single failure should not cost the whole
matrix. A separate combination kernel runs three architectures together, because
the cross-architecture ensemble cannot be assembled from separate sessions.
"""

import env_setup
from gen_kernels import code, md

#: The experiment branch. Must be pushed before a kernel can clone it.
BRANCH = "exp/beat-baseline"

ARCHITECTURES = ("A3TGCN", "STGAT", "ASTGCN", "AAGCN", "DCRNN")

#: Architectures for the cross-architecture ensemble. The three that sit at or
#: under the floor individually -- averaging in DCRNN (+1.78) would only add a
#: known-worse member to the mean.
COMBO = ("A3TGCN", "STGAT", "ASTGCN")

N_ORIGINS = 9
SEEDS = 5
COMBO_SEEDS = 3


def _clone(extra: str = "") -> str:
    return f"""
PROJECT = "https://github.com/rathishTharusha/dengue-forecasting-gnn.git"
BRANCH = "{BRANCH}"
PROJ = SCRATCH / "project"

if not PROJ.exists():
    sh("git", "clone", "--depth", "1", "--branch", BRANCH, PROJECT, str(PROJ))
print("Cloned branch:", BRANCH)

RUNNER = PROJ / "analysis" / "_build" / "run_beat_baseline.py"
assert RUNNER.exists(), f"Runner not found at {{RUNNER}}"
{extra}
"""


def _invoke(args: str, label: str) -> str:
    return f"""
started = time.time()
print("=== {label} ===", flush=True)
proc = subprocess.Popen(
    [str(PY311), "-u", str(RUNNER)] + {args},
    cwd=str(PROJ),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
for line in proc.stdout:
    print(line, end="")
proc.wait()
assert proc.returncode == 0, f"runner failed with exit code {{proc.returncode}}"
print(f"\\n{label} finished in {{(time.time() - started) / 60:.1f}} min", flush=True)
"""


_SUMMARY = '''
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

records = []
for f in sorted(Path(WORK).glob("beat_*.json")):
    rows = json.loads(f.read_text(encoding="utf-8"))
    for r in rows:
        r["source"] = f.stem
    records.extend(rows)
df = pd.DataFrame(records)
print(f"{len(df)} records from {df['source'].nunique()} run(s)")

floor = (df[df["arch"] == "persistence"]
         .groupby("origin")["RMSE_clean"].mean().to_dict())
print("\\nPersistence floor by origin (artifact-free):")
for o, v in sorted(floor.items()):
    print(f"  {o:5.2f}  {v:7.3f}")
print(f"  mean   {np.mean(list(floor.values())):7.3f}")

# Cluster by origin: average the seeds inside an origin first, so the test has
# one observation per independent fold rather than one per training run.
rows = []
model = df[df["arch"] != "persistence"]
for (src, arch, arm), grp in model.groupby(["source", "arch", "arm"]):
    per_origin = grp.groupby("origin")["RMSE_clean"].mean()
    o = sorted(per_origin.index)
    v = np.array([per_origin[k] for k in o])
    f = np.array([floor[k] for k in o])
    d = v - f
    p = stats.ttest_rel(v, f)[1] if len(v) > 1 else float("nan")
    rows.append(dict(source=src, arch=arch, arm=arm, n_origins=len(v),
                     RMSE=v.mean(), vs_floor=d.mean(),
                     wins=int((d < 0).sum()), p=p))

out = pd.DataFrame(rows).sort_values("vs_floor")
pd.set_option("display.width", 160)
print("\\n=== every arm against the floor, clustered by origin ===")
print(out.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

best = out.iloc[0]
print(f"\\nBest arm: {best['arch']} {best['arm']} -> {best['RMSE']:.3f} "
      f"({best['vs_floor']:+.3f} vs floor, {best['wins']}/{best['n_origins']} origins, "
      f"p={best['p']:.4f})")
if best["vs_floor"] < 0 and best["p"] < 0.05:
    print("BEATS THE FLOOR at p<0.05, clustered by origin.")
else:
    print("Does not clear the floor at p<0.05. Report as measured.")

out.to_csv(Path(WORK) / "beat_summary.csv", index=False)
'''


def build(arch: str) -> list:
    """Cells for one architecture's beat-the-floor kernel."""
    return [
        md(
            f"""
# Beat the Floor - {arch}

Whether **{arch}** can be pushed past naive persistence without changing what the
network learns.

The training loop and model selection here are identical to the physics sweep.
Everything measured below is either a change to *how folds are drawn* or a
correction applied to predictions **after** training, calibrated on the
validation split and applied to test.

## Why this run exists

The headline `{arch}`-vs-floor result is **pseudoreplicated**. Within an origin,
seed variance is 0.02-0.05; between origins it is 22.07 / 37.74 / 27.35. Nine
(origin, seed) rows are three independent observations, not nine, and clustering
by origin turns $-0.467, p=0.10$ into $-0.467, p=0.436$. This kernel runs
**{N_ORIGINS} disjoint origin blocks** instead of 3 overlapping ones.

## Arms

| arm | what it is |
|---|---|
| `raw` | the existing pipeline: $\\mathrm{{expm1}}(m)$ |
| `smear` | Duan's smearing, $S=\\overline{{\\exp(e)}}$ on validation residuals |
| `lognorm` | the Gaussian case $S=\\exp(v/2)$; heteroscedastic under a Gaussian head |
| `calib` | one factor per horizon fit against validation RMSE directly |
| `blend_*` | convex blend with persistence, weight per horizon fit on validation |
| `ens_*` | mean of the {SEEDS} seeds' count-space predictions |

`raw` is trained on `log1p` and scored on counts, so it reports a conditional
**median** where RMSE is minimised by the conditional **mean**. That predicts a
downward bias growing with residual variance -- and `error_diagnosis.json`
measures exactly that: $-12.92$ on outbreak windows against $+0.38$ on quiet
ones. The three corrections are three ways of estimating the same factor.
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone"),
        code(_clone(f'ARCH = "{arch}"')),
        md(f"## 3. Deterministic head ({N_ORIGINS} origins x {SEEDS} seeds)"),
        code(
            _invoke(
                f'["--arch", ARCH, "--n-origins", "{N_ORIGINS}", "--seeds", "{SEEDS}",'
                f' "--out-dir", str(WORK), "--tag", ARCH]',
                "deterministic head",
            )
        ),
        md(
            """## 4. Gaussian head

The same run with a `probabilistic` head. This is what makes the `lognorm`
correction *heteroscedastic*: the network predicts a variance per window, so the
correction is largest exactly on the high-variance outbreak windows where the
bias is measured to be largest."""
        ),
        code(
            _invoke(
                f'["--arch", ARCH, "--n-origins", "{N_ORIGINS}", "--seeds", "{SEEDS}",'
                f' "--probabilistic", "--out-dir", str(WORK), "--tag", ARCH]',
                "Gaussian head",
            )
        ),
        md("## 5. Every arm against the floor"),
        code(_SUMMARY),
    ]


def build_combo() -> list:
    """Cells for the cross-architecture combination kernel."""
    archs = ", ".join(f'"{a}"' for a in COMBO)
    return [
        md(
            f"""
# Beat the Floor - cross-architecture combination

{", ".join(COMBO)} in **one** session, so their predictions can be averaged.

The forecast-combination literature's most consistent finding across fifty years
is that averaging diverse forecasts beats picking one, because independent errors
cancel while the agreed signal survives. That cannot be assembled from separate
kernels -- the ensemble needs every member's predictions on the same folds at the
same time -- so it gets its own run.

DCRNN is excluded on purpose: at $+1.78$ against the floor it is a known-worse
member, and averaging it in would only drag the mean.

{len(COMBO)} architectures x {N_ORIGINS} disjoint origins x {COMBO_SEEDS} seeds.
The `multi_*` arms are the cross-architecture ensembles.
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone"),
        code(_clone(f"ARCHS = [{archs}]")),
        md("## 3. Combination run"),
        code(
            _invoke(
                f'["--arch"] + ARCHS + ["--n-origins", "{N_ORIGINS}",'
                f' "--seeds", "{COMBO_SEEDS}", "--out-dir", str(WORK), "--tag", "combo"]',
                "cross-architecture combination",
            )
        ),
        md("## 4. Every arm against the floor"),
        code(_SUMMARY),
    ]
