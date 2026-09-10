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

#: Where the kernels clone from. This work was merged to main in c57b9ff, so
#: main is the reproducible reference -- same rule as the physics sweep. Must be
#: pushed before a kernel can clone it.
BRANCH = "main"

ARCHITECTURES = ("A3TGCN", "STGAT", "ASTGCN", "AAGCN", "DCRNN")

#: Architectures for the cross-architecture ensemble. The three that sit at or
#: under the floor individually -- averaging in DCRNN (+1.78) would only add a
#: known-worse member to the mean.
COMBO = ("A3TGCN", "STGAT", "ASTGCN")

N_ORIGINS = 9
SEEDS = 5
COMBO_SEEDS = 3

#: DCRNN is the slowest architecture and the furthest above the floor (+1.78),
#: so it gets fewer seeds rather than a share of wall clock it cannot repay.
ARCH_SEEDS = {"DCRNN": 3}

#: Output heads, each a separate invocation of the runner.
HEADS = ("det", "gauss", "nb")


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


_HEAD_BLURB = {
    "det": """## 3. Deterministic head

Huber on `log1p` -- the existing pipeline, and the arm every correction is
measured against.""",
    "gauss": """## 4. Gaussian head

The same run with a Gaussian head. This is what makes the `lognorm` correction
*heteroscedastic*: the network predicts a variance per window, so the correction
is largest on exactly the high-variance outbreak windows where the measured bias
is $-12.9$ rather than $+0.4$.""",
    # Raw: these carry LaTeX, and `\alpha` in a plain string is a bell character.
    "nb": r"""## 5. Negative-binomial head

A count likelihood, $\mathrm{Var} = \mu + \alpha\mu^2$, emitting the conditional
**mean** directly. Where the corrections above repair a log-space median after
the fact, this never creates the problem -- there is no retransformation step to
be biased by. NB2 rather than Poisson because the target is heavily
overdispersed (median 13, max 2631, 9.7% zeros).

The `calib` factor fitted on top is a **diagnostic** here, not a repair: near
1.0 means the count likelihood really did remove the bias.""",
}


def _head_cells(arch: str) -> list:
    """One markdown + one invocation cell per output head."""
    seeds = ARCH_SEEDS.get(arch, SEEDS)
    cells = []
    for head in HEADS:
        cells.append(md(_HEAD_BLURB[head]))
        cells.append(
            code(
                _invoke(
                    f'["--arch", ARCH, "--n-origins", "{N_ORIGINS}", "--seeds", "{seeds}",'
                    f' "--head", "{head}", "--out-dir", str(WORK), "--tag", ARCH]',
                    f"{head} head",
                )
            )
        )
    return cells


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
        *_head_cells(arch),
        md("## 6. Every arm against the floor"),
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
        *[
            code(
                _invoke(
                    f'["--arch"] + ARCHS + ["--n-origins", "{N_ORIGINS}", "--seeds",'
                    f' "{COMBO_SEEDS}", "--head", "{h}", "--out-dir", str(WORK),'
                    f' "--tag", "combo"]',
                    f"combination, {h} head",
                )
            )
            for h in HEADS
        ],
        md("## 4. Every arm against the floor"),
        code(_SUMMARY),
    ]


#: Feature sets for the covariate kernels, control first.
FEATURE_SETS = ("cases", "causal", "climate")

#: Architectures that get a covariate kernel: the two that sit below the floor.
COVARIATE_ARCHS = ("A3TGCN", "STGAT")


def build_covariates(arch: str) -> list:
    """Cells for one architecture's covariate kernel."""
    return [
        md(
            f"""
# Covariates - {arch}

Ten of the eleven channels in the processed array are currently **discarded**.
`adaptive.load_dataset` takes `raw[..., 5]` and drops temperature, humidity, soil
moisture, canopy interception, precipitation and NDVI -- the variables the dengue
literature treats as the drivers of transmission. This is the largest untested
information lever left in the project.

## Arms

| features | channels | what it tests |
|---|---|---|
| `cases` | 1 | the univariate control; every other arm must beat it |
| `causal` | 8 | humidity, soil moisture, mean temperature, mean precipitation |
| `climate` | 17 | everything, plus missingness indicators |

`causal` is deliberately small. A causality-tested Vietnam dengue study found
humidity, soil moisture, wet-bulb temperature and rainfall dominated its lagged
predictor set, and with ~200 training weeks a wide input is a variance problem
before it is an information gain.

## Two data problems handled first

**A 0 K temperature.** The released array has no NaNs because the authors ran
`np.nan_to_num`, turning missing GLDAS values into zeros. Measured, the
missingness is almost entirely **one district**: `Jaffna` is zero across all five
GLDAS channels for all 459 weeks -- 1 in 25, which is where `docs/DATA.md`'s
"~4% missing" actually comes from. Interpolating along time cannot reach it, so
it is filled from its **graph neighbours**, and an indicator channel marks every
synthetic value.

**Leakage.** Covariates are normalised from training weeks only, and
`tests/test_features.py` pins that by perturbing the test tail and asserting the
training inputs do not move.

No second lag is applied: channels 6-10 are already shifted by 12 or 17 weeks.
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone"),
        code(_clone(f'ARCH = "{arch}"')),
        *[
            cell
            for i, fs in enumerate(FEATURE_SETS)
            for cell in (
                md(f"## {i + 3}. `features={fs}`"),
                code(
                    _invoke(
                        f'["--arch", ARCH, "--n-origins", "{N_ORIGINS}",'
                        f' "--seeds", "{ARCH_SEEDS.get(arch, SEEDS)}", "--features", "{fs}",'
                        f' "--out-dir", str(WORK), "--tag", ARCH]',
                        f"features={fs}",
                    )
                ),
            )
        ],
        md("## 6. Every arm against the floor"),
        code(_SUMMARY),
    ]


def build_preflight() -> list:
    """A fast smoke of every configuration, to be run before anything else.

    Each combination is exercised at ``--quick`` -- one seed, two origins, 25
    epochs -- purely to prove it constructs, trains and scores. The numbers are
    meaningless and must never be reported; what matters is that a shape error in
    a wide multivariate input surfaces in minutes rather than after a kernel has
    burned four hours.
    """
    combos = [
        ("A3TGCN", "det", "cases"),
        ("A3TGCN", "gauss", "cases"),
        ("A3TGCN", "nb", "cases"),
        ("A3TGCN", "det", "causal"),
        ("A3TGCN", "det", "climate"),
        ("STGAT", "det", "climate"),
        ("STGAT", "nb", "causal"),
        ("ASTGCN", "det", "climate"),
        ("AAGCN", "det", "causal"),
        ("DCRNN", "det", "cases"),
    ]
    checks = "\n".join(
        f'    ("{a}", "{h}", "{f}"),' for a, h, f in combos
    )
    return [
        md(
            """
# Preflight - does every configuration run at all

**Run this kernel first.** It exercises every architecture, head and feature-set
combination at `--quick` (1 seed, 2 origins, 25 epochs) and reports which ones
construct, train and score without error.

The numbers it produces are **degraded and must never be reported**. The only
output that matters is the PASS/FAIL table at the end. A wide multivariate input
changes the input width every architecture is built with -- 3 channels becomes
51 -- and that is exactly the kind of thing that fails at construction time,
after a full kernel has already spent hours on the runs before it.
"""
        ),
        md("## 1. Environment"),
        code(env_setup.PREAMBLE),
        code(env_setup.SETUP),
        code(env_setup.VERIFY_ENV),
        md("## 2. Clone"),
        code(_clone()),
        md("## 3. Smoke every configuration"),
        code(
            f'''
COMBOS = [
{checks}
]

results = []
for arch_name, head, fset in COMBOS:
    label = f"{{arch_name}}/{{head}}/{{fset}}"
    started = time.time()
    proc = subprocess.run(
        [str(PY311), "-u", str(RUNNER), "--arch", arch_name, "--head", head,
         "--features", fset, "--quick", "--out-dir", str(SCRATCH / "preflight"),
         "--tag", f"pre_{{arch_name}}_{{head}}_{{fset}}"],
        cwd=str(PROJ), capture_output=True, text=True,
    )
    ok = proc.returncode == 0
    tail = "" if ok else (proc.stdout + proc.stderr).strip().splitlines()[-1][:200]
    results.append((label, ok, time.time() - started, tail))
    print(f"{{'PASS' if ok else 'FAIL'}}  {{label:28s}} {{time.time() - started:6.1f}}s  {{tail}}",
          flush=True)

print()
n_ok = sum(1 for _, ok, _, _ in results if ok)
print(f"{{n_ok}}/{{len(results)}} configurations run.")
for label, ok, _, tail in results:
    if not ok:
        print(f"  FAILED {{label}}: {{tail}}")
assert n_ok == len(results), "fix the failures above before launching the full sweeps"
print("All configurations run. Safe to launch the full kernels.")
'''
        ),
    ]
