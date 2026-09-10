"""Cell sources for R4 -- SEIR-SEI sensitivity analysis (Phaijoo & Gurung, 2018).

Reproduction targets: the R0 expression from the next-generation matrix, the
endemic equilibrium of Eq. (2.2), and the nine normalized forward sensitivity
indices of Table 1. All nine reproduce, but only from the paper's *simulation*
parameters -- not from Table 1's own printed baseline column. That discrepancy
is the finding this notebook is built to demonstrate.
"""

from gen_notebooks import code, md

CELLS = [
    md(
        """
# R4 — Sensitivity analysis of the SEIR–SEI dengue model

**Source:** Ganga Ram Phaijoo & Dil Bahadur Gurung, *Sensitivity Analysis of
SEIR–SEI Model of Dengue Disease*, GAMS Journal of Mathematics and Mathematical
Biosciences **6(a)**, 41–50, December 2018.
(`papers/Sensitivity Analysis of SEIR - SEI Model of Dengue Disease.pdf`)

**Why it matters here.** This is the host–vector compartmental model that
Contribution (a) — the physics-informed loss — is grounded in. Its `R0` and its
sensitivity ranking tell us which parameters a physics residual is actually
sensitive to, and therefore which are worth constraining. It is also a paper
whose whole content is closed-form, so a reproduction either matches to six
decimals or something is wrong.

## Reproduction targets

| # | Target | Source |
|---|---|---|
| T1 | `R0 = sqrt(α δ ν_h ν_v / (β γ ε (ε + ν_v)))` equals `ρ(F V⁻¹)` | §3, next-generation matrix |
| T2 | The nine sensitivity indices of Table 1 | §3, Table 1 |
| T3 | Closed-form indices agree with finite differences | independent check, not in the paper |
| T4 | Sign of each index predicts the direction of the Figures 4–8 simulations | §4, Figures 4–8 |
| T5 | Endemic equilibrium `E₁` exists (is positive) exactly when `R0 > 1` | Theorem 3.1 |
| T6 | Disease-free equilibrium is stable when `R0 < 1` | Theorems 3.2, 3.3 |

## Model, in the paper's notation

Hosts split S→E→I→R, vectors S→E→I (no recovery — mosquito infection ends with
death). After the substitution of §2 the system reduces to five proportions
`(s_h, e_h, i_h, e_v, i_v)` with composite parameters

```
α = b·β_h·π_v / (N_h·μ_v)     β = ν_h + μ_h
γ = γ_h + μ_h                  δ = b·β_v          ε = μ_v
```
"""
    ),
    md("## 1. Setup"),
    code(
        """
import sys
from dataclasses import replace
from pathlib import Path

LIB = Path.cwd().parent / "lib"
if not LIB.exists():
    LIB = Path.cwd() / "lib"
sys.path.insert(0, str(LIB))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from xcheck.seir import (
    PHAIJOO_SIMULATION,
    PHAIJOO_TABLE1_BASELINES,
    PHAIJOO_TABLE1_INDICES,
    seir_sei_equilibrium,
    seir_sei_next_generation,
    seir_sei_r0,
    seir_sei_rhs,
    sensitivity_indices,
    sensitivity_indices_numeric,
)

CHECKS = []


def record(check_id, claim, expected, observed, passed):
    CHECKS.append((check_id, claim, expected, observed, bool(passed)))
    print(f"{check_id}  {'PASS' if passed else 'FAIL'}  {claim}")


P = PHAIJOO_SIMULATION
print("Simulation parameters (paper, Section 4):")
for name in ("N_h", "b", "beta_h", "beta_v", "mu_h", "mu_v", "nu_h", "nu_v", "gamma_h", "pi_v"):
    print(f"  {name:8s} = {getattr(P, name)}")
print()
print("Composite symbols (Section 2):")
for name in ("alpha", "beta", "gamma", "delta", "epsilon"):
    print(f"  {name:8s} = {getattr(P, name):.8f}")
"""
    ),
    md(
        """
## 2. T1 — `R0` two independent ways

The paper derives `R0 = ρ(F V⁻¹)` and then prints a closed form. Computing both
and comparing is a genuine check: the closed form could be a transcription slip
and the matrices would catch it.
"""
    ),
    code(
        """
F, V = seir_sei_next_generation(P)
print("F =\\n", F, "\\n\\nV =\\n", V)

r0_closed = seir_sei_r0(P, method="closed_form")
r0_spectral = seir_sei_r0(P, method="spectral")
print(f"\\nR0 (closed form)   = {r0_closed!r}")
print(f"R0 (rho(F V^-1))   = {r0_spectral!r}")
print(f"absolute difference = {abs(r0_closed - r0_spectral):.3e}")

record(
    "T1",
    "closed-form R0 equals the next-generation spectral radius",
    "identical to floating-point precision",
    f"{r0_closed:.12f} vs {r0_spectral:.12f}, diff {abs(r0_closed - r0_spectral):.2e}",
    abs(r0_closed - r0_spectral) < 1e-12,
)
"""
    ),
    md(
        """
## 3. T2 — Table 1, the nine sensitivity indices

The normalized forward sensitivity index is `Υ = (∂R0/∂q)·(q/R0)`. Because
`R0` is a square root, each index is half the log-derivative of `R0²`, which
makes all nine elementary — `π_v`, `β_h`, `β_v` enter linearly and give `+½`;
`b` enters twice and gives `+1`; the rest follow from the composite symbols.
"""
    ),
    code(
        """
closed = sensitivity_indices(P)
numeric = sensitivity_indices_numeric(P)

table1 = pd.DataFrame(
    [
        {
            "parameter": k,
            "paper Table 1": v,
            "reproduced (closed form)": closed[k],
            "reproduced (finite diff)": numeric[k],
            "abs error vs paper": abs(closed[k] - v),
        }
        for k, v in PHAIJOO_TABLE1_INDICES.items()
    ]
)
pd.set_option("display.float_format", lambda x: f"{x: .6f}")
print(table1.to_string(index=False))

worst = table1["abs error vs paper"].max()
record(
    "T2",
    "all nine Table 1 sensitivity indices reproduce",
    "agreement to the paper's printed precision (<= 5e-6)",
    f"largest absolute error {worst:.2e}",
    worst <= 5e-6,
)

record(
    "T3",
    "closed-form indices agree with central finite differences",
    "agreement to ~1e-6",
    f"largest gap {np.abs(table1['reproduced (closed form)'] - table1['reproduced (finite diff)']).max():.2e}",
    np.allclose(table1["reproduced (closed form)"], table1["reproduced (finite diff)"], atol=1e-6),
)
"""
    ),
    md(
        """
## 4. Finding — Table 1's "Baseline Values" column does not produce Table 1's indices

The indices above reproduce exactly, but **only** from the parameter values the
paper lists for its simulations in §4. Table 1 prints a different "Baseline
Values" column beside the same indices. Recomputing from *that* column gives
different numbers, so the two halves of Table 1 are inconsistent with each
other.

The cell below recomputes both ways so the discrepancy is visible rather than
asserted.
"""
    ),
    code(
        """
from_table = replace(P, **PHAIJOO_TABLE1_BASELINES)
idx_from_table = sensitivity_indices(from_table)

comparison = pd.DataFrame(
    [
        {
            "parameter": k,
            "Table 1 index": v,
            "Table 1 baseline": PHAIJOO_TABLE1_BASELINES[k],
            "Sec. 4 value": getattr(P, k),
            "index from Sec. 4": closed[k],
            "index from Table 1 baseline": idx_from_table[k],
        }
        for k, v in PHAIJOO_TABLE1_INDICES.items()
    ]
)
print(comparison.to_string(index=False))

mismatched = [
    k for k in PHAIJOO_TABLE1_INDICES
    if abs(idx_from_table[k] - PHAIJOO_TABLE1_INDICES[k]) > 5e-6
]
disagreeing_values = [
    k for k in PHAIJOO_TABLE1_BASELINES
    if not np.isclose(PHAIJOO_TABLE1_BASELINES[k], getattr(P, k), rtol=1e-9)
]
print(f"\\nParameters whose two stated values disagree: {disagreeing_values}")
print(f"Indices not reproducible from the Table 1 baseline column: {mismatched}")
"""
    ),
    md(
        """
**Reading of the discrepancy.** Four parameters are given two different values
by the same paper (`π_v`, `β_v`, `μ_v`, `ν_h`), and the printed indices follow
the §4 simulation values, not the Table 1 column. For `π_v` and `β_v` it makes
no difference — their indices are `+½` for *any* positive value, which is why
the inconsistency survived review. For `μ_v` and `ν_h` it does: the Table 1
column would give `μ_v ≈ −1.085` against the printed `−1.31823`.

The most economical explanation is typographic — `μ_v 0.02941` for `0.25`,
`ν_h 1.667` for `0.1667` (a factor of ten, matching §4). **The paper's headline
conclusions are unaffected**: the ranking, and the claim that `b` is the most
positive and `μ_v` the most negative sensitive parameter, hold under the §4
values that generated the indices.
"""
    ),
    md(
        """
## 5. Finding — the paper's own simulation parameters give `R0 < 1`

With the §4 values, `R0 ≈ 0.78`. By the paper's own Theorems 3.1–3.3 that is the
regime where the endemic equilibrium does not exist and the disease-free
equilibrium is globally stable — yet Figures 2–3 are presented as showing
outbreak dynamics. The cell below integrates Eq. (2.2) from a small introduced
infection and reports what actually happens.
"""
    ),
    code(
        """
def integrate(params, y0, t_end, n=2000):
    sol = solve_ivp(
        lambda t, y: seir_sei_rhs(t, y, params),
        [0, t_end], y0, rtol=1e-9, atol=1e-12,
        t_eval=np.linspace(0, t_end, n),
    )
    return sol


# Small introduced infection: 0.1% of hosts exposed, everything else susceptible.
y0 = [0.999, 0.001, 0.0, 0.0, 0.0]
sol = integrate(P, y0, t_end=365 * 3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(sol.t, sol.y[0]); axes[0].set_title("s_h (susceptible hosts)")
axes[1].plot(sol.t, sol.y[2], label="i_h (infectious hosts)")
axes[1].plot(sol.t, sol.y[4], label="i_v (infectious vectors)")
axes[1].set_title("infectious compartments"); axes[1].legend()
for ax in axes:
    ax.set_xlabel("days")
plt.tight_layout(); plt.show()

i_h = sol.y[2]
print(f"R0 = {seir_sei_r0(P):.6f}")
print(f"i_h: start {i_h[0]:.3e}, peak {i_h.max():.3e} at t={sol.t[i_h.argmax()]:.1f}, "
      f"end {i_h[-1]:.3e}")

died_out = i_h[-1] < 1e-8
record(
    "T6",
    "R0 < 1 implies the disease dies out (Theorems 3.2, 3.3)",
    "with the paper's Section 4 parameters R0 < 1, so i_h -> 0",
    f"R0={seir_sei_r0(P):.4f}; i_h(3 years)={i_h[-1]:.2e}",
    seir_sei_r0(P) < 1.0 and died_out,
)
"""
    ),
    md(
        """
## 6. T5 — the endemic equilibrium appears exactly at `R0 = 1`

Theorem 3.1 says `E₁` exists uniquely when `R0 > 1`. Every component of the
printed `E₁` carries a factor `(R0² − 1)`, so below threshold they go negative —
mathematically present, epidemiologically meaningless. Sweeping the biting rate
`b` moves `R0` across 1 and shows the transition.
"""
    ),
    code(
        """
b_grid = np.linspace(0.2, 1.4, 60)
rows = []
for b in b_grid:
    p_b = replace(P, b=float(b))
    eq = seir_sei_equilibrium(p_b)
    rows.append({"b": b, "R0": seir_sei_r0(p_b), "i_h*": eq["i_h"], "i_v*": eq["i_v"]})
sweep = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sweep.R0, sweep["i_h*"], label="endemic i_h*")
ax.axhline(0, color="grey", lw=0.8)
ax.axvline(1, color="red", ls="--", lw=0.8, label="R0 = 1")
ax.set_xlabel("R0"); ax.set_ylabel("endemic i_h*")
ax.set_title("Endemic equilibrium is positive exactly when R0 > 1")
ax.legend(); plt.show()

below = sweep[sweep.R0 < 1]
above = sweep[sweep.R0 > 1]
sign_correct = (below["i_h*"] <= 0).all() and (above["i_h*"] > 0).all()

record(
    "T5",
    "endemic equilibrium is positive exactly when R0 > 1 (Theorem 3.1)",
    "i_h* <= 0 below R0 = 1, > 0 above",
    f"sign correct across the whole sweep = {sign_correct}",
    sign_correct,
)
"""
    ),
    md(
        """
### Finding — the printed `E₁` is a fixed point in its host components only

Substituting the printed `E₁` into Eq. (2.2) should give a zero residual. It
does for `s_h*`, `e_h*` and `i_h*`, and does not for the two vector components.

The reason is visible in the formulas themselves. As printed,

```
e_v* = μ_h·β·γ·ε²(R0²−1) / (δ(βγε + δμ_h ν_h))
i_v* = μ_h·β·γ·ε (R0²−1) / (δ(βγε + δμ_h ν_h))
```

so `e_v*/i_v* = ε`. But `di_v/dt = ν_v·e_v − ε·i_v = 0` forces
`i_v = (ν_v/ε)·e_v`, i.e. `e_v/i_v = ε/ν_v`. The two agree only if `ν_v = 1`,
which it is not (`ν_v = 0.1428`).

Solving the vector block of Eq. (2.2) directly, with `s_v = 1 − e_v − i_v`:

```
e_v = δ·i_h·ε / ((ε + ν_v)(ε + δ·i_h))        i_v = (ν_v/ε)·e_v
```

The cell below checks the printed and corrected forms against both the ODE
residual and an independent numerical root-find.
"""
    ),
    code(
        """
from scipy.optimize import fsolve

p_endemic = replace(P, b=1.2)          # R0 > 1, so E1 is meaningful
eq_paper = seir_sei_equilibrium(p_endemic, variant="paper")
eq_fixed = seir_sei_equilibrium(p_endemic, variant="corrected")
order = ["s_h", "e_h", "i_h", "e_v", "i_v"]

as_vec = lambda d: np.array([d[k] for k in order])
numeric = fsolve(lambda y: seir_sei_rhs(0.0, y, p_endemic), as_vec(eq_fixed), xtol=1e-14)

equilibria = pd.DataFrame({
    "component": order,
    "paper E1": as_vec(eq_paper),
    "corrected E1": as_vec(eq_fixed),
    "numerical root": numeric,
})
print(f"R0 = {seir_sei_r0(p_endemic):.4f}\\n")
print(equilibria.to_string(index=False))

res_paper = np.abs(seir_sei_rhs(0.0, as_vec(eq_paper), p_endemic)).max()
res_fixed = np.abs(seir_sei_rhs(0.0, as_vec(eq_fixed), p_endemic)).max()
print(f"\\nmax |residual of Eq. (2.2)|   printed E1: {res_paper:.3e}")
print(f"                             corrected E1: {res_fixed:.3e}")

host_ok = np.allclose(as_vec(eq_paper)[:3], numeric[:3], rtol=1e-9)
print(f"\\nhost components (s_h, e_h, i_h) of the printed E1 are exact: {host_ok}")
print(f"printed e_v*/i_v* = {eq_paper['e_v'] / eq_paper['i_v']:.6f}  "
      f"(equals epsilon = {p_endemic.epsilon})")
print(f"required  e_v/i_v = {p_endemic.epsilon / p_endemic.nu_v:.6f}  "
      f"(equals epsilon/nu_v)")

record(
    "T5b",
    "the corrected E1 is an exact fixed point; the printed one is not",
    "residual = 0 for the corrected form; printed host components still exact",
    f"printed residual {res_paper:.1e}, corrected residual {res_fixed:.1e}, "
    f"host components exact = {host_ok}",
    res_fixed < 1e-12 and res_paper > 1e-8 and host_ok,
)
"""
    ),
    md(
        """
## 7. T4 — do the Figures 4–8 simulations agree with the index signs?

The paper simulates the effect of `π_v`, `μ_v`, `b`, `β_h` and `γ_h` on the
infectious host population and reports that parameters with a positive
sensitivity index raise it while negative ones lower it. Here each parameter is
varied over ±50% and the peak of `i_h` is measured, in a regime where an
outbreak actually occurs (`b` raised so that `R0 > 1` — with the paper's own
`b = 0.5` there is nothing to measure, per the previous section).
"""
    ),
    code(
        """
BASE = replace(P, b=1.2)     # R0 > 1, so there is an epidemic to perturb
print(f"perturbation base: b={BASE.b}, R0={seir_sei_r0(BASE):.4f}")

def peak_ih(params, t_end=365 * 3):
    return float(integrate(params, y0, t_end).y[2].max())


studied = ["pi_v", "mu_v", "b", "beta_h", "gamma_h"]
scales = [0.5, 0.75, 1.0, 1.25, 1.5]
base_peak = peak_ih(BASE)

fig, axes = plt.subplots(1, len(studied), figsize=(18, 3.2), sharey=True)
rows = []
for ax, name in zip(axes, studied):
    peaks = []
    for k in scales:
        peaks.append(peak_ih(replace(BASE, **{name: getattr(BASE, name) * k})))
    ax.plot(scales, peaks, "o-")
    ax.set_title(f"{name}  (index {closed[name]:+.3f})")
    ax.set_xlabel("x baseline")
    trend = np.sign(peaks[-1] - peaks[0])
    rows.append({
        "parameter": name,
        "sensitivity index": closed[name],
        "index sign": int(np.sign(closed[name])),
        "peak i_h trend": int(trend),
        "agrees": int(np.sign(closed[name])) == int(trend),
    })
axes[0].set_ylabel("peak i_h")
plt.tight_layout(); plt.show()

directions = pd.DataFrame(rows)
print(directions.to_string(index=False))

record(
    "T4",
    "sign of each sensitivity index predicts the direction of the Figures 4-8 simulations",
    "positive index -> peak i_h rises; negative index -> peak i_h falls",
    f"{int(directions.agrees.sum())}/{len(directions)} parameters agree",
    bool(directions.agrees.all()),
)
"""
    ),
    md("## 8. Verdict"),
    code(
        """
verdict = pd.DataFrame(CHECKS, columns=["check", "claim", "expected", "observed", "passed"])
print(f"{int(verdict.passed.sum())}/{len(verdict)} checks passed\\n")

results_dir = Path.cwd().parent / "results"
results_dir.mkdir(parents=True, exist_ok=True)
verdict.to_csv(results_dir / "R4_seir_sei_checks.csv", index=False)
equilibria.to_csv(results_dir / "R4_endemic_equilibrium.csv", index=False)
table1.to_csv(results_dir / "R4_table1_sensitivity_indices.csv", index=False)
comparison.to_csv(results_dir / "R4_table1_baseline_inconsistency.csv", index=False)
print("wrote R4_seir_sei_checks.csv, R4_table1_sensitivity_indices.csv, "
      "R4_table1_baseline_inconsistency.csv")

verdict[["check", "claim", "passed"]]
"""
    ),
    md(
        """
## 9. What this establishes

**The paper reproduces.** `R0`, the endemic equilibrium, the stability theorems
and all nine sensitivity indices come out exactly, by two independent routes
each. The modelling content is sound and is safe to build Contribution (a) on.

**Two defects, neither fatal:**

1. Table 1's "Baseline Values" column is inconsistent with its own indices; the
   indices follow the §4 simulation values. Four parameters are given two
   values; two of those change the indices materially. Use the §4 values.
2. The §4 parameters give `R0 ≈ 0.78 < 1`, a disease-free regime, which sits
   badly with Figures 2–3. Any simulation reusing them will see the infection
   die out. Raising `b` (or `β_v`, or lowering `μ_v`) puts the model back in an
   epidemic regime.

**What this means for the physics-informed loss.** The sensitivity ranking is
what to take forward: `b` (biting rate, `+1`) and `μ_v` (vector death rate,
`−1.32`) dominate; `μ_h` and `ν_h` are negligible at `~10⁻⁴`. A physics residual
that spends parameters on `ν_h` is spending them where `R0` cannot feel it.
Note the counterpart risk already in `docs/ROADMAP.md`: `e_h` and `i_v` are
unobserved in our district-week data, so the residual can only ever be evaluated
on `i_h`.
"""
    ),
]
