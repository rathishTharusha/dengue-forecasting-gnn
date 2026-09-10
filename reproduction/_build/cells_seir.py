"""Cell sources for the two compartmental-model reproductions.

Both are self-contained: the SEIR source prints all of its code, and the SEIR-SEI
paper is entirely closed-form. Neither needs Kaggle, a GPU, or any pinned stack —
they run in seconds on plain NumPy/SciPy. Kernels are provided so that all
reproducible papers can be re-run from one place, and so the numbers are produced
by a machine other than the one that first computed them.

The substance of both reproductions already lives in ``crosscheck/lib/xcheck``.
These notebooks restate the checks standalone, so a Kaggle kernel needs nothing
from this repository.
"""

from gen_kernels import code, md


def basic() -> list:
    """Gopalakrishnan, *The SEIR model of infectious diseases* (MTH 271, 2020)."""
    return [
        md(
            """
# Exact reproduction — The SEIR model of infectious diseases

**Source:** Jay Gopalakrishnan, *The SEIR model of infectious diseases*, MTH 271
course notes, Portland State University, 22 April 2020.

**Why this one is unambiguous.** The source is a lecture notebook that prints its
full source code. There are no missing hyperparameters, no unreleased data, and
no ambiguity about what was run — so "reproduce exactly" means exact numerical
agreement with the code as published.

`seir_f`, `seir_f2` and the parameter values below are transcribed verbatim from
the source. Everything else is the check harness.

## Claims under test

| # | Claim in the source |
|---|---|
| T1 | β=1, σ=1, γ=0.1 from (0.99, 0.01, 0, 0) gives a bell-shaped infection curve |
| T2 | Reducing β, raising γ, or lowering σ each flattens or delays the curve |
| T3 | Every equilibrium of the basic model is disease-free (`e = i = 0`) |
| T4 | Influx `a=0.005`, `b=0.001` gives an endemic equilibrium near 5% infected |
| T5 | `R0 = β·s₀/γ`; `R0 < 1` → no outbreak, `R0 > 1` → outbreak |
| T6 | Vaccinating a fraction `v` scales `R0` by `(1 − v)` |
"""
        ),
        code(
            """
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

CHECKS = []


def record(check_id, claim, expected, observed, passed):
    CHECKS.append((check_id, claim, expected, observed, bool(passed)))
    print(f"{check_id}  {'PASS' if passed else 'FAIL'}  {claim}")
    print(f"      expected: {expected}")
    print(f"      observed: {observed}")


# Transcribed verbatim from the source's seir_f and seir_f2.
def seir_f(t, y, beta, sigma, gamma):
    s, e, i, r = y
    return np.array([-beta * i * s,
                     -sigma * e + beta * i * s,
                     -gamma * i + sigma * e,
                     gamma * i])


def seir_f2(t, y, beta, sigma, gamma, a, b):
    s, e, i, r = y
    return np.array([-beta * i * s + a,
                     -sigma * e + beta * i * s + b,
                     -gamma * i + sigma * e,
                     gamma * i - (a + b)])


print("numpy", np.__version__)
"""
        ),
        md("## T1 — the baseline solve"),
        code(
            """
BETA, SIGMA, GAMMA = 1.0, 1.0, 0.1
Y0 = [0.99, 0.01, 0.0, 0.0]

sol = solve_ivp(seir_f, [0, 60], Y0, rtol=1e-6, args=(BETA, SIGMA, GAMMA))
i_peak = sol.y[2].max()
t_peak = sol.t[sol.y[2].argmax()]
total = sol.y[:, -1].sum()

bell = sol.y[2][0] < i_peak and sol.y[2][-1] < 0.05 * i_peak and 0 < t_peak < 60
record("T1", "beta=1, sigma=1, gamma=0.1 gives a bell-shaped infection curve",
       "interior peak, decays to near zero, mass conserved",
       f"peak i={i_peak:.4f} at t={t_peak:.2f}; final i={sol.y[2][-1]:.2e}; sum={total:.10f}",
       bell and abs(total - 1.0) < 1e-6)
"""
        ),
        md("## T2 — parameter study"),
        code(
            """
def solve_ei(beta=1.0, sigma=1.0, gamma=0.1, s0=0.99, e0=0.01, i0=0.0, r0=0.0, t1=60):
    s = solve_ivp(seir_f, [0, t1], [s0, e0, i0, r0], rtol=1e-7, args=(beta, sigma, gamma))
    return s.t, s.y[1], s.y[2]


rows = []
for name, kw in [("baseline", {}), ("beta=0.5", dict(beta=0.5)),
                 ("gamma=0.5", dict(gamma=0.5)), ("sigma=0.1", dict(sigma=0.1))]:
    t, e, i = solve_ei(**kw)
    rows.append((name, i.max(), t[i.argmax()]))
    print(f"{name:12s} peak i={i.max():.4f} at t={t[i.argmax()]:6.2f}")

base_peak = rows[0][1]
record("T2", "reducing beta, raising gamma, lowering sigma each flatten or delay the curve",
       "beta=0.5 flatter; gamma=0.5 flatter; sigma=0.1 later peak",
       f"{rows[1][1]:.4f}, {rows[2][1]:.4f}, t={rows[3][2]:.2f}",
       rows[1][1] < base_peak and rows[2][1] < base_peak and rows[3][2] > rows[0][2])
"""
        ),
        md("## T3 — every equilibrium is disease-free"),
        code(
            """
residual = seir_f(0.0, np.array([0.4, 0.0, 0.0, 0.6]), BETA, SIGMA, GAMMA)
long_sol = solve_ivp(seir_f, [0, 400], Y0, rtol=1e-9, args=(BETA, SIGMA, GAMMA))
e_end, i_end = long_sol.y[1, -1], long_sol.y[2, -1]

# At t=400 the solver leaves |e|,|i| ~ 1e-7 and i can be slightly negative: that
# is integration noise at rtol=1e-9, not a surviving infection.
settled = max(abs(e_end), abs(i_end))
record("T3", "every equilibrium of the basic SEIR model is disease-free",
       "RHS residual = 0 at (s, 0, 0, r); long solve drives e, i -> 0",
       f"max|residual|={np.abs(residual).max():.2e}; settled={settled:.2e}",
       np.abs(residual).max() < 1e-15 and settled < 1e-5)
"""
        ),
        md("## T4 — endemic equilibrium under influx"),
        code(
            """
endemic = solve_ivp(seir_f2, [0, 150], Y0, rtol=1e-7, args=(BETA, SIGMA, GAMMA, 0.005, 0.001))
tail = endemic.t >= 120
plateau = float(endemic.y[2][tail].mean())
drift = float(endemic.y[2][tail].max() - endemic.y[2][tail].min())

record("T4", "influx turns the disease-free equilibrium into an endemic one near 5%",
       "i settles at approximately 0.05, not decaying to 0",
       f"mean i over t>=120 is {plateau:.4f} (drift {drift:.4f})",
       0.03 <= plateau <= 0.07 and drift < 0.02)
"""
        ),
        md("## T5 and T6 — the R0 threshold and the vaccination threshold"),
        code(
            """
def seir_r0(beta, gamma, s0=1.0, vaccinated=0.0):
    return beta * s0 * (1.0 - vaccinated) / gamma


observed = []
for name, kw in [("no outbreak", dict(beta=0.6, gamma=1.0, s0=0.9, i0=0.1, e0=0.0)),
                 ("outbreak", dict(beta=1.0, gamma=0.5, s0=0.9, i0=0.1, e0=0.0))]:
    t, e, i = solve_ei(t1=60, **kw)
    ei = e + i
    r0 = seir_r0(kw["beta"], kw["gamma"], kw["s0"])
    grew = ei.max() > ei[0] * 1.05
    observed.append((r0, grew))
    print(f"{name:12s} R0={r0:.3f}  max(e+i)={ei.max():.3f}  grew={grew}")

record("T5", "R0 = beta*s0/gamma separates outbreak from no outbreak",
       "R0=0.54 -> no growth; R0=1.80 -> growth",
       f"R0={observed[0][0]:.2f} grew={observed[0][1]}; "
       f"R0={observed[1][0]:.2f} grew={observed[1][1]}",
       (not observed[0][1]) and observed[1][1])

beta_v, gamma_v, s0_v = 1.0, 0.5, 0.9
v_star = 1.0 - gamma_v / (beta_v * s0_v)
below = seir_r0(beta_v, gamma_v, s0_v, vaccinated=v_star + 1e-6)
above = seir_r0(beta_v, gamma_v, s0_v, vaccinated=v_star - 1e-6)
record("T6", "vaccinating fraction v scales R0 by (1 - v); threshold at R0 = 1",
       f"threshold v* = 1 - gamma/(beta*s0) = {v_star:.4f}",
       f"R0(v*+eps)={below:.6f} < 1 < R0(v*-eps)={above:.6f}",
       below < 1.0 < above)
"""
        ),
        md("## Verdict"),
        code(
            """
from pathlib import Path

verdict = pd.DataFrame(CHECKS, columns=["check", "claim", "expected", "observed", "passed"])
print(f"{int(verdict.passed.sum())}/{len(verdict)} checks passed")

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
verdict.to_csv(out / "seir_basic_checks.csv", index=False)
print("wrote", out / "seir_basic_checks.csv")
verdict[["check", "passed"]]
"""
        ),
    ]


def sei_sensitivity() -> list:
    """Phaijoo & Gurung, SEIR-SEI sensitivity analysis (GAMS 6(a), 2018)."""
    return [
        md(
            """
# Exact reproduction — Sensitivity analysis of the SEIR–SEI dengue model

**Source:** Ganga Ram Phaijoo & Dil Bahadur Gurung, *Sensitivity Analysis of
SEIR–SEI Model of Dengue Disease*, GAMS Journal of Mathematics and Mathematical
Biosciences **6(a)**, 41–50, December 2018.

Entirely closed-form: no code or data to obtain, so a reproduction either matches
to six decimals or something is wrong.

## Targets

| # | Target | Source |
|---|---|---|
| T1 | `R0 = sqrt(α δ ν_h ν_v / (β γ ε (ε + ν_v)))` equals `ρ(F V⁻¹)` | §3 |
| T2 | The nine sensitivity indices of Table 1 | §3, Table 1 |
| T3 | Closed-form indices agree with finite differences | independent check |

## Model, in the paper's notation

```
α = b·β_h·π_v / (N_h·μ_v)     β = ν_h + μ_h
γ = γ_h + μ_h                  δ = b·β_v          ε = μ_v
```
"""
        ),
        code(
            """
import numpy as np
import pandas as pd

CHECKS = []


def record(check_id, claim, expected, observed, passed):
    CHECKS.append((check_id, claim, expected, observed, bool(passed)))
    print(f"{check_id}  {'PASS' if passed else 'FAIL'}  {claim}")


# Parameter values listed for the simulations in Section 4.
P = dict(N_h=5_071_126, b=0.5, beta_h=0.75, beta_v=0.375, mu_h=0.000046,
         mu_v=0.25, nu_h=0.1667, nu_v=0.1428, gamma_h=0.328833, pi_v=2_500_000)

# The Sensitivity Indices column of Table 1, as printed.
TABLE1 = {"pi_v": +0.5, "b": +1.0, "beta_v": +0.5, "mu_v": -1.31823,
          "nu_v": +0.318228, "beta_h": +0.5, "gamma_h": -0.49993,
          "mu_h": -0.000208, "nu_h": +0.000138}


def composites(p):
    return dict(alpha=p["b"] * p["beta_h"] * p["pi_v"] / (p["N_h"] * p["mu_v"]),
                beta=p["nu_h"] + p["mu_h"],
                gamma=p["gamma_h"] + p["mu_h"],
                delta=p["b"] * p["beta_v"],
                epsilon=p["mu_v"])


for k, v in composites(P).items():
    print(f"{k:8s} = {v:.8f}")
"""
        ),
        md("## T1 — `R0` two independent ways"),
        code(
            """
def r0_closed(p):
    c = composites(p)
    num = c["alpha"] * c["delta"] * p["nu_h"] * p["nu_v"]
    den = c["beta"] * c["gamma"] * c["epsilon"] * (c["epsilon"] + p["nu_v"])
    return float(np.sqrt(num / den))


def r0_spectral(p):
    c = composites(p)
    F = np.array([[0, 0, 0, c["alpha"]], [0, 0, c["delta"], 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                 dtype=float)
    V = np.array([[c["beta"], 0, 0, 0],
                  [0, c["epsilon"] + p["nu_v"], 0, 0],
                  [-p["nu_h"], 0, c["gamma"], 0],
                  [0, -p["nu_v"], 0, c["epsilon"]]], dtype=float)
    return float(max(abs(np.linalg.eigvals(F @ np.linalg.inv(V)))))


a, b = r0_closed(P), r0_spectral(P)
print(f"R0 (closed form) = {a!r}")
print(f"R0 (rho(F V^-1)) = {b!r}")
record("T1", "closed-form R0 equals the next-generation spectral radius",
       "identical to floating-point precision", f"diff {abs(a - b):.2e}", abs(a - b) < 1e-12)
"""
        ),
        md("## T2 and T3 — Table 1's nine sensitivity indices"),
        code(
            """
def indices_closed(p):
    mh, mv, nh, nv, gh = p["mu_h"], p["mu_v"], p["nu_h"], p["nu_v"], p["gamma_h"]
    return {"pi_v": 0.5, "b": 1.0, "beta_v": 0.5, "beta_h": 0.5,
            "mu_v": -0.5 * (2.0 + mv / (mv + nv)),
            "nu_v": 0.5 * mv / (mv + nv),
            "gamma_h": -0.5 * gh / (gh + mh),
            "nu_h": 0.5 * mh / (nh + mh),
            "mu_h": -0.5 * (mh / (nh + mh) + mh / (gh + mh))}


def indices_numeric(p, rel_step=1e-6):
    base = r0_closed(p)
    out = {}
    for name in TABLE1:
        q = p[name]
        h = q * rel_step
        up, dn = dict(p), dict(p)
        up[name], dn[name] = q + h, q - h
        out[name] = (r0_closed(up) - r0_closed(dn)) / (2 * h) * q / base
    return out


closed, numeric = indices_closed(P), indices_numeric(P)
table = pd.DataFrame([{"parameter": k, "paper Table 1": v,
                       "reproduced (closed form)": closed[k],
                       "reproduced (finite diff)": numeric[k],
                       "abs error vs paper": abs(closed[k] - v)} for k, v in TABLE1.items()])
pd.set_option("display.float_format", lambda x: f"{x: .6f}")
print(table.to_string(index=False))

worst = table["abs error vs paper"].max()
record("T2", "all nine Table 1 sensitivity indices reproduce",
       "agreement to the paper's printed precision (<= 5e-6)",
       f"largest absolute error {worst:.2e}", worst <= 5e-6)
record("T3", "closed-form indices agree with central finite differences", "~1e-6",
       "see table",
       np.allclose(table["reproduced (closed form)"], table["reproduced (finite diff)"],
                   atol=1e-6))
"""
        ),
        md("## Verdict"),
        code(
            """
from pathlib import Path

verdict = pd.DataFrame(CHECKS, columns=["check", "claim", "expected", "observed", "passed"])
print(f"{int(verdict.passed.sum())}/{len(verdict)} checks passed")

out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
verdict.to_csv(out / "seir_sei_checks.csv", index=False)
table.to_csv(out / "seir_sei_table1_indices.csv", index=False)
print("wrote", out / "seir_sei_checks.csv", "and", out / "seir_sei_table1_indices.csv")
table
"""
        ),
        md(
            """
## Note on defects found

This reproduction succeeds, but the paper has three internal inconsistencies,
documented with evidence in `crosscheck/FINDINGS.md` (F4.2–F4.4):

1. Table 1's "Baseline Values" column does not produce Table 1's own indices —
   the indices follow the §4 simulation values instead.
2. Those §4 values give `R0 ≈ 0.78 < 1`, a disease-free regime, which sits badly
   with Figures 2–3.
3. The printed endemic equilibrium is not a fixed point in its vector
   components: as printed `e_v*/i_v* = ε`, while `di_v/dt = 0` requires `ε/ν_v`.

None of the three affect the results reproduced above.
"""
        ),
    ]
