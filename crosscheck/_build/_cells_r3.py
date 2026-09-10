"""Cell sources for R3 -- the basic SEIR model (Gopalakrishnan, MTH 271, 2020).

Reproduction target: the source is a lecture notebook that prints all of its
code, so the target is exact numerical agreement with that code, plus the four
qualitative claims it makes (bell curve, disease-free equilibrium, endemic
equilibrium under influx, and the R0 outbreak threshold).
"""

from gen_notebooks import code, md

CELLS = [
    md(
        """
# R3 — The SEIR model of infectious diseases

**Source:** Jay Gopalakrishnan, *The SEIR model of infectious diseases*,
MTH 271 course notes, Portland State University, 22 April 2020.
(`papers/09_SEIR_model.pdf`)

**What this is.** Not a research paper — a lecture notebook that prints its full
source. That makes it the one item in this workspace where "reproduce the
paper's results" means *exact numerical agreement*, with no missing
hyperparameters to guess. It is included because the SEIR compartmental
structure is the backbone of Contribution (a), the physics-informed loss, and
because it is a clean calibration check that this workspace's tooling is sound
before it is pointed at harder targets.

## Reproduction targets

| # | Claim in the source | How this notebook checks it |
|---|---|---|
| T1 | `seir_f` with β=1, σ=1, γ=0.1 from (0.99, 0.01, 0, 0) produces a bell-shaped infection curve over [0, 60] | Integrate and locate the peak |
| T2 | Reducing β, raising γ, or lowering σ each flattens or delays the curve | Re-solve per variation, compare peak height and time |
| T3 | Every equilibrium of the basic model is disease-free (`e = i = 0`) | Residual of the RHS at the claimed equilibrium; long-horizon solve |
| T4 | Adding influx `a=0.005`, `b=0.001` gives an **endemic** equilibrium near 5% infected | Integrate to t=150 and read the plateau |
| T5 | `R0 = β·s₀/γ`; `R0 < 1` → no outbreak, `R0 > 1` → outbreak | Two solves: (β=0.6, γ=1, s₀=0.9, i₀=0.1) and (β=1, γ=0.5, s₀=0.9, i₀=0.1) |
| T6 | Vaccinating a fraction `v` scales `R0` by `(1 − v)` | Solve for the `v` that brings `R0` under 1 |

Every check writes a PASS/FAIL row into the verdict table at the end.
"""
    ),
    md("## 1. Setup"),
    code(
        """
import sys
from pathlib import Path

# The workspace library lives in crosscheck/lib and is deliberately NOT the
# project's src/dengue_gnn. Add it to the path rather than installing it.
LIB = Path.cwd().parent / "lib"
if not LIB.exists():                      # tolerate being run from crosscheck/
    LIB = Path.cwd() / "lib"
sys.path.insert(0, str(LIB))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from xcheck.seir import seir_r0, seir_rhs, seir_rhs_influx

CHECKS = []  # (id, claim, expected, observed, passed)


def record(check_id, claim, expected, observed, passed):
    CHECKS.append((check_id, claim, expected, observed, bool(passed)))
    print(f"{check_id}  {'PASS' if passed else 'FAIL'}  {claim}")
    print(f"      expected: {expected}")
    print(f"      observed: {observed}")


print("numpy", np.__version__)
"""
    ),
    md(
        """
## 2. T1 — the baseline solve

The source's parameters are β=1, σ=1, γ=0.1 with 1% of the population exposed
at t=0, integrated over [0, 60] at `rtol=1e-6`. Reproduced exactly.
"""
    ),
    code(
        """
BETA, SIGMA, GAMMA = 1.0, 1.0, 0.1
Y0 = [0.99, 0.01, 0.0, 0.0]

sol = solve_ivp(seir_rhs, [0, 60], Y0, rtol=1e-6, args=(BETA, SIGMA, GAMMA))

fig, ax = plt.subplots(figsize=(7, 4))
for row, name in zip(sol.y, ["S", "E", "I", "R"]):
    ax.plot(sol.t, row, label=name)
ax.set_xlabel("time"); ax.set_ylabel("proportion")
ax.set_title("SEIR, beta=1, sigma=1, gamma=0.1")
ax.legend(); plt.show()

i_peak = sol.y[2].max()
t_peak = sol.t[sol.y[2].argmax()]
final = sol.y[:, -1]

# A bell curve: infections rise from ~0, peak in the interior, and fall back.
bell = (
    sol.y[2][0] < i_peak
    and sol.y[2][-1] < 0.05 * i_peak
    and 0 < t_peak < 60
)
record(
    "T1",
    "beta=1, sigma=1, gamma=0.1 gives a bell-shaped infection curve",
    "interior peak, decays to near zero, mass conserved",
    f"peak i={i_peak:.4f} at t={t_peak:.2f}; final i={sol.y[2][-1]:.2e}; "
    f"sum(S,E,I,R)={final.sum():.10f}",
    bell and abs(final.sum() - 1.0) < 1e-6,
)
"""
    ),
    md(
        """
## 3. T2 — parameter study

The source varies one parameter at a time: `beta=0.5`, `gamma=0.5`, `sigma=0.1`.
Its claims are qualitative ("what happens if β is reduced?"), so the check is on
the direction of the change in peak height and timing.
"""
    ),
    code(
        """
def solve_ei(beta=1.0, sigma=1.0, gamma=0.1, s0=0.99, e0=0.01, i0=0.0, r0=0.0, t1=60):
    \"\"\"Solve and return (t, e, i), mirroring the source's plot_ei.\"\"\"
    s = solve_ivp(
        seir_rhs, [0, t1], [s0, e0, i0, r0], rtol=1e-7, args=(beta, sigma, gamma)
    )
    return s.t, s.y[1], s.y[2]


variants = {
    "baseline (b=1, s=1, g=0.1)": dict(),
    "beta=0.5": dict(beta=0.5),
    "gamma=0.5": dict(gamma=0.5),
    "sigma=0.1": dict(sigma=0.1),
}

fig, ax = plt.subplots(figsize=(7, 4))
rows = []
for name, kwargs in variants.items():
    t, e, i = solve_ei(**kwargs)
    ax.plot(t, i, label=name)
    rows.append((name, i.max(), t[i.argmax()]))
ax.set_xlabel("time"); ax.set_ylabel("infected proportion")
ax.set_title("Effect of each parameter on the infection curve")
ax.legend(); plt.show()

for name, peak, when in rows:
    print(f"{name:28s} peak i={peak:.4f} at t={when:6.2f}")

base_peak = rows[0][1]
lower_beta = rows[1][1] < base_peak      # less transmission -> flatter
higher_gamma = rows[2][1] < base_peak    # faster recovery  -> flatter
slower_sigma = rows[3][2] > rows[0][2]   # slower incubation -> later peak

record(
    "T2",
    "reducing beta, raising gamma, and lowering sigma each flatten or delay the curve",
    "beta=0.5 flatter; gamma=0.5 flatter; sigma=0.1 later peak",
    f"beta=0.5 flatter={lower_beta}; gamma=0.5 flatter={higher_gamma}; "
    f"sigma=0.1 later={slower_sigma}",
    lower_beta and higher_gamma and slower_sigma,
)
"""
    ),
    md(
        """
## 4. T3 — all equilibria are disease-free

The source asks the reader to conclude (as an exercise) that the only solutions
of `dY/dt = 0` are `s ≡ const, e = i = 0, r ≡ const`. Two checks: the residual
vanishes at such a state, and a long solve settles into one.
"""
    ),
    code(
        """
candidate = np.array([0.4, 0.0, 0.0, 0.6])           # s=0.4, e=i=0, r=0.6
residual = seir_rhs(0.0, candidate, BETA, SIGMA, GAMMA)

long_sol = solve_ivp(seir_rhs, [0, 400], Y0, rtol=1e-9, args=(BETA, SIGMA, GAMMA))
e_end, i_end = long_sol.y[1, -1], long_sol.y[2, -1]

# Tolerance note: at t=400 the solver leaves |e|, |i| ~ 1e-7 and i can be very
# slightly negative. That is integration noise around zero at rtol=1e-9, not a
# surviving infection, so the check is on absolute value at solver precision.
settled = max(abs(e_end), abs(i_end))

record(
    "T3",
    "every equilibrium of the basic SEIR model is disease-free",
    "RHS residual = 0 at (s, 0, 0, r); long solve drives e, i -> 0",
    f"max|residual|={np.abs(residual).max():.2e}; "
    f"|e(400)|={abs(e_end):.2e}, |i(400)|={abs(i_end):.2e}, s(400)={long_sol.y[0, -1]:.6f}",
    np.abs(residual).max() < 1e-15 and settled < 1e-5,
)
"""
    ),
    md(
        """
## 5. T4 — endemic equilibrium under traveller influx

Adding influx terms `a` into S and `b` into E, the source reports that "the
percentage of the population with the disease now remains at around 5% and never
quite vanishes". Target: a non-vanishing plateau near 0.05.
"""
    ),
    code(
        """
A_IN, B_IN = 0.005, 0.001
endemic = solve_ivp(
    seir_rhs_influx, [0, 150], Y0, rtol=1e-7,
    args=(BETA, SIGMA, GAMMA, A_IN, B_IN),
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(endemic.t, endemic.y[1], "--", color="brown", label="Exposed")
ax.plot(endemic.t, endemic.y[2], color="red", label="Infected")
ax.set_xlabel("time"); ax.set_ylabel("proportion")
ax.set_title(f"SEIR with influx a={A_IN}, b={B_IN}")
ax.legend(); plt.show()

tail = endemic.t >= 120
i_plateau = float(endemic.y[2][tail].mean())
i_drift = float(endemic.y[2][tail].max() - endemic.y[2][tail].min())

record(
    "T4",
    "influx turns the disease-free equilibrium into an endemic one near 5% infected",
    "i settles at approximately 0.05, not decaying to 0",
    f"mean i over t>=120 is {i_plateau:.4f} (drift {i_drift:.4f})",
    0.03 <= i_plateau <= 0.07 and i_drift < 0.02,
)
"""
    ),
    md(
        """
## 6. T5 — R0 as the outbreak threshold

`R0 = β·s₀/γ`. The source runs two cases and reads off the behaviour of `e + i`:

* `beta=0.6, gamma=1, s0=0.9, i0=0.1` → R0 = 0.54 < 1 → no outbreak
* `beta=1, gamma=0.5, s0=0.9, i0=0.1` → R0 = 1.80 > 1 → outbreak

The source notes a small initial dip in `i` in the second case and suggests
plotting `e + i` instead, which is what is checked here.
"""
    ),
    code(
        """
cases = [
    ("no outbreak", dict(beta=0.6, gamma=1.0, s0=0.9, i0=0.1, e0=0.0)),
    ("outbreak", dict(beta=1.0, gamma=0.5, s0=0.9, i0=0.1, e0=0.0)),
]

fig, ax = plt.subplots(figsize=(7, 4))
observed = []
for name, kwargs in cases:
    t, e, i = solve_ei(t1=60, **kwargs)
    r0 = seir_r0(kwargs["beta"], kwargs["gamma"], kwargs["s0"])
    ei = e + i
    grew = ei.max() > ei[0] * 1.05
    ax.plot(t, ei, label=f"{name}: R0={r0:.2f}")
    observed.append((name, r0, ei[0], ei.max(), grew))
ax.set_xlabel("time"); ax.set_ylabel("e + i")
ax.set_title("Outbreak threshold: e + i grows only when R0 > 1")
ax.legend(); plt.show()

for name, r0, start, peak, grew in observed:
    print(f"{name:12s} R0={r0:.3f}  (e+i)(0)={start:.3f}  max(e+i)={peak:.3f}  grew={grew}")

quiet, loud = observed[0], observed[1]
record(
    "T5",
    "R0 = beta*s0/gamma separates outbreak from no outbreak",
    "R0=0.54 -> e+i never grows; R0=1.80 -> e+i grows",
    f"R0={quiet[1]:.2f} grew={quiet[4]}; R0={loud[1]:.2f} grew={loud[4]}",
    (not quiet[4]) and loud[4],
)
"""
    ),
    md(
        """
## 7. T6 — the vaccination threshold

Vaccinating a fraction `v` moves that share straight from S to R, so
`R0 → β·s₀·(1 − v)/γ`. The threshold is the smallest `v` with `R0 < 1`.
"""
    ),
    code(
        """
beta_v, gamma_v, s0_v = 1.0, 0.5, 0.9
r0_unvaccinated = seir_r0(beta_v, gamma_v, s0_v)
v_star = 1.0 - gamma_v / (beta_v * s0_v)          # solves R0(v) = 1

grid = np.linspace(0.0, 0.9, 10)
for v in grid:
    print(f"v={v:.2f}  R0={seir_r0(beta_v, gamma_v, s0_v, vaccinated=v):.4f}")

just_below = seir_r0(beta_v, gamma_v, s0_v, vaccinated=v_star + 1e-6)
just_above = seir_r0(beta_v, gamma_v, s0_v, vaccinated=v_star - 1e-6)

record(
    "T6",
    "vaccinating fraction v scales R0 by (1 - v); threshold at R0 = 1",
    f"R0(0)={r0_unvaccinated:.2f}; threshold v* = 1 - gamma/(beta*s0) = {v_star:.4f}",
    f"R0(v*+eps)={just_below:.6f} < 1 < R0(v*-eps)={just_above:.6f}",
    just_below < 1.0 < just_above,
)
"""
    ),
    md("## 8. Verdict"),
    code(
        """
import pandas as pd

verdict = pd.DataFrame(
    CHECKS, columns=["check", "claim", "expected", "observed", "passed"]
)
n_pass = int(verdict.passed.sum())
print(f"{n_pass}/{len(verdict)} checks passed\\n")

out = Path.cwd().parent / "results" / "R3_seir_basic_checks.csv"
out.parent.mkdir(parents=True, exist_ok=True)
verdict.to_csv(out, index=False)
print(f"wrote {out}")

verdict[["check", "claim", "passed"]]
"""
    ),
    md(
        """
## 9. What this establishes

If every row above passes, the source reproduces exactly — as it should, since
it ships its own code. The value is threefold:

1. **Calibration.** The workspace's ODE tooling, plotting and check harness are
   sound before being pointed at R4, where the target *is* contested.
2. **`seir_r0` is verified** against the threshold behaviour it predicts, not
   just against an algebraic expression.
3. **A working, tested SEIR integrator** for the physics-informed loss of
   Contribution (a). Note what does *not* carry over: this is a single-population
   host-only SEIR with no vector compartment. Dengue needs the host–vector
   SEIR–SEI of R4, and the exposed-human and infected-mosquito compartments are
   unobserved in our data — which is the risk `docs/ROADMAP.md` flags for
   Phase 3.
"""
    ),
]
