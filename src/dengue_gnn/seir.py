"""SEIR-SEI transmission model, used to derive constraints rather than to fit.

Implements the compartmental model of Phaijoo & Gurung (2018), *Sensitivity
Analysis of SEIR-SEI Model of Dengue Disease*, GAMS J. Math. & Math. Biosci.
6(a), 41-49. Hosts move S -> E -> I -> R and vectors S -> E -> I, with no vector
recovery because a mosquito's infection ends at its death.

Why this module exists
----------------------
We do not fit this system. One of its seven compartments is observed, and only
as under-ascertained reported cases, so a hard compartmental residual is
structurally under-determined (EXP-014 and EXP-024; the modules that tried it
are at tag ``phase23-archive``). What the model *can* supply is a defensible
bound on how fast an epidemic can grow, which is a statement about observables.

That bound was previously a guess. ``MAX_WEEKLY_LOG_GROWTH`` was set to 0.70
from ``ln(R)/GI`` with R=8 and a 3-week generation interval. That formula is
wrong for this system, and not slightly: it assumes every onward infection lands
exactly one generation interval later, whereas the model's stage durations are
exponential. For a given R0 a high-variance generation interval produces much
faster early growth than a fixed delay does -- the standard Wallinga & Lipsitch
point. Against the linearised model the naive value understates achievable
growth by roughly 3-4x, and 20.9% of the observed district-week growth rates in
our own data exceed it. A "physical impossibility" that a fifth of the data
violates is not a constraint; it is a bias toward flat forecasts.

The parameterisation
--------------------
Following the paper's proportional form (its eq. 2.2), with lumped parameters

    alpha = b * beta_h * m      m = N_v / N_h = pi_v / (mu_v * N_h)
    delta = b * beta_v
    beta  = nu_h + mu_h         gamma = gamma_h + mu_h        eps = mu_v

``m`` replaces the paper's ``pi_v`` because the vector recruitment rate only ever
enters through the vector-to-host ratio, and a ratio transfers between settings
where a raw recruitment count does not.

Rates are per day throughout; :func:`weekly_log_growth_ceiling` converts.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "CEILING_R0_MAX",
    "DEFAULT_RANGES",
    "INFECTED_STATES",
    "MAX_WEEKLY_LOG_GROWTH",
    "PAPER_SECTION4",
    "Params",
    "growth_rate",
    "next_generation_matrices",
    "r0",
    "sensitivity_indices",
    "weekly_log_growth_ceiling",
]

#: State ordering for the infected subsystem, fixed by the paper's F and V.
INFECTED_STATES = ("e_h", "e_v", "i_h", "i_v")


@dataclass(frozen=True)
class Params:
    """Per-day rates of the SEIR-SEI model.

    Attributes:
        b: Vector biting rate. The paper's most positively sensitive parameter
            (index +1, because R0 depends on it through both transmission
            directions).
        beta_h: Transmission probability, vector to host.
        beta_v: Transmission probability, host to vector.
        m: Vectors per host.
        mu_v: Vector death rate; ``1/mu_v`` is mean mosquito lifespan. The most
            negatively sensitive parameter (index -1.32).
        nu_v: Rate out of the vector exposed class; ``1/nu_v`` is the extrinsic
            incubation period.
        nu_h: Rate out of the host exposed class; ``1/nu_h`` is the intrinsic
            incubation period.
        gamma_h: Host recovery rate; ``1/gamma_h`` is the infectious period.
        mu_h: Host birth and death rate. Nearly irrelevant to R0 (index
            -0.0002) but retained because the paper's formulae carry it.
    """

    b: float
    beta_h: float
    beta_v: float
    m: float
    mu_v: float
    nu_v: float
    nu_h: float
    gamma_h: float
    mu_h: float = 0.000046

    @property
    def alpha(self) -> float:
        return self.b * self.beta_h * self.m

    @property
    def delta(self) -> float:
        return self.b * self.beta_v

    @property
    def beta(self) -> float:
        return self.nu_h + self.mu_h

    @property
    def gamma(self) -> float:
        return self.gamma_h + self.mu_h

    @property
    def eps(self) -> float:
        return self.mu_v


#: The paper's own section-4 simulation values, with pi_v = 2.5e6 and
#: N_h = 5,071,126 folded into m. Retained as a regression fixture: these are the
#: values at which the paper's Table 1 sensitivity indices actually reproduce.
#:
#: They are NOT the values Table 1 lists in its "Baseline Values" column -- that
#: column disagrees with its own indices for nu_h, mu_h, beta_v, mu_v and pi_v.
#: ``scripts/verify_seir_paper.py`` demonstrates both facts. Cite these.
PAPER_SECTION4 = Params(
    b=0.5,
    beta_h=0.75,
    beta_v=0.375,
    m=2500000.0 / (0.25 * 5071126.0),
    mu_v=0.25,
    nu_v=0.1428,
    nu_h=0.1667,
    gamma_h=0.328833,
    mu_h=0.000046,
)

#: OWNER: confirm before any of this reaches the paper (D13).
#: Plausible dengue ranges, per day. These endpoints are the epidemiological
#: judgement in this module and the only part not mechanically derived from
#: Phaijoo & Gurung -- everything downstream of them is.
#:   mu_v    mosquito lifespan 7-30 days
#:   nu_v    extrinsic incubation 7-14 days
#:   nu_h    intrinsic incubation 4-12 days
#:   gamma_h infectious period 2-7 days
#:   m       vectors per host is the least constrained; 0.5-10 is wide on purpose
DEFAULT_RANGES: dict[str, tuple[float, ...]] = {
    "b": (0.5, 0.75, 1.0),
    "beta_h": (0.1, 0.25, 0.4, 0.75),
    "beta_v": (0.1, 0.25, 0.4, 0.75),
    "m": (0.5, 1.0, 2.0, 5.0, 10.0),
    "mu_v": (1 / 30, 1 / 21, 1 / 14, 1 / 10, 1 / 7),
    "nu_v": (1 / 14, 1 / 12, 1 / 10, 1 / 7),
    "nu_h": (1 / 12, 1 / 8, 1 / 6, 1 / 4),
    "gamma_h": (1 / 7, 1 / 5, 1 / 4, 1 / 3, 1 / 2),
}


def next_generation_matrices(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(F, V)`` for the infected subsystem, in :data:`INFECTED_STATES` order.

    Transcribed from the paper's section 3. ``F`` holds new infections and ``V``
    the transitions out of and between infected classes, which is what makes
    ``R0 = rho(F V^-1)`` the next-generation number rather than an arbitrary
    matrix eigenvalue.
    """
    F = np.array(
        [
            [0.0, 0.0, 0.0, p.alpha],
            [0.0, 0.0, p.delta, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    V = np.array(
        [
            [p.beta, 0.0, 0.0, 0.0],
            [0.0, p.eps + p.nu_v, 0.0, 0.0],
            [-p.nu_h, 0.0, p.gamma, 0.0],
            [0.0, -p.nu_v, 0.0, p.eps],
        ]
    )
    return F, V


def r0(p: Params) -> float:
    """Basic reproduction number, the paper's closed form.

    ``sqrt(alpha delta nu_h nu_v / (beta gamma eps (eps + nu_v)))``.

    The square root is not cosmetic: transmission takes two steps, host to
    vector and vector to host, so the per-generation factor is the geometric
    mean of the two. :func:`next_generation_matrices` gives the same number as
    ``rho(F V^-1)``, and the tests check the two agree.
    """
    return float(
        np.sqrt(p.alpha * p.delta * p.nu_h * p.nu_v / (p.beta * p.gamma * p.eps * (p.eps + p.nu_v)))
    )


def growth_rate(p: Params) -> float:
    """Initial exponential growth rate per day: the spectral abscissa of ``F - V``.

    Near the disease-free equilibrium the infected subsystem is ``du/dt = (F-V)u``,
    so infections grow as ``exp(lambda t)`` with ``lambda`` the largest real part
    of the eigenvalues. The paper uses the sign of this quantity for its stability
    theorems (``lambda < 0`` iff ``R0 < 1``); here we want its magnitude.

    This is the quantity ``ln(R0)/GI`` approximates and, for exponential stage
    durations, badly understates.
    """
    F, V = next_generation_matrices(p)
    return float(np.max(np.linalg.eigvals(F - V).real))


def sensitivity_indices(p: Params, h: float = 1e-6) -> dict[str, float]:
    """Normalised forward sensitivity indices of R0, ``(dR0/dq)(q/R0)``.

    The paper's section 3 measure. Central differences; the indices for ``b``,
    ``beta_h``, ``beta_v`` and ``m`` are exactly +1, +0.5, +0.5, +0.5
    analytically, which makes this partly self-checking.

    Eight of the nine reproduce the paper's Table 1 exactly. ``mu_v`` does not,
    and should not: it comes out at -0.818 against the paper's -1.318, a
    difference of exactly 0.5. That is the reparameterisation, not an error.
    The paper varies ``mu_v`` holding the recruitment rate ``pi_v`` fixed, so
    ``mu_v`` also enters ``alpha = b beta_h pi_v / (N_h mu_v)`` and picks up a
    second -0.5. We vary it holding the standing ratio ``m = N_v/N_h`` fixed, so
    ``alpha = b beta_h m`` is independent of it.

    Both are correct answers to different questions. Killing mosquitoes faster
    without changing how many are produced (the paper) shrinks the standing
    population as well as shortening infectious lifespans, so it looks twice as
    effective as shortening lifespans alone (here). Ours is the one that
    isolates lifespan, which is what the growth-rate bound needs.

    Returns:
        One index per parameter of :class:`Params`.
    """
    base = r0(p)
    out: dict[str, float] = {}
    for field in Params.__dataclass_fields__:
        v = getattr(p, field)
        step = v * h
        up = r0(replace(p, **{field: v + step}))
        dn = r0(replace(p, **{field: v - step}))
        out[field] = (up - dn) / (2 * step) * v / base
    return out


def weekly_log_growth_ceiling(
    r0_max: float,
    ranges: dict[str, tuple[float, ...]] | None = None,
    mu_h: float = 0.000046,
) -> tuple[float, Params]:
    """Fastest weekly log growth the model can produce at ``R0 <= r0_max``.

    Maximises the daily growth rate over the parameter grid subject to an R0
    cap, and returns ``7 * lambda``.

    The cap is what makes the answer meaningful. Maximising over the raw
    parameter box alone puts the optimum at R0 ~ 31, which no dengue setting
    reaches, and the resulting ceiling would be so loose it constrains nothing.
    Capping R0 asks the epidemiologically real question -- given an outbreak no
    worse than this, how fast can cases climb -- which is what the paper's R0
    formula is for.

    Args:
        r0_max: Upper bound on R0. This is a judgement about dengue, not a
            derivation; dengue R0 is usually reported in 1-6, occasionally
            higher. OWNER: confirm.
        ranges: Parameter grid; defaults to :data:`DEFAULT_RANGES`.
        mu_h: Host birth/death rate, held fixed -- its R0 index is -0.0002, so
            gridding it would multiply the search for no effect.

    Returns:
        ``(ceiling, argmax_params)``. Returning the argmax matters: a ceiling
        with no attached parameter set cannot be argued with, and this one
        should be.

    Raises:
        ValueError: If no grid point satisfies the cap.
    """
    grid = ranges or DEFAULT_RANGES
    keys = list(grid)
    best: tuple[float, Params] | None = None
    for combo in itertools.product(*(grid[k] for k in keys)):
        p = Params(mu_h=mu_h, **dict(zip(keys, combo)))
        if r0(p) > r0_max:
            continue
        lam = growth_rate(p)
        if best is None or lam > best[0]:
            best = (lam, p)
    if best is None:
        raise ValueError(f"no grid point satisfies R0 <= {r0_max}")
    return 7.0 * best[0], best[1]


# ---------------------------------------------------------------------------
# Derived ceiling
#
# These lived in `dengue_gnn.mechanistic` alongside the loss terms that used
# them. EXP-014 retired those terms -- with the corrected ceiling the hinge never
# activates, so it contributed exactly zero gradient on all 96 rows -- and the
# module was removed in the Phase-2/3 cleanup. The constants stay because they
# are derivations *of this model*, they are asserted against `seir` in
# `tests/test_seir.py`, and the 1.07%-of-observations figure is cited in
# EXP-014. See tag `phase23-archive` for the loss terms themselves.
# ---------------------------------------------------------------------------

#: Ceiling on ``|d log(1+cases)/dweek|``, derived rather than guessed.
#:
#: This was 0.70, from ``ln(R)/GI`` with R=8 and a 3-week generation interval.
#: That was wrong, for a reason worth stating: ``ln(R)/GI`` assumes every onward
#: infection lands exactly one generation interval later, but the SEIR-SEI stage
#: durations are exponential, and for a given R0 a high-variance generation
#: interval produces markedly faster early growth than a fixed delay does
#: (Wallinga & Lipsitch 2007). It understated the bound by 3-4x.
#:
#: The consequence was not academic. **20.9% of the observed district-week growth
#: rates in our own data exceeded 0.70** -- the term was declaring a fifth of the
#: record physically impossible and penalising forecasts for tracking real
#: outbreaks. That is the most likely reason the band constraint never helped in
#: EXP-008 and EXP-014: not the weight, the constant.
#:
#: The value below is the largest weekly log growth the linearised SEIR-SEI model
#: can produce subject to ``R0 <= 6``, i.e. ``7 * max Re eig(F - V)``, from
#: Phaijoo & Gurung (2018). Only 1.07% of observations exceed it, which is the
#: right order for a genuine physical ceiling. Recomputed and asserted in
#: ``tests/test_seir.py``; derivation and data comparison in
#: ``scripts/verify_seir_paper.py`` and ``results/table_seir_validation.md``.
#:
#: Hard-coded rather than computed at import because the search takes ~3 s and
#: every pool worker would pay it.
#:
#: OWNER: confirm the ``R0 <= 6`` cap and the parameter ranges in
#: ``seir.DEFAULT_RANGES``. Those are the epidemiological judgements (D13);
#: everything downstream of them is mechanical.
MAX_WEEKLY_LOG_GROWTH = 2.3884

#: The R0 cap the ceiling above was derived at. Kept beside it so the two cannot
#: drift apart, and so the test can recompute one from the other.
CEILING_R0_MAX = 6.0
