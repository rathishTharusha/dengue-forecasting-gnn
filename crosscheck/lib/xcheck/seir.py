"""Compartmental dengue models: basic SEIR and the SEIR-SEI host-vector model.

Two papers are reproduced here.

**Gopalakrishnan, "The SEIR model of infectious diseases" (MTH 271, 2020)** --
:func:`seir_rhs`, :func:`seir_rhs_influx` and :func:`seir_r0`. The source is a
lecture notebook whose code is printed in full, so the reproduction target is
exact numerical agreement with the code as written.

**Phaijoo & Gurung, "Sensitivity Analysis of SEIR-SEI Model of Dengue Disease"
(GAMS J. Math. Math. Biosci. 6(a), 2018)** -- :class:`SeirSeiParams`,
:func:`seir_sei_rhs`, :func:`seir_sei_r0`, :func:`seir_sei_equilibrium` and
:func:`sensitivity_indices`. The reproduction targets are the paper's R0
expression (next-generation matrix), its endemic equilibrium, and the nine
normalized forward sensitivity indices of Table 1.

Both sensitivity paths are provided -- closed form and finite difference -- so
they check each other rather than resting on one derivation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "PHAIJOO_SIMULATION",
    "PHAIJOO_TABLE1_BASELINES",
    "PHAIJOO_TABLE1_INDICES",
    "SeirSeiParams",
    "seir_r0",
    "seir_rhs",
    "seir_rhs_influx",
    "seir_sei_equilibrium",
    "seir_sei_next_generation",
    "seir_sei_r0",
    "seir_sei_rhs",
    "sensitivity_indices",
    "sensitivity_indices_numeric",
]


# --------------------------------------------------------------------------
# Basic SEIR (Gopalakrishnan)
# --------------------------------------------------------------------------


def seir_rhs(t: float, y: np.ndarray, beta: float, sigma: float, gamma: float) -> np.ndarray:
    """Right-hand side of the proportion-form SEIR system.

    ``ds/dt = -beta*i*s``, ``de/dt = beta*i*s - sigma*e``,
    ``di/dt = sigma*e - gamma*i``, ``dr/dt = gamma*i``.

    Transcribed from the source's ``seir_f``, including its argument order, so
    that ``scipy.integrate.solve_ivp(seir_rhs, ...)`` reproduces the published
    figures.
    """
    s, e, i, r = y  # noqa: RUF059  -- names mirror the source's seir_f exactly
    return np.array(
        [
            -beta * i * s,
            -sigma * e + beta * i * s,
            -gamma * i + sigma * e,
            gamma * i,
        ]
    )


def seir_rhs_influx(
    t: float,
    y: np.ndarray,
    beta: float,
    sigma: float,
    gamma: float,
    a: float,
    b: float,
) -> np.ndarray:
    """SEIR with traveller influx ``a`` into S and ``b`` into E (source's ``seir_f2``).

    The influx terms are what turn the disease-free equilibrium into an endemic
    one; the source demonstrates this with ``a=0.005, b=0.001``.
    """
    s, e, i, r = y  # noqa: RUF059  -- names mirror the source's seir_f2 exactly
    return np.array(
        [
            -beta * i * s + a,
            -sigma * e + beta * i * s + b,
            -gamma * i + sigma * e,
            gamma * i - (a + b),
        ]
    )


def seir_r0(beta: float, gamma: float, s0: float = 1.0, vaccinated: float = 0.0) -> float:
    """Basic reproduction number ``R0 = beta * s0 * (1 - v) / gamma``.

    The source derives this from ``d(e+i)/dt = (beta*s - gamma)*i``, and notes
    that vaccinating a fraction ``v`` moves that share of the population
    straight from S to R, scaling ``s0`` by ``(1 - v)``.
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not 0.0 <= vaccinated <= 1.0:
        raise ValueError("vaccinated fraction must lie in [0, 1]")
    return beta * s0 * (1.0 - vaccinated) / gamma


# --------------------------------------------------------------------------
# SEIR-SEI (Phaijoo & Gurung)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeirSeiParams:
    """Parameters of the SEIR-SEI dengue model, in the paper's own notation.

    Attributes:
        N_h: Total (constant) host population.
        b: Biting rate of the vector.
        beta_h: Transmission probability, vector to host.
        beta_v: Transmission probability, host to vector.
        mu_h: Host birth rate, assumed equal to the host death rate.
        mu_v: Vector death rate.
        nu_h: Host incubation rate (exposed to infectious).
        nu_v: Vector incubation rate.
        gamma_h: Host recovery rate.
        pi_v: Vector recruitment rate.
    """

    N_h: float
    b: float
    beta_h: float
    beta_v: float
    mu_h: float
    mu_v: float
    nu_h: float
    nu_v: float
    gamma_h: float
    pi_v: float

    # -- the paper's composite symbols, Section 2 --

    @property
    def alpha(self) -> float:
        """``alpha = b * beta_h * pi_v / (N_h * mu_v)``."""
        return self.b * self.beta_h * self.pi_v / (self.N_h * self.mu_v)

    @property
    def beta(self) -> float:
        """``beta = nu_h + mu_h`` (exit rate from the exposed-host class)."""
        return self.nu_h + self.mu_h

    @property
    def gamma(self) -> float:
        """``gamma = gamma_h + mu_h`` (exit rate from the infectious-host class)."""
        return self.gamma_h + self.mu_h

    @property
    def delta(self) -> float:
        """``delta = b * beta_v``."""
        return self.b * self.beta_v

    @property
    def epsilon(self) -> float:
        """``epsilon = mu_v``."""
        return self.mu_v


#: Parameter values listed for the numerical simulations (paper, Section 4).
PHAIJOO_SIMULATION = SeirSeiParams(
    N_h=5_071_126,
    b=0.5,
    beta_h=0.75,
    beta_v=0.375,
    mu_h=0.000046,
    mu_v=0.25,
    nu_h=0.1667,
    nu_v=0.1428,
    gamma_h=0.328833,
    pi_v=2_500_000,
)

#: The "Baseline Values" column of Table 1, as printed.
PHAIJOO_TABLE1_BASELINES = {
    "pi_v": 5000.0,
    "b": 0.5,
    "beta_v": 1.0,
    "mu_v": 0.02941,
    "nu_v": 0.1428,
    "beta_h": 0.75,
    "gamma_h": 0.328833,
    "mu_h": 0.0045,
    "nu_h": 1.667,
}

#: The "Sensitivity Indices" column of Table 1, as printed.
PHAIJOO_TABLE1_INDICES = {
    "pi_v": +0.5,
    "b": +1.0,
    "beta_v": +0.5,
    "mu_v": -1.31823,
    "nu_v": +0.318228,
    "beta_h": +0.5,
    "gamma_h": -0.49993,
    "mu_h": -0.000208,
    "nu_h": +0.000138,
}


def seir_sei_rhs(t: float, y: np.ndarray, p: SeirSeiParams) -> np.ndarray:
    """Right-hand side of the reduced proportion system, paper's Eq. (2.2).

    State is ``[s_h, e_h, i_h, e_v, i_v]``; ``r_h`` and ``s_v`` are recovered by
    ``r_h = 1 - s_h - e_h - i_h`` and ``s_v = 1 - e_v - i_v``.
    """
    s_h, e_h, i_h, e_v, i_v = y
    s_v = 1.0 - e_v - i_v
    return np.array(
        [
            p.mu_h * (1.0 - s_h) - p.alpha * s_h * i_v,
            p.alpha * s_h * i_v - p.beta * e_h,
            p.nu_h * e_h - p.gamma * i_h,
            p.delta * s_v * i_h - (p.epsilon + p.nu_v) * e_v,
            p.nu_v * e_v - p.epsilon * i_v,
        ]
    )


def seir_sei_next_generation(p: SeirSeiParams) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's transmission matrix ``F`` and transition matrix ``V``.

    Ordering follows the paper: ``[e_h, e_v, i_h, i_v]``.
    """
    f = np.array(
        [
            [0.0, 0.0, 0.0, p.alpha],
            [0.0, 0.0, p.delta, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    v = np.array(
        [
            [p.beta, 0.0, 0.0, 0.0],
            [0.0, p.epsilon + p.nu_v, 0.0, 0.0],
            [-p.nu_h, 0.0, p.gamma, 0.0],
            [0.0, -p.nu_v, 0.0, p.epsilon],
        ]
    )
    return f, v


def seir_sei_r0(p: SeirSeiParams, method: str = "closed_form") -> float:
    """Basic reproduction number of the SEIR-SEI model.

    Args:
        p: Model parameters.
        method: ``"closed_form"`` evaluates the paper's expression
            ``sqrt(alpha*delta*nu_h*nu_v / (beta*gamma*epsilon*(epsilon+nu_v)))``;
            ``"spectral"`` computes ``rho(F V^-1)`` from
            :func:`seir_sei_next_generation`. The two must agree -- that is the
            point of offering both.

    Returns:
        R0 as a float.
    """
    if method == "closed_form":
        num = p.alpha * p.delta * p.nu_h * p.nu_v
        den = p.beta * p.gamma * p.epsilon * (p.epsilon + p.nu_v)
        return float(np.sqrt(num / den))
    if method == "spectral":
        f, v = seir_sei_next_generation(p)
        return float(max(abs(np.linalg.eigvals(f @ np.linalg.inv(v)))))
    raise ValueError(f"unknown method {method!r}")


def seir_sei_equilibrium(p: SeirSeiParams, variant: str = "paper") -> dict[str, float]:
    """Endemic equilibrium ``E1`` of Eq. (2.2).

    Args:
        p: Model parameters.
        variant: ``"paper"`` transcribes the expressions as printed.
            ``"corrected"`` replaces the two vector components with expressions
            derived from Eq. (2.2) itself; see the note below.

    Returns:
        A mapping with keys ``s_h``, ``e_h``, ``i_h``, ``e_v``, ``i_v``. The
        components are only epidemiologically meaningful when ``R0 > 1``; below
        that they go negative and the disease-free equilibrium is the stable one.

    Note:
        The printed ``s_h*``, ``e_h*`` and ``i_h*`` are exact -- substituting
        them into :func:`seir_sei_rhs` gives a zero residual. The printed
        ``e_v*`` and ``i_v*`` are not: their ratio is ``epsilon``, whereas
        ``di_v/dt = 0`` forces ``i_v = (nu_v/epsilon) * e_v``, so the pair can
        only satisfy the system when ``nu_v = 1``. Solving the vector block of
        Eq. (2.2) directly, with ``s_v = 1 - e_v - i_v``, gives

        ``e_v = delta*i_h*epsilon / ((epsilon + nu_v) * (epsilon + delta*i_h))``

        which is what ``variant="corrected"`` returns. This does not affect the
        paper's ``R0``, its stability theorems, or its sensitivity analysis --
        none of which depend on the vector components of ``E1``.
    """
    if variant not in {"paper", "corrected"}:
        raise ValueError(f"variant must be 'paper' or 'corrected', got {variant!r}")

    a, be, ga = p.alpha, p.beta, p.gamma
    de, ep, nv, nh, mh = p.delta, p.epsilon, p.nu_v, p.nu_h, p.mu_h
    r0sq = seir_sei_r0(p) ** 2

    common = de * nh * (ep * mh + (a + mh) * nv)
    s_h = (be * ga * ep + de * mh * nh) * (ep + nv) / common
    e_h = mh * ga * ep * (ep + nv) * (r0sq - 1.0) / common
    i_h = mh * ep * (ep + nv) * (r0sq - 1.0) / (de * (ep * mh + (a + mh) * nv))

    if variant == "paper":
        e_v = mh * be * ga * ep**2 * (r0sq - 1.0) / (de * (be * ga * ep + de * mh * nh))
        i_v = mh * be * ga * ep * (r0sq - 1.0) / (de * (be * ga * ep + de * mh * nh))
    else:
        e_v = de * i_h * ep / ((ep + nv) * (ep + de * i_h))
        i_v = nv / ep * e_v

    return {
        "s_h": float(s_h),
        "e_h": float(e_h),
        "i_h": float(i_h),
        "e_v": float(e_v),
        "i_v": float(i_v),
    }


def sensitivity_indices(p: SeirSeiParams) -> dict[str, float]:
    """Normalized forward sensitivity indices of R0, in closed form.

    ``Upsilon = (dR0/dq) * (q/R0)`` for each parameter ``q``. Because
    ``R0 = sqrt(...)``, every index is half the corresponding log-derivative of
    ``R0**2``, which makes them all elementary:

    * ``pi_v``, ``beta_h``, ``beta_v`` enter ``R0**2`` linearly     -> ``+1/2``
    * ``b`` enters twice (through ``alpha`` and ``delta``)          -> ``+1``
    * ``mu_v`` enters as ``1/(mu_v**2 (mu_v + nu_v))``
    * ``nu_v`` enters as ``nu_v/(mu_v + nu_v)``
    * ``gamma_h`` through ``gamma = gamma_h + mu_h``
    * ``nu_h`` through ``beta = nu_h + mu_h`` and the numerator ``nu_h``
    * ``mu_h`` through both ``beta`` and ``gamma``
    """
    mh, mv, nh, nv, gh = p.mu_h, p.mu_v, p.nu_h, p.nu_v, p.gamma_h
    return {
        "pi_v": 0.5,
        "b": 1.0,
        "beta_v": 0.5,
        "beta_h": 0.5,
        "mu_v": float(-0.5 * (2.0 + mv / (mv + nv))),
        "nu_v": float(0.5 * mv / (mv + nv)),
        "gamma_h": float(-0.5 * gh / (gh + mh)),
        "nu_h": float(0.5 * mh / (nh + mh)),
        "mu_h": float(-0.5 * (mh / (nh + mh) + mh / (gh + mh))),
    }


def sensitivity_indices_numeric(p: SeirSeiParams, rel_step: float = 1e-6) -> dict[str, float]:
    """Sensitivity indices by central finite difference, as an independent check.

    Perturbs each parameter by ``rel_step`` relative and differences ``R0``.
    Agreement with :func:`sensitivity_indices` to ~6 decimals confirms the
    closed forms; disagreement means one of them is wrong.
    """
    names = ("pi_v", "b", "beta_v", "beta_h", "mu_v", "nu_v", "gamma_h", "nu_h", "mu_h")
    r0 = seir_sei_r0(p)
    out: dict[str, float] = {}
    for name in names:
        q = getattr(p, name)
        h = q * rel_step
        up = seir_sei_r0(replace(p, **{name: q + h}))
        dn = seir_sei_r0(replace(p, **{name: q - h}))
        out[name] = float((up - dn) / (2 * h) * q / r0)
    return out
