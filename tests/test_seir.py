"""Tests for the SEIR-SEI model and the growth ceiling derived from it.

The important tests here are not unit tests in the usual sense. They check our
transcription against a published result: if ``F``, ``V`` or the ``R0`` formula
were wrong anywhere, the sensitivity indices would not reproduce Phaijoo &
Gurung's Table 1, because each index depends on the whole expression.

The last test is the one that matters operationally. ``MAX_WEEKLY_LOG_GROWTH`` is
hard-coded for import speed, so nothing stops it drifting away from the
derivation it claims to come from -- except this.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from dengue_gnn.seir import (
    CEILING_R0_MAX,
    MAX_WEEKLY_LOG_GROWTH,
    PAPER_SECTION4,
    Params,
    growth_rate,
    next_generation_matrices,
    r0,
    sensitivity_indices,
    weekly_log_growth_ceiling,
)

#: Phaijoo & Gurung (2018) Table 1. ``mu_v`` carries a -0.5 offset under our
#: reparameterisation from pi_v to m; see seir.sensitivity_indices.
PAPER_TABLE1 = {
    "m": 0.5,
    "b": 1.0,
    "beta_v": 0.5,
    "mu_v": -1.31823 - (-0.5),
    "nu_v": 0.318228,
    "beta_h": 0.5,
    "gamma_h": -0.49993,
    "mu_h": -0.000208,
    "nu_h": 0.000138,
}


@pytest.mark.parametrize(("param", "expected"), sorted(PAPER_TABLE1.items()))
def test_sensitivity_indices_reproduce_the_published_table(param, expected):
    got = sensitivity_indices(PAPER_SECTION4)[param]
    assert got == pytest.approx(expected, abs=2e-5)


def test_closed_form_r0_agrees_with_the_next_generation_matrices():
    # Two independent routes: the paper's algebraic expression, and the spectral
    # radius it was derived from. Agreement means the transcription of F and V is
    # consistent with the transcription of the formula.
    f, v = next_generation_matrices(PAPER_SECTION4)
    rho = float(max(abs(np.linalg.eigvals(f @ np.linalg.inv(v)))))
    assert r0(PAPER_SECTION4) == pytest.approx(rho, rel=1e-10)


def test_growth_rate_sign_tracks_the_threshold():
    """The paper's Theorem 3.2: the DFE is stable iff R0 < 1.

    Our growth rate is the spectral abscissa whose sign that theorem is about, so
    this is a check on both quantities at once -- they are computed by different
    routes and must agree on which side of the threshold a parameter set is.
    """
    below = PAPER_SECTION4
    assert r0(below) < 1
    assert growth_rate(below) < 0

    # Raise only the biting rate, the parameter with the largest positive index.
    above = replace(below, b=below.b * 4)
    assert r0(above) > 1
    assert growth_rate(above) > 0


def test_r0_scales_as_the_square_root_of_a_transmission_parameter():
    # R0 is a geometric mean over the two-step host-vector cycle, so doubling a
    # single-direction probability multiplies it by sqrt(2), not 2. This is the
    # property most easily lost by dropping the square root.
    doubled = replace(PAPER_SECTION4, beta_h=PAPER_SECTION4.beta_h * 2)
    assert r0(doubled) == pytest.approx(r0(PAPER_SECTION4) * np.sqrt(2), rel=1e-12)

    # The biting rate acts in both directions, so it scales R0 linearly.
    bitten = replace(PAPER_SECTION4, b=PAPER_SECTION4.b * 2)
    assert r0(bitten) == pytest.approx(r0(PAPER_SECTION4) * 2, rel=1e-12)


def test_ceiling_rises_with_the_r0_cap():
    caps = [2.0, 4.0, 6.0, 8.0]
    ceilings = [weekly_log_growth_ceiling(c)[0] for c in caps]
    assert ceilings == sorted(ceilings)


def test_ceiling_beats_the_naive_generation_interval_formula():
    """``ln(R0)/GI`` understates the bound, which is why the old constant was wrong.

    Exponential stage durations produce faster early growth for a given R0 than
    the fixed delay that formula assumes. If this ever stops holding, the
    justification for changing the constant no longer holds either.
    """
    for cap in (2.0, 4.0, 6.0, 8.0):
        ceiling, _ = weekly_log_growth_ceiling(cap)
        assert ceiling > 3.0 * float(np.log(cap) / 3.0)


def test_argmax_actually_satisfies_the_cap():
    # A ceiling is only defensible if the parameters attaining it are admissible.
    ceiling, argmax = weekly_log_growth_ceiling(6.0)
    assert r0(argmax) <= 6.0
    assert 7.0 * growth_rate(argmax) == pytest.approx(ceiling, rel=1e-12)


def test_impossible_cap_raises_rather_than_returning_a_silent_default():
    with pytest.raises(ValueError, match="no grid point"):
        weekly_log_growth_ceiling(0.0)


def test_hard_coded_ceiling_still_matches_its_derivation():
    """The constant in mechanistic.py is hard-coded for import speed.

    Nothing else prevents it drifting from the derivation it cites -- a change to
    DEFAULT_RANGES or to the model would silently leave the loss using a number
    with no basis. This is the only thing tying them together.
    """
    derived, _ = weekly_log_growth_ceiling(CEILING_R0_MAX)
    assert pytest.approx(derived, abs=5e-5) == MAX_WEEKLY_LOG_GROWTH


def test_ceiling_admits_the_growth_actually_present_in_the_record():
    """A physical bound must not rule out most of what was recorded.

    The previous constant, 0.70, was exceeded by 20.9% of observed district-week
    growth rates. This asserts the replacement is not making that mistake again;
    it is a property of the constant, checked against real data.
    """
    from dengue_gnn.data import load_cases

    repo = Path(__file__).resolve().parent.parent
    cases = load_cases(repo / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy")
    g = np.abs(np.diff(np.log1p(np.clip(cases, 0, None)), axis=0))

    assert (g > 0.70).mean() > 0.15, "the old constant should still look bad"
    assert (g > MAX_WEEKLY_LOG_GROWTH).mean() < 0.03


def test_params_lump_parameters_as_the_paper_does():
    p = Params(
        b=0.5, beta_h=0.4, beta_v=0.3, m=2.0, mu_v=0.1, nu_v=0.1, nu_h=0.2, gamma_h=0.25, mu_h=0.01
    )
    assert p.alpha == pytest.approx(0.5 * 0.4 * 2.0)
    assert p.delta == pytest.approx(0.5 * 0.3)
    assert p.beta == pytest.approx(0.2 + 0.01)
    assert p.gamma == pytest.approx(0.25 + 0.01)
    assert p.eps == pytest.approx(0.1)
