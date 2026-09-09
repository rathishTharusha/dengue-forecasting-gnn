"""Tests for dengue_gnn.augment.

The invariants that matter here are structural: real data must survive
augmentation untouched, synthetic rows must be appended rather than substituted,
and the GAN must actually condition on its input rather than ignoring it.
"""

import pytest

torch = pytest.importorskip("torch")

from dengue_gnn.augment import (  # noqa: E402
    AUGMENT_KINDS,
    TimeSeriesWGAN,
    augment_training_set,
    fit_gan,
    jitter,
    window_warp,
)

N_WIN, N_NODES, WINDOW, N_FEAT, HORIZON = 12, 5, 3, 4, 3
COND = N_FEAT * WINDOW


@pytest.fixture
def fold():
    torch.manual_seed(0)
    x = torch.randn(N_WIN, N_NODES, COND)
    y = torch.randn(N_WIN, N_NODES, HORIZON)
    p = torch.randn(N_WIN, N_NODES, HORIZON)
    return x, y, p


# --------------------------------------------------------------------------
# cheap augmentations
# --------------------------------------------------------------------------


def test_jitter_perturbs_but_stays_close():
    x = torch.zeros(5, 8)
    out = jitter(x, sigma=0.05)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)
    assert out.abs().max() < 1.0  # 20 sigma; effectively never fires by chance


def test_window_warp_preserves_shape():
    x = torch.randn(N_NODES, COND)
    out = window_warp(x, WINDOW, N_FEAT)
    assert out.shape == x.shape


def test_window_warp_is_identity_for_a_constant_window():
    """Interpolating a flat series cannot change it, whatever the warp rate."""
    x = torch.full((N_NODES, COND), 2.0)
    assert torch.allclose(window_warp(x, WINDOW, N_FEAT), x, atol=1e-5)


def test_window_warp_handles_degenerate_window():
    x = torch.randn(N_NODES, N_FEAT * 1)
    assert torch.allclose(window_warp(x, 1, N_FEAT), x)


# --------------------------------------------------------------------------
# the augmentation contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["jitter", "window_warp"])
def test_real_rows_are_preserved_exactly(fold, kind):
    """Synthetic data is appended; the real data must come back untouched."""
    x, y, p = fold
    xa, ya, pa = augment_training_set(x, y, p, kind, 0.5, WINDOW, N_FEAT)
    assert torch.equal(xa[:N_WIN], x)
    assert torch.equal(ya[:N_WIN], y)
    assert torch.equal(pa[:N_WIN], p)


@pytest.mark.parametrize("kind", ["jitter", "window_warp"])
def test_ratio_controls_how_many_are_added(fold, kind):
    x, y, p = fold
    xa, _, _ = augment_training_set(x, y, p, kind, 0.5, WINDOW, N_FEAT)
    assert xa.shape[0] == N_WIN + round(0.5 * N_WIN)


def test_none_and_zero_ratio_are_no_ops(fold):
    x, y, p = fold
    for kind, ratio in [("none", 1.0), ("jitter", 0.0)]:
        xa, ya, pa = augment_training_set(x, y, p, kind, ratio, WINDOW, N_FEAT)
        assert torch.equal(xa, x) and torch.equal(ya, y) and torch.equal(pa, p)


def test_unknown_kind_raises(fold):
    x, y, p = fold
    with pytest.raises(ValueError, match="unknown augmentation"):
        augment_training_set(x, y, p, "diffusion", 0.5, WINDOW, N_FEAT)


def test_augmentation_is_seeded(fold):
    x, y, p = fold
    a = augment_training_set(x, y, p, "jitter", 0.5, WINDOW, N_FEAT, seed=7)[0]
    b = augment_training_set(x, y, p, "jitter", 0.5, WINDOW, N_FEAT, seed=7)[0]
    assert torch.equal(a, b)


def test_all_kinds_are_reachable():
    assert set(AUGMENT_KINDS) == {"none", "jitter", "window_warp", "gan"}


# --------------------------------------------------------------------------
# the GAN
# --------------------------------------------------------------------------


def test_gan_shapes():
    torch.manual_seed(0)
    gan = TimeSeriesWGAN(COND, HORIZON)
    cond = torch.randn(7, COND)
    assert gan.generate(cond).shape == (7, HORIZON)
    assert gan.score(cond, gan.generate(cond)).shape == (7, 1)


def test_gan_output_depends_on_the_condition():
    """RCGAN-style conditioning: different conditions must give different output.

    A generator that ignores its condition has collapsed to an unconditional
    model, which on 459 sequences is the failure mode the risk register names.
    """
    torch.manual_seed(0)
    gan = TimeSeriesWGAN(COND, HORIZON)
    torch.manual_seed(1)
    a = gan.generate(torch.zeros(4, COND))
    torch.manual_seed(1)  # same noise draw, different condition
    b = gan.generate(torch.ones(4, COND))
    assert not torch.allclose(a, b)


def test_gan_trains_without_diverging():
    torch.manual_seed(0)
    cond = torch.randn(64, COND)
    target = torch.randn(64, HORIZON)
    gan = fit_gan(cond, target, epochs=5, batch=32, n_critic=2)
    out = gan.generate(cond)
    assert torch.isfinite(out).all()


def test_gan_augmentation_appends_finite_rows(fold):
    x, y, p = fold
    xa, ya, pa = augment_training_set(x, y, p, "gan", 0.5, WINDOW, N_FEAT, seed=0, gan_epochs=5)
    assert xa.shape[0] == ya.shape[0] == pa.shape[0] == N_WIN + round(0.5 * N_WIN)
    assert torch.equal(xa[:N_WIN], x)
    assert torch.isfinite(ya).all()
