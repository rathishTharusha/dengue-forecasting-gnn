"""Tests for the cross-check library.

Two jobs. First, guard the workspace's own numbers -- a silent change in how
``mape_weng`` floors its denominator or how ``segment_cv`` slices would move
every finding downstream. Second, and specific to this workspace: confirm that
``xcheck.metrics`` and ``dengue_gnn.metrics`` agree where they overlap. They
were written independently from the same definitions, so agreement is evidence
that both are right, and a failure here means one of them is wrong -- which is
exactly what a cross-check exists to detect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from xcheck.data import build_edge_index, make_windows, segment_split
from xcheck.graph import gravity_mobility, hop_distance, hybrid_adjacency, symmetric_normalize
from xcheck.metrics import (
    crps_gaussian,
    mae,
    mape_masked,
    mape_weng,
    morans_i,
    mpiw,
    picp,
    rmse,
    smape,
)
from xcheck.protocol import (
    WENG_SEGMENTS,
    FoldResult,
    batched_mean_metric,
    rolling_origin_folds,
    segment_cv,
    summarize,
)
from xcheck.seir import (
    PHAIJOO_SIMULATION,
    PHAIJOO_TABLE1_INDICES,
    seir_r0,
    seir_rhs,
    seir_sei_equilibrium,
    seir_sei_r0,
    seir_sei_rhs,
    sensitivity_indices,
    sensitivity_indices_numeric,
)

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_perfect_forecast_scores_zero():
    y = np.array([0.0, 13.0, 250.0, 2631.0])
    assert rmse(y, y) == pytest.approx(0.0)
    assert mae(y, y) == pytest.approx(0.0)
    assert smape(y, y) == pytest.approx(0.0)
    assert mape_masked(y, y) == pytest.approx(0.0)


def test_known_metric_values():
    pred = np.array([2.0, 4.0])
    truth = np.array([4.0, 4.0])
    assert rmse(pred, truth) == pytest.approx(math.sqrt(2.0))
    assert mae(pred, truth) == pytest.approx(1.0)
    assert mape_masked(pred, truth) == pytest.approx(25.0)


def test_weng_mape_explodes_on_zero_truth():
    """The reference MAPE's 1e-15 floor is not a stabilizer -- that is the point."""
    assert mape_weng(np.array([5.0]), np.array([0.0])) > 1e16
    # A single zero week dominates an otherwise perfect batch.
    mixed = mape_weng(np.array([5.0, 10.0]), np.array([0.0, 10.0]))
    assert mixed > 1e16


def test_masked_mape_ignores_zero_weeks():
    pred = np.array([5.0, 10.0])
    truth = np.array([0.0, 20.0])
    assert mape_masked(pred, truth) == pytest.approx(50.0)


def test_masked_mape_is_nan_when_nothing_qualifies():
    assert math.isnan(mape_masked(np.array([1.0]), np.array([0.0])))


def test_metrics_reject_shape_mismatch_and_empty():
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="empty"):
        mae(np.array([]), np.array([]))


def test_morans_i_detects_clustering():
    """A path graph: a monotone gradient clusters, an alternating pattern does not."""
    w = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    assert morans_i([1.0, 2.0, 3.0, 4.0], w) > 0
    assert morans_i([1.0, 4.0, 1.0, 4.0], w) < 0


def test_morans_i_is_nan_for_constant_field():
    w = np.array([[0, 1], [1, 0]], dtype=float)
    assert math.isnan(morans_i([2.0, 2.0], w))


def test_probabilistic_metrics():
    mu = np.zeros(4)
    sigma = np.ones(4)
    truth = np.zeros(4)
    # CRPS of a perfectly centred standard normal: 1/sqrt(pi) - 2*phi(0) ... check sign only.
    assert crps_gaussian(mu, sigma, truth) > 0
    lo, hi = mu - 1.96, mu + 1.96
    assert picp(lo, hi, truth) == pytest.approx(1.0)
    assert mpiw(lo, hi) == pytest.approx(3.92)


# --------------------------------------------------------------------------
# cross-implementation agreement with the main project
# --------------------------------------------------------------------------


def test_agrees_with_project_metrics():
    """xcheck and dengue_gnn were written independently; they must still agree.

    This is the cross-check's own cross-check. If it fails, one implementation
    has drifted and every reproduction in this workspace is suspect.
    """
    dengue_gnn = pytest.importorskip("dengue_gnn.metrics", reason="main project not on the path")

    rng = np.random.default_rng(0)
    truth = rng.poisson(13, size=(40, 25, 3)).astype(float)
    pred = np.clip(truth + rng.normal(0, 8, truth.shape), 0, None)

    theirs = dengue_gnn.score(pred, truth)
    assert theirs["RMSE"] == pytest.approx(rmse(pred, truth))
    assert theirs["MAE"] == pytest.approx(mae(pred, truth))
    assert theirs["SMAPE"] == pytest.approx(smape(pred, truth))
    assert theirs["MAPE"] == pytest.approx(mape_masked(pred, truth))


# --------------------------------------------------------------------------
# data / windowing
# --------------------------------------------------------------------------


def _toy_series(t=40, n=5, f=11):
    rng = np.random.default_rng(1)
    return rng.gamma(2.0, 5.0, size=(t, n, f))


def test_windowing_shapes_and_alignment():
    raw = _toy_series()
    w = make_windows(raw, window=3, horizon=3, use_all_features=False, normalize="global")
    assert w.x.shape == (40 - 3 - 3, 5, 1, 3)
    assert w.y.shape == (40 - 3 - 3, 5, 3)
    # y for window k must be the raw cases at [index, index+horizon), z-scored.
    k, i = 4, int(w.index[4])
    np.testing.assert_allclose(w.inverse(w.y[k]), raw[i : i + 3, :, 5].T, rtol=1e-9)


def test_train_normalization_excludes_later_weeks():
    raw = _toy_series()
    glob = make_windows(raw, normalize="global")
    trn = make_windows(raw, normalize="train", train_end=20)
    assert glob.mean != trn.mean
    assert trn.mean == pytest.approx(raw[:20, :, 5].mean())
    assert trn.std == pytest.approx(raw[:20, :, 5].std())


def test_windowing_rejects_bad_arguments():
    raw = _toy_series()
    with pytest.raises(ValueError, match="train_end"):
        make_windows(raw, normalize="train")
    with pytest.raises(ValueError, match="unknown normalize"):
        make_windows(raw, normalize="sideways")
    with pytest.raises(ValueError, match="too short"):
        make_windows(raw[:4], window=3, horizon=3)


def test_edge_index_round_trip():
    a = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=float)
    ei = build_edge_index(a)
    assert ei.shape == (2, int(a.sum()))
    rebuilt = np.zeros_like(a)
    rebuilt[ei[0], ei[1]] = 1.0
    np.testing.assert_array_equal(rebuilt, a)


def test_segment_split_matches_reference_arithmetic():
    train, val, test = segment_split(100, 0.7, 0.1)
    assert (train.start, train.stop) == (0, 70)
    assert (val.start, val.stop) == (70, 80)
    assert (test.start, test.stop) == (80, 100)


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------


def test_segment_cv_full_includes_training_windows():
    """The finding at the heart of R1, pinned as a test."""
    plan_full = segment_cv(100, WENG_SEGMENTS, report_on="full")
    plan_test = segment_cv(100, WENG_SEGMENTS, report_on="test")
    for (frac, train, ev_full), (_, _, ev_test) in zip(plan_full, plan_test):
        n_seg = int(100 * frac)
        assert (ev_full.start, ev_full.stop) == (0, n_seg)  # whole segment
        assert ev_test.start == train.stop  # held-out tail only
        assert train.stop == int(n_seg * 0.7)


def test_segment_cv_rejects_bad_report_on():
    with pytest.raises(ValueError, match="report_on"):
        segment_cv(100, report_on="both")


def test_batched_mean_rmse_is_at_most_pooled_rmse():
    """Jensen: averaging per-window RMSE cannot exceed the pooled RMSE."""
    rng = np.random.default_rng(2)
    truth = rng.gamma(1.5, 20.0, size=(60, 25, 3))
    pred = truth + rng.normal(0, 30, truth.shape)
    assert batched_mean_metric(pred, truth, rmse, 1) <= rmse(pred, truth) + 1e-9
    # MAE is a mean of means, so batching leaves it unchanged.
    assert batched_mean_metric(pred, truth, mae, 1) == pytest.approx(mae(pred, truth))


def test_rolling_origin_folds_are_chronological():
    for _origin, train, val, test in rolling_origin_folds(400):
        assert train.stop == val.start
        assert val.stop == test.start
        assert train.start == 0
        assert test.stop > test.start


def test_summarize_uses_population_std():
    folds = [
        FoldResult(label=i, n_train=1, n_eval=1, metrics={"RMSE": v})
        for i, v in enumerate([1.0, 2.0, 3.0])
    ]
    mean, std = summarize(folds)["RMSE"]
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(np.std([1.0, 2.0, 3.0]))


# --------------------------------------------------------------------------
# SEIR (R3)
# --------------------------------------------------------------------------


def test_seir_rhs_conserves_population():
    y = np.array([0.7, 0.1, 0.15, 0.05])
    assert seir_rhs(0.0, y, 1.0, 1.0, 0.1).sum() == pytest.approx(0.0, abs=1e-15)


def test_seir_disease_free_state_is_an_equilibrium():
    y = np.array([0.4, 0.0, 0.0, 0.6])
    np.testing.assert_allclose(seir_rhs(0.0, y, 1.0, 1.0, 0.1), 0.0, atol=1e-16)


def test_seir_r0_and_vaccination():
    assert seir_r0(1.0, 0.5, s0=0.9) == pytest.approx(1.8)
    assert seir_r0(1.0, 0.5, s0=0.9, vaccinated=0.5) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="gamma"):
        seir_r0(1.0, 0.0)
    with pytest.raises(ValueError, match="vaccinated"):
        seir_r0(1.0, 0.5, vaccinated=1.5)


# --------------------------------------------------------------------------
# SEIR-SEI (R4) -- the published targets, pinned
# --------------------------------------------------------------------------


def test_seir_sei_r0_two_ways_agree():
    p = PHAIJOO_SIMULATION
    assert seir_sei_r0(p, "closed_form") == pytest.approx(seir_sei_r0(p, "spectral"), rel=1e-12)


def test_reproduces_phaijoo_table1_indices():
    """All nine published sensitivity indices, to the paper's printed precision."""
    got = sensitivity_indices(PHAIJOO_SIMULATION)
    for name, published in PHAIJOO_TABLE1_INDICES.items():
        assert got[name] == pytest.approx(published, abs=5e-6), name


def test_closed_form_indices_match_finite_differences():
    closed = sensitivity_indices(PHAIJOO_SIMULATION)
    numeric = sensitivity_indices_numeric(PHAIJOO_SIMULATION)
    for name in closed:
        assert closed[name] == pytest.approx(numeric[name], abs=1e-6), name


def _state(eq: dict[str, float]) -> np.ndarray:
    return np.array([eq["s_h"], eq["e_h"], eq["i_h"], eq["e_v"], eq["i_v"]])


def test_corrected_endemic_equilibrium_is_a_fixed_point():
    from dataclasses import replace

    p = replace(PHAIJOO_SIMULATION, b=1.2)
    assert seir_sei_r0(p) > 1.0
    eq = seir_sei_equilibrium(p, variant="corrected")
    np.testing.assert_allclose(seir_sei_rhs(0.0, _state(eq), p), 0.0, atol=1e-12)
    assert eq["i_h"] > 0
    assert eq["i_v"] > 0


def test_published_endemic_equilibrium_fails_in_its_vector_components():
    """Finding F4.4, pinned so a future edit cannot quietly erase it.

    The printed E1 satisfies Eq. (2.2) in its host components and not in its
    vector ones: as printed ``e_v*/i_v* = epsilon``, while ``di_v/dt = 0``
    requires ``epsilon/nu_v``.
    """
    from dataclasses import replace

    p = replace(PHAIJOO_SIMULATION, b=1.2)
    paper = seir_sei_equilibrium(p, variant="paper")
    fixed = seir_sei_equilibrium(p, variant="corrected")

    # Host components agree between the two variants -- the paper has those right.
    for key in ("s_h", "e_h", "i_h"):
        assert paper[key] == pytest.approx(fixed[key], rel=1e-12), key

    # The vector components do not, and the printed pair is not an equilibrium.
    assert np.abs(seir_sei_rhs(0.0, _state(paper), p)).max() > 1e-8
    assert paper["e_v"] / paper["i_v"] == pytest.approx(p.epsilon, rel=1e-12)
    assert fixed["e_v"] / fixed["i_v"] == pytest.approx(p.epsilon / p.nu_v, rel=1e-12)


def test_seir_sei_equilibrium_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        seir_sei_equilibrium(PHAIJOO_SIMULATION, variant="published")


def test_endemic_components_are_negative_below_threshold():
    """Theorem 3.1: E1 exists only when R0 > 1."""
    p = PHAIJOO_SIMULATION
    assert seir_sei_r0(p) < 1.0  # the paper's own simulation parameters
    assert seir_sei_equilibrium(p, variant="corrected")["i_h"] < 0


# --------------------------------------------------------------------------
# dynamic graph (R2)
# --------------------------------------------------------------------------


def test_gravity_mobility_decays_with_distance():
    d = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    g = gravity_mobility(np.ones(3), d)
    np.testing.assert_allclose(np.diag(g), 0.0)
    assert g[0, 1] > g[0, 2]
    assert g.max() == pytest.approx(1.0)


def test_hybrid_adjacency_endpoints():
    a, b = np.eye(3), np.ones((3, 3))
    np.testing.assert_allclose(hybrid_adjacency(a, b, alpha=1.0), a)
    np.testing.assert_allclose(hybrid_adjacency(a, b, alpha=0.0), b)
    with pytest.raises(ValueError, match="alpha"):
        hybrid_adjacency(a, b, alpha=2.0)


def test_hop_distance_on_a_path_graph():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    d = hop_distance(a)
    assert d[0, 2] == pytest.approx(2.0)
    assert d[0, 1] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diag(d), 0.0)


def test_hop_distance_handles_disconnected_components():
    """Islands must not leave infinities in the gravity kernel."""
    a = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
    d = hop_distance(a)
    assert np.isfinite(d).all()
    assert d[0, 2] == pytest.approx(2.0)


def test_symmetric_normalize_is_symmetric_and_bounded():
    a = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    n = symmetric_normalize(a)
    np.testing.assert_allclose(n, n.T, atol=1e-12)
    assert np.abs(np.linalg.eigvals(n)).max() <= 1.0 + 1e-9


# --------------------------------------------------------------------------
# models (R1, R2) -- skipped wherever torch is absent, e.g. CI
# --------------------------------------------------------------------------


def test_stgat_shapes():
    torch = pytest.importorskip("torch", reason="STGAT needs torch")
    pytest.importorskip("torch_geometric", reason="STGAT needs PyTorch Geometric")
    from xcheck.models import STGAT

    n_nodes, window, horizon, batch = 25, 3, 3, 4
    a = np.eye(n_nodes) + np.eye(n_nodes, k=1) + np.eye(n_nodes, k=-1)
    edge_index = torch.tensor(build_edge_index(a))

    torch.manual_seed(0)
    model = STGAT(in_channels=window, n_pred=horizon, n_nodes=n_nodes)
    out = model(torch.randn(batch, n_nodes, window), edge_index)
    assert out.shape == (batch, n_nodes, horizon)


def test_dynamic_stgnn_shapes_and_ablations():
    torch = pytest.importorskip("torch", reason="DynamicSTGNN needs torch")
    from xcheck.models import DynamicSTGNN

    batch, window, nodes, feats = 6, 8, 25, 1
    x = torch.randn(batch, window, nodes, feats)
    adj = torch.tensor(symmetric_normalize(np.eye(nodes)), dtype=torch.float32)

    point = DynamicSTGNN(n_features=feats, horizons=2)
    assert point(x, adj).shape == (batch, nodes, 2)

    # The attention ablation must change the module, not just the arithmetic.
    no_attn = DynamicSTGNN(n_features=feats, horizons=2, temporal_attention=False)
    assert no_attn.attention is None
    assert no_attn(x, adj).shape == (batch, nodes, 2)

    prob = DynamicSTGNN(n_features=feats, horizons=2, probabilistic=True)
    assert prob(x, adj).shape == (batch, nodes, 2, 2)  # (mu, log_var)


def test_dynamic_stgnn_accepts_time_varying_graph():
    torch = pytest.importorskip("torch", reason="DynamicSTGNN needs torch")
    from xcheck.models import DynamicSTGNN

    batch, window, nodes = 3, 5, 10
    x = torch.randn(batch, window, nodes, 1)
    per_step = torch.randn(batch, window, nodes, nodes).softmax(dim=-1)
    model = DynamicSTGNN(n_features=1, horizons=1)
    assert model(x, per_step).shape == (batch, nodes, 1)

    with pytest.raises(ValueError, match="2-D or 4-D"):
        model(x, torch.randn(batch, nodes, nodes))


def test_gaussian_nll_is_minimized_at_the_truth():
    torch = pytest.importorskip("torch", reason="needs torch")
    from xcheck.models import gaussian_nll

    target = torch.zeros(100)
    log_var = torch.zeros(100)
    centred = gaussian_nll(torch.zeros(100), log_var, target)
    offset = gaussian_nll(torch.full((100,), 2.0), log_var, target)
    assert centred < offset


def test_stgat_batching_matches_per_window_forward():
    """A batched forward must equal running each window alone.

    Passing a single-graph edge_index for a batched input leaves every graph
    after the first with no edges -- shapes still line up, so the failure is
    silent. STGAT offsets the edge index the way PyG's Batch collation does;
    this pins that.
    """
    torch = pytest.importorskip("torch", reason="STGAT needs torch")
    pytest.importorskip("torch_geometric", reason="STGAT needs PyTorch Geometric")
    from xcheck.models import STGAT

    n_nodes = 25
    ring = np.array([[i, (i + 1) % n_nodes] for i in range(n_nodes)]).T
    edge_index = torch.tensor(ring)

    torch.manual_seed(0)
    model = STGAT(in_channels=3, n_pred=3, n_nodes=n_nodes)
    model.eval()
    x = torch.randn(7, n_nodes, 3)

    with torch.no_grad():
        batched = model(x, edge_index)
        one_by_one = torch.cat([model(x[i : i + 1], edge_index) for i in range(7)])
    assert torch.allclose(batched, one_by_one, atol=1e-6)
