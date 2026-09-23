"""The configuration grids. Each function returns a list of configs for sweep.run."""

from __future__ import annotations


def _c(name, **kw):
    return dict(name=name, **kw)


def screen(epochs: int = 300) -> list[dict]:
    """Which of the four defects actually cost accuracy, and does the graph help?

    Factors varied one at a time from a common reference so each difference is
    attributable: head, loss, graph mode, and the seasonal feature EDA F9 says
    no model in this project has ever seen.
    """
    out = []
    for head in ("direct", "residual", "foi", "foi_res"):
        out.append(_c(f"head={head}", backbone="gcn", head=head, loss="mse_z", epochs=epochs))
    for loss in ("mse_z", "huber_z", "mse_raw", "smape"):
        out.append(_c(f"loss={loss}", backbone="gcn", head="residual", loss=loss, epochs=epochs))
    for bb in ("none", "gcn", "adaptive", "hybrid"):
        out.append(_c(f"graph={bb}", backbone=bb, head="residual", loss="mse_z", epochs=epochs))
    out.append(_c("feat=season", backbone="gcn", head="residual", loss="mse_z",
                  use_season=True, epochs=epochs))
    out.append(_c("feat=climate", backbone="gcn", head="residual", loss="mse_z",
                  use_climate=True, epochs=epochs))
    out.append(_c("feat=all", backbone="gcn", head="residual", loss="mse_z",
                  use_climate=True, use_ndvi=True, use_season=True, epochs=epochs))
    return out


def foi(epochs: int = 300) -> list[dict]:
    """Does the physics head survive the two defects ``diagnose_foi.py`` measured?

    The diagnostic found the force-of-infection path broken in two separable
    ways: ``sigmoid`` starts 809x above the inverted median and puts 57% of
    cells where its gradient is 150x below maximum, and the ``lambda = 0`` floor
    already overshoots 14% of targets because ``E0`` is seeded from
    ``cases[t-2]/rho``. Each fix is varied alone, then together, against the two
    non-physics heads on the same backbone so the comparison is attributable.
    """
    out = []
    for head in ("direct", "residual"):
        out.append(_c(f"ref:{head}", backbone="gcn", head=head, loss="mse_z", epochs=epochs))
    for head in ("foi", "foi_res"):
        for lam in ("sigmoid", "log"):
            for fit in (False, True):
                tag = f"{head} lam={lam}{' +statefit' if fit else ''}"
                out.append(_c(tag, backbone="gcn", head=head, loss="mse_z",
                              lam_param=lam, state_fit=fit, epochs=epochs))
    return out


def converge(epochs: int = 3000) -> list[dict]:
    """Is the physics head undertrained, or converged and losing?

    The objection the short screen cannot answer: graph networks routed through
    a simulator may simply need more steps than a direct regressor. Ten times
    the epoch budget and five times the patience, so a run that is still
    improving has room to show it. ``best_epoch`` against ``epochs_ran`` in the
    output says which of the two happened.
    """
    out = []
    for head in ("direct", "residual", "foi", "foi_res"):
        for lr in (3e-3, 1e-3):
            out.append(_c(f"{head} lr={lr:g}", backbone="gcn", head=head, loss="mse_z",
                          lam_param="log" if head.startswith("foi") else "sigmoid",
                          state_fit=head.startswith("foi"),
                          lr=lr, epochs=epochs, patience=200))
    return out


def real(epochs: int = 300) -> list[dict]:
    """The comparison the paper needs: six encoders, one harness, one protocol.

    Every published architecture plus Liu et al.'s LSTM, each behind the direct
    head and behind the force-of-infection head. ``direct`` is what the five GNN
    papers do; ``foi`` is the proposal. ``LSTM + foi`` is SEIR-LSTM. Because all
    of them share folds, loss, early stopping and metric here, the differences
    are attributable to the encoder and the head and to nothing else -- which was
    never true of the S4 vs S5 comparison in the paper.
    """
    out = []
    for bb in ("LSTM", "STGAT", "A3TGCN", "ASTGCN", "AAGCN", "DCRNN"):
        for head in ("direct", "foi", "foi_res"):
            out.append(_c(f"{bb}+{head}", backbone=bb, head=head, loss="mse_z",
                          lam_param="log", state_fit=True, epochs=epochs))
    return out


def combo(epochs: int = 400) -> list[dict]:
    """Stack the levers that individually moved the metric.

    The six-encoder grid says architecture buys little: the spread from best to
    worst *working* encoder is 0.15 val RMSE, while the seasonal feature alone
    was worth 0.88 in the screen. So the search moves off architecture and onto
    what is fed in and what is optimised. AAGCN and ASTGCN carry the two best
    encoders forward, LSTM stays as Liu et al.'s control, and every cell is
    crossed with the feature set, the output distribution and the head.
    """
    out = []
    for bb in ("AAGCN", "ASTGCN", "LSTM"):
        for feats, tag in (({"use_season": True}, "season"),
                           ({"use_season": True, "use_climate": True,
                             "use_ndvi": True}, "all")):
            for dist, loss in (("point", "mse_z"), ("nb", "nb")):
                for head in ("direct", "foi_res"):
                    out.append(_c(f"{bb}+{head} {tag} {dist}", backbone=bb, head=head,
                                  loss=loss, dist=dist, lam_param="log", state_fit=True,
                                  epochs=epochs, **feats))
    return out


def window(epochs: int = 400) -> list[dict]:
    """The one protocol deviation worth testing: how much history the model sees.

    ``WINDOW = 3`` is inherited from the benchmark paper, and three weekly points
    can barely estimate a trend. It is the largest untested lever in the study,
    and under plan R5 changing it is a deviation -- logged, not silent.

    A longer window shifts the fold boundaries, so `sweep.run` emits a separate
    persistence row per window and arms are never paired across windows. The four
    carried arms are the survivors of `combo`: the best direct and the best
    physics arm, each with its matched LSTM control.
    """
    out = []
    for w in (3, 6, 12):
        for bb, head in (("AAGCN", "direct"), ("LSTM", "direct"),
                         ("ASTGCN", "foi_res"), ("LSTM", "foi_res")):
            out.append(_c(f"{bb}+{head} w={w}", backbone=bb, head=head, loss="nb",
                          dist="nb", use_season=True, lam_param="log", state_fit=True,
                          window=w, epochs=epochs))
    return out


def confirm(epochs: int = 400) -> list[dict]:
    """The confirmatory stage (plan S9), on nine origins with disjoint test spans.

    Three origins cannot clear p = 0.25 on an exact sign-flip test, so nothing in
    the screening grids is significant at the pairing unit that matters. Nine
    disjoint origins take the floor to 0.004.

    The finalists are the survivors of `combo`, frozen before this runs: the best
    direct arm, the best physics arm, and the matched LSTM control for each, plus
    the physics arm's own encoder on the direct head so that "the graph helps
    only through the physics" has its control.

    Note this replaces `analysis/_build/run_s9_confirmatory.py`, which claims nine
    origins but runs on three and subtracts hardcoded scalars instead of pairing
    (EXP-037).
    """
    arms = (("AAGCN", "direct"), ("LSTM", "direct"),
            ("ASTGCN", "foi_res"), ("LSTM", "foi_res"), ("ASTGCN", "direct"))
    return [_c(f"{bb}+{head}", backbone=bb, head=head, loss="nb", dist="nb",
               use_season=True, lam_param="log", state_fit=True,
               origins="nine", epochs=epochs)
            for bb, head in arms]


def remedies(epochs: int = 400) -> list[dict]:
    """The pre-registered remedies from docs/REMEDIES_PLAN.md, screened on the frozen three origins.

    ``B`` is the best configuration found so far and the control every remedy is
    paired against. Each remedy changes exactly one thing relative to it, except
    R3 and R4a, which are different model families by design. ``SEIR-LSTM`` is
    Liu et al.'s encoder behind the same physics head, loss and folds, kept so
    the proposal's comparison is always in view. Run with ``--keep`` so R5 can
    be formed afterwards by ``ensemble.py``.
    """
    base = dict(head="direct", loss="nb", dist="nb", use_season=True, epochs=epochs)
    phys = dict(lam_param="log", state_fit=True)
    return [
        _c("B", backbone="AAGCN", **base),
        _c("R1a revin_mean", backbone="AAGCN", norm="revin_mean", **base),
        _c("R1b revin", backbone="AAGCN", norm="revin", **base),
        _c("R2 node_emb", backbone="AAGCN", node_emb=16, **base),
        _c("R3 stid", backbone="none", node_emb=16, **base),
        _c("R4a nbglm", backbone="linear", node_emb=8, **base),
        _c("R4b knn", backbone="knn"),
        _c("R6 aux_phys 0.1", backbone="AAGCN", aux_phys=0.1, **base, **phys),
        _c("R6 aux_phys 0.3", backbone="AAGCN", aux_phys=0.3, **base, **phys),
        _c("SEIR-LSTM", backbone="LSTM", head="foi_res", loss="nb", dist="nb",
           use_season=True, epochs=epochs, **phys),
    ]


#: Pre-registered equal-weight ensembles per grid (docs/REMEDIES_PLAN.md, R5).
#: Fixed before the grid runs; ``build_seirgnn2_kernel.py --keep`` computes them.
ENSEMBLES = {
    "remedies": [["B", "R4b knn"],
                 ["B", "R4a nbglm"],
                 ["B", "R4a nbglm", "R4b knn"]],
}
