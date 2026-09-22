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
