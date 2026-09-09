"""Phase-3 ablation under the widened rolling-origin protocol.

Two changes from Phase 2, both sanctioned by docs/ROADMAP.md "Evaluation protocol".

1. **Eight origins instead of three**, with *disjoint* test windows. Phase-2
   effects (~0.5 RMSE) sat far inside the fold-to-fold spread (±16 to ±21), so
   nothing was resolvable at n=9. Widening is the cheapest available increase in
   statistical power.

   Origins 0.40 .. 0.925 spaced 0.075 with ``test_frac=0.075`` makes consecutive
   test windows exactly adjacent rather than overlapping. Overlapping windows
   would inflate the apparent sample size and make a paired test anti-conservative
   — 8 folds that share 57% of their test data are not 8 independent observations.
   The cost is a shorter test window per fold (~34 weeks against 68), traded for
   folds that a paired test can legitimately treat as independent.

   The Phase-2 origins (0.55, 0.70, 0.85) are a subset of this grid, so the old
   folds remain identifiable inside the new results.

2. **Parallel across runs, single-threaded within each.** Measured on this
   workload: one torch thread is ~2.7x faster per epoch than ten, because the
   model is 7k parameters on a (1, 25, 33) input and thread synchronisation costs
   more than the arithmetic. So each worker pins itself to one thread and the pool
   fills the cores instead. See docs/PHASE3_PLAN.md §5.1.

Usage:
    python scripts/run_phase3.py [--quick] [--workers N] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dengue_gnn.experiment import Config  # noqa: E402
from dengue_gnn.mechanistic import MAX_WEEKLY_LOG_GROWTH  # noqa: E402

NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
OUT = REPO / "results" / "phase3_runs.csv"

#: Disjoint test windows; see the module docstring.
ORIGINS = (0.400, 0.475, 0.550, 0.625, 0.700, 0.775, 0.850, 0.925)
TEST_FRAC = 0.075

# Loaded once per worker process by _init, not pickled per job.
_RAW = None
_ADJ = None


def _init(npy: str, adj: str) -> None:
    import torch

    from dengue_gnn.experiment import load_dataset

    # One thread per worker: the pool provides the parallelism, and intra-op
    # threading actively hurts at this tensor size.
    torch.set_num_threads(1)
    global _RAW, _ADJ
    _RAW, _ADJ = load_dataset(npy, adj)[:2]


def _job(payload):
    """Run one unit of work: a graph model, a non-graph baseline, or persistence."""
    from dengue_gnn.baselines import run_baseline_single
    from dengue_gnn.experiment import persistence_rows, run_single

    cfg, fold_idx, seed, kind = payload
    if kind == "persistence":
        return persistence_rows(_RAW, cfg, fold_idx)
    if kind == "model":
        return run_single(_RAW, _ADJ, cfg, fold_idx, seed)
    return run_baseline_single(_RAW, cfg, fold_idx, seed, kind)


#: Deterministic baselines need one seed, not three. Running three identical
#: fits would inflate the paired-test sample with duplicates.
DETERMINISTIC = {"seasonal_naive", "ridge"}


def configs(quick: bool) -> list[Config]:
    common = dict(
        origins=ORIGINS,
        test_frac=TEST_FRAC,
        **(dict(epochs=3, patience=3, seeds=(0,)) if quick else {}),
    )
    return [
        # Stage 1-2 rows, re-run so every row shares the widened protocol.
        Config(label="dense_fixed", use_adaptive=False, lambda_phys=0.0, **common),
        Config(label="adaptive", use_adaptive=True, lambda_phys=0.0, **common),
        # Stage 2b: kept because a negative result still needs its row (D8).
        Config(label="adaptive_lam0.01", use_adaptive=True, lambda_phys=0.01, **common),
        Config(label="adaptive_lam0.1", use_adaptive=True, lambda_phys=0.1, **common),
        Config(label="adaptive_lam1.0", use_adaptive=True, lambda_phys=1.0, **common),
        # Hyper-parameter axis 2: embedding width. The Phase-2 study swept only
        # lambda, which reads as one sweep rather than a tuning study.
        Config(label="adaptive_d4", use_adaptive=True, emb_dim=4, **common),
        Config(label="adaptive_d20", use_adaptive=True, emb_dim=20, **common),
        # Hyper-parameter axis 3: hidden width.
        Config(label="adaptive_h32", use_adaptive=True, hidden=32, **common),
        Config(label="adaptive_h128", use_adaptive=True, hidden=128, **common),
    ]


def stage3_configs(quick: bool) -> list[Config]:
    """Stage-3 trials: mechanistic constraints on observables, plus a curriculum.

    Lambda values come from scripts/measure_loss_scale.py rather than guesswork.
    The growth-smoothness term sits at ~0.17x the data loss, so lambda 0.3 / 1.5
    put it at roughly 5% and 20% of the total -- two orders of magnitude away from
    the 0.01-1.0 grid Stage 2b used, which is exactly why the magnitude check is
    now mandatory (LL-019).

    One variable per row (D7): band alone, smoothness alone, both, and the
    curriculum applied to each of the new term and the old failing one.
    """
    common = dict(
        origins=ORIGINS,
        test_frac=TEST_FRAC,
        **(dict(epochs=3, patience=3, seeds=(0,)) if quick else {}),
    )
    return [
        Config(label="mech_smooth0.3", lambda_mech=0.3, mech_mode="smooth", **common),
        Config(label="mech_smooth1.5", lambda_mech=1.5, mech_mode="smooth", **common),
        Config(label="mech_band1.5", lambda_mech=1.5, mech_mode="band", **common),
        Config(label="mech_both1.5", lambda_mech=1.5, mech_mode="both", **common),
        # Curriculum on the NEW term.
        Config(
            label="mech_smooth1.5_curr",
            lambda_mech=1.5,
            mech_mode="smooth",
            curriculum=0.5,
            **common,
        ),
        # Curriculum on the OLD term that failed in Stage 2b. If this recovers,
        # the failure was optimisation (Krishnapriyan) rather than the constraint.
        Config(label="spatial0.1_curr", lambda_phys=0.1, curriculum=0.5, **common),
    ]


def search_configs(quick: bool) -> list[Config]:
    """The Stage-3 search: replication first, breadth second.

    Mirrors notebooks/04_kaggle_search.ipynb exactly, so a local run and a Kaggle
    run produce interchangeable rows.

    The split of seeds is deliberate. A paired power analysis on our own results
    says the effects we care about need n~44-54 to reach significance and we had
    24, so **Stage A** re-runs the configurations we already believe in at 8 seeds
    (n=64). **Stage B** only scans lambda to locate an optimum, which does not need
    that power, so it runs 3 seeds; whichever lambda wins can be promoted to Stage A
    afterwards. Spending the whole budget on breadth would buy false positives
    rather than answers.
    """
    common = dict(origins=ORIGINS, test_frac=TEST_FRAC)
    a_seeds = (0,) if quick else tuple(range(8))
    b_seeds = (0,) if quick else (0, 1, 2)
    ep = dict(epochs=3, patience=3) if quick else {}

    stage_a = [
        Config(label="A_dense_fixed", use_adaptive=False, seeds=a_seeds, **common, **ep),
        Config(label="A_adaptive", use_adaptive=True, seeds=a_seeds, **common, **ep),
        Config(
            label="A_mech0.3",
            lambda_mech=0.3,
            mech_mode="smooth",
            seeds=a_seeds,
            **common,
            **ep,
        ),
        Config(
            label="A_mech0.3_curr",
            lambda_mech=0.3,
            mech_mode="smooth",
            curriculum=0.5,
            seeds=a_seeds,
            **common,
            **ep,
        ),
    ]
    stage_b = [
        Config(
            label=f"B_mech{lam}",
            lambda_mech=lam,
            mech_mode="smooth",
            seeds=b_seeds,
            **common,
            **ep,
        )
        for lam in (0.05, 0.10, 0.15, 0.20, 0.45, 0.60)
    ]
    return stage_a + stage_b


def promote_configs(quick: bool) -> list[Config]:
    """Promote the winning smoothness weight from the 3-seed scan to 8 seeds.

    This is the second half of the Stage A/B design in :func:`search_configs`,
    which was never run. Stage B scanned lambda at 3 seeds to locate an optimum
    and left promotion to Stage A for whichever won; lambda=0.15 took the best
    mean of any arm in the study (58.63) but at 17/24, p=0.064, which our own
    power analysis says is exactly the regime where an effect looks real and is
    not yet resolvable.

    Eight seeds gives n=64 paired comparisons, the size at which lambda=0.3
    resolved to p=0.0006. lambda=0.05 comes along because it was the other arm
    under the control's mean and the two bracket the optimum; without it a win
    for 0.15 cannot be told from a broad plateau.

    The control and persistence rows are re-run here rather than borrowed, so
    this file is interpretable on its own -- the search CSV predates the
    resid_ratio and max_growth columns and will not merge with it.
    """
    seeds = (0,) if quick else tuple(range(8))
    ep = dict(epochs=3, patience=3) if quick else {}
    common = dict(origins=ORIGINS, test_frac=TEST_FRAC, seeds=seeds, **ep)
    return [
        Config(label="P_adaptive", lambda_mech=0.0, **common),
        *(
            Config(label=f"P_mech{lam}", lambda_mech=lam, mech_mode="smooth", **common)
            for lam in (0.05, 0.15, 0.30)
        ),
    ]


def shrink_configs(quick: bool) -> list[Config]:
    """Residual shrinkage, as a self-contained comparison.

    The network predicts a correction to persistence, so scaling that correction
    by gamma interpolates between persistence (gamma=0) and the raw model
    (gamma=1). A probe over three folds found an *interior* optimum near 0.6 that
    beat both endpoints -- the model carries signal, but less than full weight
    implies.

    gamma is selected on the validation fold, never on test. This file carries its
    own control and persistence rows so it is interpretable without merging
    against runs that predate the shrink_gamma column.
    """
    seeds = (0,) if quick else tuple(range(8))
    ep = dict(epochs=3, patience=3) if quick else {}
    common = dict(origins=ORIGINS, test_frac=TEST_FRAC, seeds=seeds, **ep)
    return [
        Config(label="S_adaptive", use_adaptive=True, **common),
        Config(label="S_adaptive_shrunk", use_adaptive=True, shrink=True, **common),
        Config(
            label="S_mech0.3_shrunk",
            lambda_mech=0.3,
            mech_mode="smooth",
            shrink=True,
            **common,
        ),
    ]


def final_configs(quick: bool) -> list[Config]:
    """The final Phase-3 comparison: shrinkage and augmentation, one schema.

    Two arms with different seed budgets, for a reason. The **shrinkage** arm
    carries the claim that might beat persistence, so it gets 8 seeds (n=64,
    above the ~54 the paired power analysis demands). The **augmentation** arm
    only has to establish an ordering between the GAN and cheap augmentation --
    a much larger expected effect -- so 3 seeds suffice, and spending 8 there
    would buy precision nobody needs.

    Every augmentation arm runs unshrunk, so augmentation and shrinkage are never
    confounded (D7: one variable at a time).
    """
    ep = dict(epochs=3, patience=3) if quick else {}
    many = (0,) if quick else tuple(range(8))
    few = (0,) if quick else (0, 1, 2)
    base = dict(origins=ORIGINS, test_frac=TEST_FRAC, **ep)
    gan_ep = dict(gan_epochs=30) if quick else {}

    shrink_arm = [
        Config(label="F_adaptive", seeds=many, **base),
        Config(label="F_adaptive_shrunk", shrink=True, seeds=many, **base),
        Config(
            label="F_mech0.3_shrunk",
            lambda_mech=0.3,
            mech_mode="smooth",
            shrink=True,
            seeds=many,
            **base,
        ),
    ]
    # The temporal encoder is the largest architectural gap: the model had no
    # temporal operator at all, while every method it is compared against has one.
    temporal_arm = [
        Config(label="F_gtcn", temporal="gtcn", seeds=many, **base),
        Config(label="F_gtcn_shrunk", temporal="gtcn", shrink=True, seeds=many, **base),
    ]
    augment_arm = [
        Config(label=f"F_aug_{kind}", augment=kind, augment_ratio=0.5, seeds=few, **base, **gan_ep)
        for kind in ("jitter", "window_warp", "gan")
    ]
    return shrink_arm + temporal_arm + augment_arm


def window_configs(quick: bool) -> list[Config]:
    """Receptive-field study: how much history does a dengue forecaster need?

    W=3 was inherited from Weng et al. and never questioned. It is almost exactly
    the information persistence uses (one week), which makes "the GNN cannot beat
    persistence" close to tautological: the model is given no more history than the
    baseline it must beat.

    The target's autocorrelation is 0.91 at lag 1, 0.72 at lag 4 and 0.33 at lag
    13, so structure demonstrably exists out to roughly a quarter. There is no
    annual cycle (lag 52 is -0.03), which is also why seasonal naive fails badly.
    The grid therefore spans one week to three quarters.

    All arms use the gated temporal convolution: a wider window is only useful to a
    model that has a temporal operator to exploit it.
    """
    seeds = (0,) if quick else (0, 1, 2)
    ep = dict(epochs=3, patience=3) if quick else {}
    return [
        Config(
            label=f"W{w}",
            window=w,
            temporal="gtcn",
            origins=ORIGINS,
            test_frac=TEST_FRAC,
            seeds=seeds,
            **ep,
        )
        for w in (3, 8, 16, 26, 39)
    ]


def arch_configs(quick: bool) -> list[Config]:
    """Published architectures, reimplemented on our shared propagation path.

    `torch_geometric_temporal` is not installed and ADR-0002 rejected it for this
    pipeline, so STGAT and A3TGCN are reimplemented here rather than imported.
    That is a deliberate deviation with a cost and a benefit.

    The cost: these are *our* renderings, not the authors' code, so a gap could be
    an implementation difference rather than a real one.

    The benefit: every arm shares one propagation path, one target transform, one
    optimiser and one evaluation, so a difference between rows is attributable to
    the architecture alone (D7). Importing five upstream implementations would
    reintroduce exactly the confound review finding F6 was about.

    Composition:
      GAT     = spatial gat, no temporal operator
      STGAT   = spatial gat + LSTM temporal
      A3TGCN  = spatial gcn + GRU temporal with attention over timesteps
    """
    seeds = (0,) if quick else (0, 1, 2)
    ep = dict(epochs=3, patience=3) if quick else {}
    common = dict(origins=ORIGINS, test_frac=TEST_FRAC, seeds=seeds, window=3, **ep)
    return [
        Config(label="X_gcn", spatial="gcn", temporal="none", **common),
        Config(label="X_gcn_gtcn", spatial="gcn", temporal="gtcn", **common),
        Config(label="X_gat", spatial="gat", temporal="none", **common),
        Config(label="X_stgat", spatial="gat", temporal="lstm", **common),
        Config(label="X_a3tgcn", spatial="gcn", temporal="gru_attn", **common),
        # Capacity controls. At hidden=64 the arms above are NOT parameter-matched
        # on this dataset (25 nodes, 11 features, 459 weeks): GCN 7,032,
        # GCN+gTCN 11,960, A3TGCN 23,865, GAT 52,408, STGAT 87,992. A 7.5x
        # parameter gap is on its own a plausible explanation for any GAT result
        # in either direction, so "GAT vs GCN" at default width varies
        # architecture and capacity together -- the confound D7 and review
        # finding F6 are about.
        #
        # Matched pairs make it attributable:
        #   X_gat_h1   7,160 vs control 7,032  (1.8%)  attention vs uniform
        #   X_gat_h2  13,624 vs h96    13,368  (1.9%)  does a 2nd head pay?
        #   X_gcn_h128 21,752 vs A3TGCN 23,865 (8.9%)  recurrence vs just wider
        Config(label="X_gat_h1", spatial="gat", temporal="none", gat_heads=1, **common),
        Config(label="X_gat_h2", spatial="gat", temporal="none", gat_heads=2, **common),
        Config(label="X_gcn_h96", spatial="gcn", temporal="none", hidden=96, **common),
        Config(label="X_gcn_h128", spatial="gcn", temporal="none", hidden=128, **common),
    ]


def ceiling_configs(quick: bool) -> list[Config]:
    """Does the band constraint work once its ceiling is derived rather than guessed?

    Stage 2b and Stage 3 both found the growth-band term inert or harmful, and
    both diagnosed it as a weighting problem. It was not. The ceiling was set at
    0.70 from ``ln(8)/3``, a formula that assumes a fixed generation interval;
    the SEIR-SEI stage durations are exponential, and the linearised model of
    Phaijoo & Gurung (2018) puts the bound at 2.39 for R0 <= 6. The old value was
    exceeded by 20.9% of the observed district-week growth rates, so the term was
    penalising forecasts for reproducing real outbreaks.

    One variable: the ceiling. Both arms share lambda, mode, seeds and folds, so
    a difference between them is the constant and nothing else (D7). The
    unconstrained control is there because the interesting outcome is not "new
    beats old" -- it is whether either beats no constraint at all.
    """
    seeds = (0,) if quick else (0, 1, 2)
    ep = dict(epochs=3, patience=3) if quick else {}
    common = dict(origins=ORIGINS, test_frac=TEST_FRAC, seeds=seeds, **ep)
    arms = [Config(label="C_none", lambda_mech=0.0, **common)]
    for lam in (0.3, 1.5):
        for tag, ceil in (("old0.70", 0.70), ("new2.39", MAX_WEEKLY_LOG_GROWTH)):
            arms.append(
                Config(
                    label=f"C_band{lam}_{tag}",
                    lambda_mech=lam,
                    mech_mode="band",
                    max_growth=ceil,
                    **common,
                )
            )
    return arms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 epochs, 1 seed — pipeline check")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument(
        "--only",
        choices=(
            "all",
            "models",
            "baselines",
            "stage3",
            "search",
            "shrink",
            "final",
            "window",
            "arch",
            "ceiling",
            "promote",
        ),
        default="all",
        help="'baselines' runs only the non-graph comparators, into --out",
    )
    ap.add_argument(
        "--labels",
        default="",
        help="Comma-separated label filter, e.g. 'X_gat_h1,X_gcn_h96'. Runs only "
        "those arms, so one arm can be added to an existing study without "
        "re-running the rest. Persistence is always kept, since the paired "
        "tests need it.",
    )
    ap.add_argument(
        "--note",
        default="",
        help="One line recorded in the manifest saying what question this run answers.",
    )
    args = ap.parse_args()

    from dengue_gnn.baselines import BASELINE_KINDS

    cfgs = configs(args.quick)
    jobs = []

    if args.only in ("all", "models"):
        for cfg in cfgs:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs[0].origins)):
            jobs.append((cfgs[0], fold_idx, 0, "persistence"))

    if args.only == "stage3":
        for cfg in stage3_configs(args.quick):
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))

    if args.only == "search":
        cfgs_s = search_configs(args.quick)
        for cfg in cfgs_s:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_s[0].origins)):
            jobs.append((cfgs_s[0], fold_idx, 0, "persistence"))

    if args.only == "shrink":
        cfgs_k = shrink_configs(args.quick)
        for cfg in cfgs_k:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_k[0].origins)):
            jobs.append((cfgs_k[0], fold_idx, 0, "persistence"))

    if args.only == "final":
        cfgs_f = final_configs(args.quick)
        for cfg in cfgs_f:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_f[0].origins)):
            jobs.append((cfgs_f[0], fold_idx, 0, "persistence"))

    if args.only == "window":
        cfgs_w = window_configs(args.quick)
        for cfg in cfgs_w:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        # persistence is window-independent; emit against the narrowest arm
        for fold_idx in range(len(cfgs_w[0].origins)):
            jobs.append((cfgs_w[0], fold_idx, 0, "persistence"))

    if args.only == "promote":
        cfgs_p = promote_configs(args.quick)
        for cfg in cfgs_p:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_p[0].origins)):
            jobs.append((cfgs_p[0], fold_idx, 0, "persistence"))

    if args.only == "ceiling":
        cfgs_c = ceiling_configs(args.quick)
        for cfg in cfgs_c:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_c[0].origins)):
            jobs.append((cfgs_c[0], fold_idx, 0, "persistence"))

    if args.only == "arch":
        cfgs_x = arch_configs(args.quick)
        for cfg in cfgs_x:
            for fold_idx in range(len(cfg.origins)):
                for seed in cfg.seeds:
                    jobs.append((cfg, fold_idx, seed, "model"))
        for fold_idx in range(len(cfgs_x[0].origins)):
            jobs.append((cfgs_x[0], fold_idx, 0, "persistence"))

    if args.only in ("all", "baselines"):
        base_cfg = cfgs[0]
        for kind in BASELINE_KINDS:
            seeds = (0,) if kind in DETERMINISTIC else base_cfg.seeds
            for fold_idx in range(len(base_cfg.origins)):
                for seed in seeds:
                    jobs.append((base_cfg, fold_idx, seed, kind))

    if args.labels:
        keep = {x.strip() for x in args.labels.split(",") if x.strip()}
        unknown = keep - {j[0].label for j in jobs}
        if unknown:
            raise SystemExit(f"--labels names arms not in mode '{args.only}': {sorted(unknown)}")
        jobs = [j for j in jobs if j[0].label in keep or j[3] == "persistence"]

    print(f"{len(jobs)} jobs ({args.only}) on {args.workers} workers", flush=True)

    # Provenance beside the results, written by the run itself rather than
    # transcribed afterwards (D2, and review finding F1). Written *before* the
    # pool starts, so a run that crashes half way still leaves a record of what
    # it was and which commit it ran against -- which is exactly the run whose
    # partial CSV would otherwise be unattributable.
    from dengue_gnn.provenance import write_manifest

    cfgs_seen = list({j[0].label: j[0] for j in jobs}.values())
    manifest = write_manifest(
        args.out,
        cfgs_seen,
        len(jobs),
        repo=REPO,
        extra={
            "mode": args.only,
            "quick": args.quick,
            "workers": args.workers,
            "note": args.note,
            "status": "started",
        },
    )
    print(f"manifest -> {manifest.name}", flush=True)

    rows: list[dict] = []
    started = time.time()
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init, initargs=(str(NPY), str(ADJ))
    ) as pool:
        futures = {pool.submit(_job, j): j for j in jobs}
        for fut in as_completed(futures):
            _cfg, _fold_idx, seed, _kind = futures[fut]
            got = fut.result()
            rows.extend(got)
            done += 1
            if got:
                pooled = next(r for r in got if r["horizon"] == 0)
                print(
                    f"  [{done:3d}/{len(jobs)}] {pooled['label']:18s} "
                    f"fold{pooled['fold']} seed{seed}  RMSE {pooled['rmse']:6.2f}"
                    f"  ({time.time() - started:5.0f}s elapsed)",
                    flush=True,
                )

    # Second write, now that the outcome is known. Overwrites the pre-run
    # manifest; if the run had crashed, that first one would still stand.
    write_manifest(
        args.out,
        cfgs_seen,
        len(jobs),
        repo=REPO,
        extra={
            "mode": args.only,
            "quick": args.quick,
            "workers": args.workers,
            "note": args.note,
            "status": "complete",
            "n_rows": len(rows),
            "wall_seconds": round(time.time() - started, 1),
        },
    )

    rows.sort(key=lambda r: (r["label"], r["fold"], r["seed"], r["horizon"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - started
    # resolve() first: --out may be relative, and relative_to() raises on a
    # relative/absolute mix -- which crashed this report line *after* the CSV had
    # already been written, making a successful run look like a failure.
    try:
        shown = args.out.resolve().relative_to(REPO)
    except ValueError:
        shown = args.out
    print(
        f"\nwrote {len(rows)} rows to {shown} in {elapsed:.0f}s "
        f"({elapsed / max(len(jobs), 1):.1f}s per job)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
