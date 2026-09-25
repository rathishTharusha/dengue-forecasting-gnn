# %% [markdown]
# ## 7. Every configuration, trained here
#
# Each configuration the paper reports is defined once below and trained on every
# (origin, seed). Because everything runs in one environment with one code path, any
# two configurations evaluated on the same origins can be paired directly — a shared
# control (the best model, **B**) is trained once and used by every comparison that
# needs it.
#
# * **Three origins** (0.55, 0.70, 0.85) × seeds 0, 1, 2 — the screening protocol.
# * **Nine disjoint origins** (0.40 + k/15) × seeds 0, 1, 2 — the confirmation, where
#   the origin-level paired test can reach p = 0.004.

# %%
ENC = ["AAGCN", "ASTGCN", "LSTM", "STGAT", "A3TGCN", "DCRNN"]
PHYS = dict(lam_param="log", state_fit=True)
NBS = dict(loss="nb", dist="nb", use_season=True, epochs=400)            # NB likelihood + seasonal features
B = dict(backbone="AAGCN", head="direct", **NBS)                          # the best model on validation
BLOCKS_3 = ((2, 5), (6, 9), (10, 13))
BLOCKS_6 = ((2, 5), (6, 9), (10, 13), (14, 17), (18, 21), (22, 25))

CONFIGS3: dict[str, dict] = {}
# Six encoders x five heads, squared error, no seasonal features (paper Tables 2 and 5).
for e in ENC:
    for h in ("direct", "residual", "gated", "foi", "foi_res"):
        CONFIGS3[f"{e}+{h}"] = dict(backbone=e, head=h, loss="mse_z", epochs=300, **PHYS)
# The first screen, on a toy graph layer: does the graph help, does the season help?
for g in ("gcn", "none", "adaptive"):
    CONFIGS3[f"graph={g}"] = dict(backbone=g, head="residual", loss="mse_z", epochs=300)
CONFIGS3["feat=season"] = dict(backbone="gcn", head="residual", loss="mse_z", use_season=True, epochs=300)
# The best model and the likelihood comparison.
CONFIGS3["B"] = B
CONFIGS3["B, squared error"] = dict(B, loss="mse_z", dist="point")
for e in ("AAGCN", "ASTGCN", "LSTM"):
    CONFIGS3[f"{e}+foi_res, NB+season"] = dict(backbone=e, head="foi_res", **NBS, **PHYS)
for e in ("ASTGCN", "LSTM", "STGAT", "A3TGCN"):
    CONFIGS3[f"{e}+direct, NB+season"] = dict(backbone=e, head="direct", **NBS)
# Architecture changes to B.
CONFIGS3["A1 residual head"] = dict(B, head="residual")
CONFIGS3["A2 district seasonal curves"] = dict(B, dseason=True)
CONFIGS3["A3 MLP head"] = dict(B, head_mlp=64)
CONFIGS3["A4 MLP head + climate 2-13"] = dict(B, head_mlp=64, clim_blocks=BLOCKS_3)
CONFIGS3["A5 global context"] = dict(B, global_ctx=True)
CONFIGS3["A6 combined"] = dict(B, head_mlp=64, dseason=True, global_ctx=True, node_emb=16, clim_blocks=BLOCKS_3)
# Remedies from the forecasting literature.
CONFIGS3["R1 RevIN"] = dict(B, norm="revin")
CONFIGS3["R2 district embedding"] = dict(B, node_emb=16)
CONFIGS3["R3 STID-style MLP"] = dict(B, backbone="none", node_emb=16)
CONFIGS3["R4a NB-GLM"] = dict(B, backbone="linear", node_emb=8)
CONFIGS3["R4b k-NN"] = dict(kind="knn")
CONFIGS3["R6 SEIR auxiliary loss"] = dict(B, aux_phys=0.1, **PHYS)
# Data.
for frac in (0.25, 0.5, 0.75):
    CONFIGS3[f"B, {int(frac * 100)}% of training"] = dict(B, train_frac=frac)
CONFIGS3["G2 TimeGAN"] = dict(B, augment="timegan")
CONFIGS3["G3 SEIR pre-training"] = dict(B, augment="seir_pretrain_cal")
CONFIGS3["K1 climate lags 2-4"] = dict(B, use_climate=True)
CONFIGS3["K2 climate lags 2-13"] = dict(B, clim_blocks=BLOCKS_3)
CONFIGS3["K3 climate lags 2-25"] = dict(B, clim_blocks=BLOCKS_6)
CONFIGS3["K6 trees + climate"] = dict(kind="gbm", clim_blocks=BLOCKS_3)
CONFIGS3["K7 trees"] = dict(kind="gbm")
# Physics as structure.
CONFIGS3["P1 spatial penalty"] = dict(B, spatial=0.001, spatial_kind="ratio")
CONFIGS3["P2 spatial penalty (log)"] = dict(B, spatial=0.05, spatial_kind="log")
CONFIGS3["P4 metapopulation SEIR"] = dict(backbone="AAGCN", head="foi_meta", **NBS, **PHYS)
CONFIGS3["P5 metapopulation + penalty + season"] = dict(backbone="AAGCN", head="foi_meta", spatial=0.001,
                                                        dseason=True, **NBS, **PHYS)

CONFIGS9 = {
    "B": B,
    "LSTM+direct, NB+season": dict(backbone="LSTM", head="direct", **NBS),
    "ASTGCN+direct, NB+season": dict(backbone="ASTGCN", head="direct", **NBS),
    "AAGCN+foi_res, NB+season": dict(backbone="AAGCN", head="foi_res", **NBS, **PHYS),
    "ASTGCN+foi_res, NB+season": dict(backbone="ASTGCN", head="foi_res", **NBS, **PHYS),
    "LSTM+foi_res, NB+season": dict(backbone="LSTM", head="foi_res", **NBS, **PHYS),
    "P4 metapopulation SEIR": dict(backbone="AAGCN", head="foi_meta", **NBS, **PHYS),
}

SEEDS = (0, 1, 2)
ORIGIN_SETS = {"three": (ORIGINS, TEST_FRAC), "nine": (ORIGINS_9, TEST_FRAC_9)}
if QUICK:
    SEEDS = (0,)
    ORIGIN_SETS = {"three": ((0.70,), TEST_FRAC), "nine": ((0.40 + 4 / 15, 0.40 + 5 / 15), TEST_FRAC_9)}
FOLDS = {k: {f.origin: f for f in build_folds(o, tf)} for k, (o, tf) in ORIGIN_SETS.items()}

JOBS = [("three", name, o, s) for name in CONFIGS3 for o in FOLDS["three"] for s in SEEDS]
JOBS += [("nine", name, o, s) for name in CONFIGS9 for o in FOLDS["nine"] for s in SEEDS]
print(f"{len(CONFIGS3)} configurations x {len(FOLDS['three'])} origins x {len(SEEDS)} seeds "
      f"+ {len(CONFIGS9)} x {len(FOLDS['nine'])} x {len(SEEDS)}  =  {len(JOBS)} training runs")

# %%
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


#: The configurations whose training-window predictions are kept for the ceiling analysis (8.6).
CEILING = {"B", "ASTGCN+direct, NB+season", "LSTM+direct, NB+season", "STGAT+direct, NB+season",
           "A3TGCN+direct, NB+season", "LSTM+foi_res, NB+season"}


def run_job(job):
    oset, name, origin, seed = job
    torch.set_num_threads(1)
    cfg = dict((CONFIGS3 if oset == "three" else CONFIGS9)[name])
    kind = cfg.pop("kind", "net")
    if QUICK:
        cfg["epochs"] = 3
    fold = FOLDS[oset][origin]
    t0 = time.time()
    if kind == "knn":
        row, extra = run_knn(fold, seed)
    elif kind == "gbm":
        row, extra = run_gbm(fold, seed, **cfg)
    else:
        row, extra = run_fold(fold, seed=seed, keep_train=oset == "three" and name in CEILING, **cfg)
    row.update(origin_set=oset, name=name, origin=origin, seed=seed, elapsed=round(time.time() - t0, 1))
    return job, row, extra


def expected_cost(job):
    """Rough relative cost, so the longest runs start first and the pool stays busy."""
    cfg = (CONFIGS3 if job[0] == "three" else CONFIGS9)[job[1]]
    return ({"DCRNN": 8, "ASTGCN": 4, "A3TGCN": 3}.get(cfg.get("backbone"), 1)
            * (6 if cfg.get("augment") in ("timegan", "seir_pretrain_cal") else 1))


LOG = OUT / "runs.jsonl"
done = {}
if LOG.exists():                         # resume an interrupted session
    for line in LOG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        done[(r["origin_set"], r["name"], r["origin"], r["seed"])] = r
PRED_FILE = OUT / "predictions.pkl"
PREDS = pickle.loads(PRED_FILE.read_bytes()) if PRED_FILE.exists() else {}
todo = sorted((j for j in JOBS if j not in done), key=expected_cost, reverse=True)
print(f"{len(done)} runs already on disk, {len(todo)} to train with {WORKERS} worker(s)")

t_start = time.time()
if todo:
    use_pool = WORKERS > 1 and "fork" in mp.get_all_start_methods()
    pool = ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) if use_pool else None
    futures = [pool.submit(run_job, j) for j in todo] if pool else None
    stream = (f.result() for f in as_completed(futures)) if pool else (run_job(j) for j in todo)
    with LOG.open("a", encoding="utf-8") as log:
        for k, (job, row, extra) in enumerate(stream, 1):
            done[job] = row
            PREDS[job] = extra
            log.write(json.dumps(row) + "\n")
            log.flush()
            if k % 25 == 0 or k == len(todo):
                el = time.time() - t_start
                print(f"  {k}/{len(todo)} runs  {el / 60:6.1f} min elapsed, ~{el / k * (len(todo) - k) / 60:6.1f} min left",
                      flush=True)
                PRED_FILE.write_bytes(pickle.dumps(PREDS))
    if pool:
        pool.shutdown()
PRED_FILE.write_bytes(pickle.dumps(PREDS))
print(f"training finished: {len(done)} runs in {(time.time() - t_start) / 3600:.2f} h this session")
