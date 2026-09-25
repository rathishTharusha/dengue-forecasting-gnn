# %% [markdown]
# ## 8. Results — computed only from the runs above
#
# **How to read the numbers.** RMSE is in weekly cases; lower is better. Validation
# decides and test is only reported: every run was early-stopped on validation, and no
# choice below looks at test. Pairs are compared on matched (origin, seed) runs with an
# exact two-sided sign-flip permutation test, Benjamini–Hochberg-corrected across each
# table. With three origins the *origin*-level test cannot reach p < 0.25, so
# three-origin p-values (9 (origin, seed) pairs) measure run-to-run stability; claims
# about the series rest on the nine-origin runs.

# %%
import itertools

rows = pd.DataFrame(list(done.values()))


def persistence_rows(oset):
    out = []
    for o, fold in FOLDS[oset].items():
        pk = {s: build_tensors(fold, s) for s in ("val", "test")}
        r = score(pk["test"]["p_raw"].numpy(), pk["test"]["y_raw"].numpy())
        r.update(origin_set=oset, name="persistence", origin=o, seed=-1,
                 val_RMSE=rmse(pk["val"]["p_raw"].numpy(), pk["val"]["y_raw"].numpy()))
        out.append(r)
    return out


def ensemble_rows(members, name, oset="three"):
    out = []
    for o in FOLDS[oset]:
        for s in SEEDS:
            parts = [PREDS[(oset, m, o, s)] for m in members]
            val = np.mean([p["pred"]["val"] for p in parts], 0)
            test = np.mean([p["pred"]["test"] for p in parts], 0)
            r = score(test, parts[0]["truth"]["test"])
            r.update(origin_set=oset, name=name, origin=o, seed=s, val_RMSE=rmse(val, parts[0]["truth"]["val"]))
            out.append(r)
    return out


rows = pd.concat([rows, pd.DataFrame(persistence_rows("three") + persistence_rows("nine")),
                  pd.DataFrame(ensemble_rows(["B", "R4a NB-GLM", "R4b k-NN"], "R5 ensemble B + NB-GLM + k-NN"))],
                 ignore_index=True)
R3 = rows[rows.origin_set == "three"]
R9 = rows[rows.origin_set == "nine"]
rows.to_json(OUT / "runs.json", orient="records", indent=1)


def diffs(df, arm, ref, metric="val_RMSE", unit="origin_seed"):
    a = df[df.name == arm].groupby(["origin", "seed"])[metric].mean()
    r = df[df.name == ref].groupby(["origin", "seed"])[metric].mean()
    if (r.index.get_level_values("seed") < 0).all():             # persistence: one row per origin
        r = pd.Series({(o, s): r.xs(o, level="origin").iloc[0] for o, s in a.index})
    d = (a - r).dropna()
    return d.groupby(level="origin").mean().to_numpy() if unit == "origin" else d.to_numpy()


def sign_flip_p(d, n_perm=100_000, seed=42):
    d = np.asarray(d, float)
    flips = (np.array(list(itertools.product((-1, 1), repeat=len(d))), float) if len(d) <= 16
             else np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_perm, len(d))))
    return float((np.abs(flips @ d / len(d)) >= abs(d.mean()) - 1e-12).mean())


def bh(p):
    p = np.asarray(p, float); m = len(p); adj = np.empty(m); run = 1.0
    for rank, i in reversed(list(enumerate(np.argsort(p), start=1))):
        run = min(run, p[i] * m / rank); adj[i] = min(run, 1.0)
    return adj


def compare(df, arm, ref, metric="val_RMSE", unit="origin_seed"):
    d = diffs(df, arm, ref, metric, unit)
    return {"delta": float(d.mean()), "wins": f"{int((d < 0).sum())}/{len(d)}", "p": sign_flip_p(d)}


def mean_of(df, name, metric="val_RMSE"):
    return float(df.loc[df.name == name, metric].mean())


pd.set_option("display.width", 170)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
RESULTS = {}           # every headline number, written to outputs/results.json at the end

# %% [markdown]
# ### 8.1 Persistence, the floor every model competes with
#
# The forecast "next three weeks = last observed week". Last week's cases explain almost
# all of this week's; the best climate variable at a causal lag explains almost nothing.

# %%
def r2_pooled(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)


RESULTS["r2_lag1"] = r2_pooled(CASES[:-1].ravel(), CASES[1:].ravel())
RESULTS["r2_best_climate"] = max(r2_pooled(CLIMATE[:-lag, :, c].ravel(), CASES[lag:].ravel())
                                 for c in range(CLIMATE.shape[-1]) for lag in range(2, 9))
RESULTS["persistence_3"] = {"val": mean_of(R3, "persistence"), "test": mean_of(R3, "persistence", "RMSE")}
RESULTS["persistence_9"] = {"val": mean_of(R9, "persistence"), "test": mean_of(R9, "persistence", "RMSE")}
RESULTS["array_floor"] = {"all_windows": float(ARRAY_FLOOR[0]), "without_row_395": float(ARRAY_FLOOR[1])}
print(f"cases(t-1) -> cases(t): r2 = {RESULTS['r2_lag1']:.3f};  best climate variable (lags 2-8): "
      f"r2 = {RESULTS['r2_best_climate']:.3f}")
print(f"persistence, three origins: val {RESULTS['persistence_3']['val']:.2f}  test {RESULTS['persistence_3']['test']:.2f}")
print(f"persistence, nine origins:  val {RESULTS['persistence_9']['val']:.2f}  test {RESULTS['persistence_9']['test']:.2f}")

# %% [markdown]
# ### 8.2 Six encoders, one harness (paper Table 2)

# %%
tab = pd.DataFrame({h: [f"{mean_of(R3, f'{e}+{h}'):.2f} / {mean_of(R3, f'{e}+{h}', 'RMSE'):.2f}" for e in ENC]
                    for h in ("direct", "foi", "foi_res")}, index=ENC)
tab.columns = ["direct", "SEIR decoder", "gated SEIR"]
print("mean validation / test RMSE; persistence "
      f"{RESULTS['persistence_3']['val']:.2f} / {RESULTS['persistence_3']['test']:.2f}")
print(tab.to_string())
enc_pairs = []
for e in ENC:
    for h in ("foi", "foi_res"):
        v, t = compare(R3, f"{e}+{h}", f"{e}+direct"), compare(R3, f"{e}+{h}", f"{e}+direct", "RMSE")
        enc_pairs.append({"encoder": e, "head": h, "val delta": v["delta"], "val wins": v["wins"], "p": v["p"],
                          "test delta": t["delta"], "test wins": t["wins"]})
ENC_PAIRS = pd.DataFrame(enc_pairs)
print("\neach SEIR head against the same encoder's direct head:")
print(ENC_PAIRS.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
x = np.arange(len(ENC))
for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
    for k, (h, lab) in enumerate((("direct", "direct"), ("foi", "SEIR decoder"), ("foi_res", "gated SEIR"))):
        ax.bar(x + (k - 1) * 0.27, [mean_of(R3, f"{e}+{h}", metric) for e in ENC], 0.27, label=lab)
    ax.axhline(mean_of(R3, "persistence", metric), color="k", ls="--", lw=1, label="persistence")
    ax.set_xticks(x, ENC); ax.set_title(title); ax.set_ylabel("RMSE")
axes[0].legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig(OUT / "fig_encoders.png", dpi=150); plt.show()

# %% [markdown]
# ### 8.3 Anchor, gate or physics? (paper Table 5)
#
# The gated SEIR head differs from the direct head in three ways: a persistence anchor, a
# learned gate, and the SEIR simulator. The pre-registered rule credits a repair to the
# physics only if gated SEIR beats its no-physics twin (`gated`) with ≥ 7/9 wins and
# BH-adjusted p < 0.05 on each failing encoder.

# %%
H4 = ["direct", "residual", "gated", "foi_res"]
print(pd.DataFrame({h: [mean_of(R3, f"{e}+{h}") for e in ENC] for h in H4}, index=ENC).to_string())
res = []
for e in ENC:
    v, t = compare(R3, f"{e}+foi_res", f"{e}+gated"), compare(R3, f"{e}+foi_res", f"{e}+gated", "RMSE")
    d, g, f = (mean_of(R3, f"{e}+{h}") for h in ("direct", "gated", "foi_res"))
    res.append({"encoder": e, "SEIR - gated (val)": v["delta"], "wins": v["wins"], "p": v["p"],
                "SEIR - gated (test)": t["delta"], "test wins": t["wins"],
                "share of repair from anchor+gate": (d - g) / (d - f) if abs(d - f) > 1e-9 else np.nan})
RESCUE = pd.DataFrame(res)
failing = RESCUE.encoder.isin(["STGAT", "A3TGCN", "DCRNN"])
RESCUE.loc[failing, "p_adj"] = bh(RESCUE.loc[failing, "p"])
print("\n" + RESCUE.to_string(index=False))
credited = [r["encoder"] for _, r in RESCUE[failing].iterrows()
            if r["SEIR - gated (val)"] < 0 and int(r["wins"].split("/")[0]) >= 7 and r["p_adj"] < 0.05]
print(f"\nencoders whose repair is credited to the SEIR simulator: {credited or 'none'}")

# %% [markdown]
# ### 8.4 Every lever, paired against its control (paper Table 3)

# %%
LEVERS = [
    ("worked", "NB likelihood (vs squared error)", "B", "B, squared error"),
    ("worked", "seasonal features", "feat=season", "graph=gcn"),
    ("worked", "best model B vs persistence", "B", "persistence"),
    ("graph/architecture", "graph (GCN vs no message passing)", "graph=gcn", "graph=none"),
    ("graph/architecture", "learned adjacency", "graph=adaptive", "graph=gcn"),
    ("graph/architecture", "district seasonal curves", "A2 district seasonal curves", "B"),
    ("graph/architecture", "nonlinear (MLP) head", "A3 MLP head", "B"),
    ("graph/architecture", "MLP head + climate lags 2-13", "A4 MLP head + climate 2-13", "B"),
    ("graph/architecture", "global context node", "A5 global context", "B"),
    ("graph/architecture", "residual head + NB", "A1 residual head", "B"),
    ("graph/architecture", "combined architecture changes", "A6 combined", "B"),
    ("literature remedy", "RevIN", "R1 RevIN", "B"),
    ("literature remedy", "district identity embedding", "R2 district embedding", "B"),
    ("literature remedy", "STID-style MLP", "R3 STID-style MLP", "B"),
    ("literature remedy", "NB-GLM (no network)", "R4a NB-GLM", "B"),
    ("literature remedy", "k-NN analogues (no network)", "R4b k-NN", "B"),
    ("literature remedy", "ensemble B + NB-GLM + k-NN", "R5 ensemble B + NB-GLM + k-NN", "B"),
    ("data", "half the training windows", "B, 50% of training", "B"),
    ("data", "TimeGAN augmentation", "G2 TimeGAN", "B"),
    ("data", "SEIR-simulated pre-training", "G3 SEIR pre-training", "B"),
    ("data", "climate, lags 2-4", "K1 climate lags 2-4", "B"),
    ("data", "climate, lags 2-13", "K2 climate lags 2-13", "B"),
    ("data", "climate, lags 2-25", "K3 climate lags 2-25", "B"),
    ("physics", "SEIR as auxiliary loss", "R6 SEIR auxiliary loss", "B"),
    ("physics", "SEIR simulator vs no-physics twin (AAGCN)", "AAGCN+foi_res", "AAGCN+gated"),
    ("physics", "spatial physics penalty", "P1 spatial penalty", "B"),
    ("physics", "metapopulation vs gated SEIR head", "P4 metapopulation SEIR", "AAGCN+foi_res, NB+season"),
    ("physics", "gated SEIR-GNN vs SEIR-LSTM", "AAGCN+foi_res, NB+season", "LSTM+foi_res, NB+season"),
]
lev = []
for group, label, arm, ref in LEVERS:
    v, t = compare(R3, arm, ref), compare(R3, arm, ref, "RMSE")
    lev.append({"group": group, "lever": label, "arm": arm, "control": ref, "val delta": v["delta"],
                "wins": v["wins"], "p": v["p"], "test delta": t["delta"]})
LEV = pd.DataFrame(lev)
LEV["p_adj"] = bh(LEV.p)
LEV["consistent gain"] = (LEV["val delta"] < 0) & (LEV.wins.str.split("/").str[0].astype(int) >= 7)
print(LEV.drop(columns=["arm", "control"]).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 7))
y = np.arange(len(LEV))[::-1]
ax.barh(y, LEV["val delta"], color=np.where(LEV["consistent gain"], "#2a9d8f",
                                             np.where(LEV["val delta"] < 0, "#9aa5b1", "#e76f51")))
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y, LEV.lever, fontsize=8)
ax.set_xlabel("validation RMSE change vs own control (negative = better)")
fig.tight_layout(); fig.savefig(OUT / "fig_levers.png", dpi=150); plt.show()

extra = {"trees gain from climate": compare(R3, "K6 trees + climate", "K7 trees"),
         "learning curve 25% vs 100%": compare(R3, "B, 25% of training", "B"),
         "learning curve 75% vs 100%": compare(R3, "B, 75% of training", "B")}
for k, v in extra.items():
    print(f"{k}: {v['delta']:+.3f} ({v['wins']}, p {v['p']:.3f})")

# %% [markdown]
# ### 8.5 The best model against persistence, and SEIR-GNN against SEIR-LSTM (paper Table 4)

# %%
cmp = []
for label, df, arm, ref, unit in [
    ("B vs persistence, 3 origins", R3, "B", "persistence", "origin_seed"),
    ("B vs persistence, 9 origins", R9, "B", "persistence", "origin"),
    ("gated SEIR-GNN (ASTGCN) vs SEIR-LSTM, 3 origins", R3, "ASTGCN+foi_res, NB+season", "LSTM+foi_res, NB+season", "origin_seed"),
    ("gated SEIR-GNN (AAGCN) vs SEIR-LSTM, 3 origins", R3, "AAGCN+foi_res, NB+season", "LSTM+foi_res, NB+season", "origin_seed"),
    ("AAGCN vs LSTM, direct head, 3 origins", R3, "B", "LSTM+direct, NB+season", "origin_seed"),
    ("gated SEIR-GNN (ASTGCN) vs SEIR-LSTM, 9 origins", R9, "ASTGCN+foi_res, NB+season", "LSTM+foi_res, NB+season", "origin"),
    ("gated SEIR-GNN (AAGCN) vs SEIR-LSTM, 9 origins", R9, "AAGCN+foi_res, NB+season", "LSTM+foi_res, NB+season", "origin"),
    ("metapopulation SEIR-GNN vs SEIR-LSTM, 9 origins", R9, "P4 metapopulation SEIR", "LSTM+foi_res, NB+season", "origin"),
    ("AAGCN vs LSTM, direct head, 9 origins", R9, "B", "LSTM+direct, NB+season", "origin"),
    ("gated SEIR-GNN (AAGCN) vs persistence, 9 origins", R9, "AAGCN+foi_res, NB+season", "persistence", "origin"),
]:
    v, t = compare(df, arm, ref, "val_RMSE", unit), compare(df, arm, ref, "RMSE", unit)
    cmp.append({"comparison": label, "unit": unit, "val delta": v["delta"], "val wins": v["wins"], "val p": v["p"],
                "test delta": t["delta"], "test wins": t["wins"], "test p": t["p"]})
CMP = pd.DataFrame(cmp)
print(CMP.to_string(index=False))

origins9 = sorted(R9.origin.unique())
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
for ax, metric, title in ((axes[0], "val_RMSE", "validation"), (axes[1], "RMSE", "test")):
    for arm in ("B", "AAGCN+foi_res, NB+season", "LSTM+foi_res, NB+season", "persistence"):
        ax.plot(origins9, [R9[(R9.name == arm) & (R9.origin == o)][metric].mean() for o in origins9],
                marker="o", ms=3, ls="--" if arm == "persistence" else "-", label=arm)
    ax.set_yscale("log"); ax.set_title(f"nine origins, {title}"); ax.set_xlabel("origin")
axes[1].legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_nine.png", dpi=150); plt.show()

# %% [markdown]
# ### 8.6 The ceiling: every family makes the same errors
#
# Residuals (forecast − truth) on the validation windows of the best configuration of
# each encoder (negative-binomial likelihood, seasonal features, direct head) and of the
# SEIR-LSTM, correlated cell by cell; and how much of each model's validation residual a
# linear model on the same inputs can still explain.

# %%
members = {"AAGCN": "B", "ASTGCN": "ASTGCN+direct, NB+season", "LSTM": "LSTM+direct, NB+season",
           "STGAT": "STGAT+direct, NB+season", "A3TGCN": "A3TGCN+direct, NB+season",
           "SEIR-LSTM": "LSTM+foi_res, NB+season", "k-NN": "R4b k-NN"}
resid = {}
for lab, name in members.items():
    resid[lab] = np.concatenate([(PREDS[("three", name, o, s)]["pred"]["val"]
                                  - PREDS[("three", name, o, s)]["truth"]["val"]).ravel()
                                 for o in FOLDS["three"] for s in SEEDS])
RESID_CORR = pd.DataFrame(np.corrcoef(np.stack(list(resid.values()))), index=list(resid), columns=list(resid))
print(RESID_CORR.round(3).to_string())
RESULTS["residual_corr_range_vs_B"] = [float(RESID_CORR["AAGCN"].drop("AAGCN").min()),
                                       float(RESID_CORR["AAGCN"].drop("AAGCN").max())]


def pooled(name, split):
    parts = [PREDS[("three", name, o, s)] for o in FOLDS["three"] for s in SEEDS]
    cat = lambda k: np.concatenate([p[k][split] for p in parts])  # noqa: E731
    return {"pred": cat("pred"), "truth": cat("truth"), "persist": cat("persist"), "idx": cat("idx")}


# How the errors look: a lag on moves, and concentration in a few district-weeks.
# A week "moves" when log growth from the last observed week exceeds 0.3 (about 35%).
ceil = []
for lab, name in list(members.items()) + [("persistence", None)]:
    v = pooled(members["AAGCN"] if name is None else name, "val")
    pred = v["persist"] if name is None else v["pred"]
    g = np.log1p(v["truth"]) - np.log1p(v["persist"])
    err, sse = pred - v["truth"], (pred - v["truth"]) ** 2
    flat = np.sort(sse.ravel())[::-1]
    ceil.append({"model": lab, "bias on rising weeks": err[g > 0.3].mean(), "bias on falling weeks": err[g < -0.3].mean(),
                 "share of squared error in the top 5% of cells": flat[: len(flat) // 20].sum() / flat.sum()})
CEIL = pd.DataFrame(ceil).set_index("model")
print("\n" + CEIL.to_string())


# Signal left on the table: fit a linear model to each model's TRAINING residuals (log space) on
# what the model was given plus its own forecast, and score it on VALIDATION.
def residual_features(part):
    c = np.log1p(np.nan_to_num(CASES))
    hist = np.stack([c[i - WINDOW: i].T for i in part["idx"]])
    nbr = np.einsum("ij,kjw->kiw", FIXED.numpy(), hist)[..., -1:]
    season = np.repeat(seasonal_features(part["idx"])[:, None, :], N, 1)
    pred = np.log1p(part["pred"])
    out = []
    for h in range(HORIZON):
        onehot = np.zeros(pred.shape[:2] + (HORIZON,))
        onehot[..., h] = 1
        out.append(np.concatenate([hist, nbr, season, pred[..., h:h + 1], onehot], -1))
    return np.stack(out, 2).reshape(-1, out[0].shape[-1])


recover = {}
for lab, name in members.items():
    if name not in CEILING:
        continue
    tr, va = pooled(name, "train"), pooled(name, "val")
    xt, xv = residual_features(tr), residual_features(va)
    yt = (np.log1p(tr["truth"]) - np.log1p(tr["pred"])).ravel()
    yv = (np.log1p(va["truth"]) - np.log1p(va["pred"])).ravel()
    beta = np.linalg.lstsq(np.c_[xt, np.ones(len(xt))], yt, rcond=None)[0]
    res_v = yv - np.c_[xv, np.ones(len(xv))] @ beta
    recover[lab] = float(1 - np.sum(res_v ** 2) / np.sum((yv - yv.mean()) ** 2))
RECOVER = pd.Series(recover, name="validation residual variance a linear model recovers")
print("\n" + RECOVER.round(3).to_string())

# TimeGAN's samples against the real training windows.
tg = R3[R3.name == "G2 TimeGAN"]
RESULTS["timegan_max"] = {"synthetic": float(tg.synthetic_max.max()), "real": float(tg.real_train_max.max())}
print(f"\nlargest weekly count: TimeGAN samples {RESULTS['timegan_max']['synthetic']:,.0f}, "
      f"real training windows {RESULTS['timegan_max']['real']:,.0f}")

# %% [markdown]
# ### 8.7 Physics as structure (paper section 5.5)

# %%
for arm, ref in (("P1 spatial penalty", "B"), ("P2 spatial penalty (log)", "B"),
                 ("P4 metapopulation SEIR", "AAGCN+foi_res, NB+season"),
                 ("P5 metapopulation + penalty + season", "AAGCN+foi_res, NB+season")):
    r = compare(R3, arm, ref)
    print(f"{arm:40s} vs {ref:28s} {r['delta']:+.3f}  {r['wins']}  p {r['p']:.4f}")

# Moran's I of log growth on the district graph: is there spatial structure for a
# coupling to learn? (log growth = log R over one week, model-free.)
lg = np.diff(np.log1p(CASES), axis=0)
w = ((FIXED > 0).numpy() & ~np.eye(N, dtype=bool)).astype(float)


def morans_i(v):
    ok = np.isfinite(v)
    if ok.sum() < N:
        return np.nan
    z = v - v.mean()
    return float(N / w.sum() * (z @ w @ z) / (z @ z)) if (z @ z) > 0 else np.nan


RESULTS["morans_i_log_growth"] = float(np.nanmean([morans_i(r) for r in lg]))
RESULTS["morans_i_log_cases"] = float(np.nanmean([morans_i(np.log1p(r)) for r in CASES]))
print(f"Moran's I on the district graph, mean over weeks: log cases {RESULTS['morans_i_log_cases']:.3f}, "
      f"weekly log growth {RESULTS['morans_i_log_growth']:.3f}")

# %% [markdown]
# ### 8.8 Validation and test disagree
#
# For each family of runs: does the configuration validation ranks first also do well
# on test?

# %%
families = {
    "encoders x heads": [f"{e}+{h}" for e in ENC for h in ("direct", "residual", "gated", "foi", "foi_res")],
    "screen": ["graph=gcn", "graph=none", "graph=adaptive", "feat=season"],
    "likelihood and heads": ["B", "B, squared error", "AAGCN+foi_res, NB+season", "ASTGCN+foi_res, NB+season",
                             "LSTM+foi_res, NB+season", "ASTGCN+direct, NB+season", "LSTM+direct, NB+season"],
    "architecture": ["B"] + [n for n in CONFIGS3 if re.match(r"A\d ", n)],
    "remedies": ["B"] + [n for n in rows.name.unique() if re.match(r"R\d", str(n))],
    "data": ["B"] + [n for n in CONFIGS3 if re.match(r"(G\d|K\d|B, )", n)],
    "physics": ["B"] + [n for n in CONFIGS3 if re.match(r"P\d", n)] + ["AAGCN+foi_res, NB+season",
                                                                     "LSTM+foi_res, NB+season"],
    "nine origins": [n for n in CONFIGS9],
}
vt = []
for fam, names in families.items():
    df = R9 if fam == "nine origins" else R3
    m = df[df.name.isin(names)].groupby("name")[["val_RMSE", "RMSE"]].mean()
    pick, best_t = m.val_RMSE.idxmin(), m.RMSE.idxmin()
    vt.append({"family": fam, "configs": len(m), "rank corr (val, test)": m.val_RMSE.rank().corr(m.RMSE.rank()),
               "picked by validation": pick, "its test": m.loc[pick, "RMSE"], "best on test": best_t,
               "best test": m.loc[best_t, "RMSE"], "persistence test": mean_of(df, "persistence", "RMSE")})
VT = pd.DataFrame(vt)
print(VT.to_string(index=False))
anchored = re.compile(r"foi_res|residual|gated|k-NN|foi_meta|metapopulation|SEIR")
print(f"\nfamilies where validation's pick is worse than persistence on test: "
      f"{int((VT['its test'] > VT['persistence test']).sum())} of {len(VT)}")
print(f"families whose best-on-test configuration is anchored to last week: "
      f"{sum(bool(anchored.search(n)) for n in VT['best on test'])} of {len(VT)}")
