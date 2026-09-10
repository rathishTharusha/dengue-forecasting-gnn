"""Cell sources for the dataset EDA notebook.

Every number in the narrative is computed by the cell above it -- nothing is
asserted from memory. Where a finding contradicts one of this project's own
documents, the notebook says so explicitly rather than quietly disagreeing.
"""

from gen_analysis import code, md

CELLS = [
    md(
        """
# EDA — the Sri Lanka dengue dataset

The array every model in this project consumes is
`notebooks/baseline/sri_lanka_2013-2022_shifted.npy`, shape
`(459 weeks, 25 districts, 11 features)`. This notebook establishes what is
actually in it, and what that implies for the ceiling on forecast accuracy.

## Why this notebook exists

`reproduction/` established that the published numbers are real: four of the five
GNNs reproduce to within 3.4%, and AAGCN to 0.002%. So the interesting question
is no longer "are the numbers right" but **"why do they stop there"**. That is an
EDA question, and this notebook answers it with five findings:

| # | Finding |
|---|---|
| F1 | The 11 channels are identified exactly, by name and lag |
| F2 | The unshifted array — thought unavailable — is reconstructible |
| F3 | 4% of weather values are `0` where they should be missing, compressing their usable range 26× |
| F4 | Week 395 is a 19× reporting artifact, not an epidemic |
| F5 | The 2017 outbreak lands in **test** for one CV fold and **train** for the other four |
| F7 | The hand-built adjacency has two one-way edges; Jaffna has degree 1 |
| F8 | Neighbours correlate at 0.62, non-neighbours at 0.55 — the graph adds only +0.07 |
| F9 | Seasonality is bimodal (2.9x), and no model is given a seasonal feature |

and one measurement that frames everything else: the covariates carry almost no
linear signal, while the target's own history carries a great deal.

## Setup
"""
    ),
    code(
        """
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path.cwd().parent.parent if (Path.cwd().parent.parent / "notebooks").exists() else Path.cwd()
NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
RESULTS = Path.cwd().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N = np.nan_to_num(np.load(NPY, allow_pickle=True)).astype(float)
DISTRICTS = sorted(json.loads(ADJ.read_text(encoding="utf-8")))
CASES_IDX = 5
cases = N[..., CASES_IDX]
T, D, F = N.shape

print(f"array {N.shape}  ->  {T} weeks x {D} districts x {F} features")
print(f"districts: {DISTRICTS[:4]} ... {DISTRICTS[-2:]}")
"""
    ),
    md(
        """
## F1 — What the 11 channels actually are

The array ships without column names. This project's `docs/DATA.md` describes
them from the paper's prose, and gets it partly wrong.

The authors' repository also contains
`Data/Datasets/sri_lanka_2013-2022_vertical.csv`, a long-form table **with named
columns**. Matching each `.npy` channel against every CSV column at every lag
identifies all eleven at 100% exact agreement.

Set `MLOS2_CSV` below to that file to re-run the match; the result is hard-coded
after it so the rest of the notebook works without the authors' repo.
"""
    ),
    code(
        '''
# The named-column source is Data/Datasets/sri_lanka_2013-2022_vertical.csv from
# github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2. It is not in this
# repo (see .gitignore: reference_repo/ belongs to its authors), so look in the
# usual places and fall back to the hard-coded map below if it is absent.
#
# To fetch it:  python reproduction/verify_local.py --env-only   then point
# MLOS2_CSV at <clone>/Data/Datasets/sri_lanka_2013-2022_vertical.csv
VERTICAL = "Data/Datasets/sri_lanka_2013-2022_vertical.csv"
CANDIDATES = [
    REPO / "reference_repo" / VERTICAL,
    REPO / "reference_repo" / "sri_lanka_2013-2022_vertical.csv",
    Path.home() / "rp" / "mlos2" / VERTICAL,
    Path("/tmp/repro/mlos2") / VERTICAL,
]
MLOS2_CSV = next((c for c in CANDIDATES if c.exists()), CANDIDATES[0])

#: Verified by exhaustive (column x lag) matching -- see the cell below.
CHANNELS = [
    ("meanTair_F_Inst", 0), ("minTair_F_Inst", 0), ("maxTair_F_Inst", 0),
    ("meanQair_F_Inst", 0), ("meanSoilmoi0_10Cm_Inst", 0),
    ("cases", 0),
    ("meanCanopint_Inst", 12), ("meanPrecipitationcal", 12),
    ("minPrecipitationcal", 12), ("maxPrecipitationcal", 12),
    ("minNdvi", 17),
]
NAMES = [c for c, _ in CHANNELS]


def load_vertical(path):
    """Long-form CSV -> (time, district, feature) array plus column names."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    cols = [c for c in rows[0] if c not in ("", "index", "time", "region")]
    regions = sorted({r["region"] for r in rows})
    times = sorted({float(r["time"]) for r in rows})
    ri = {r: i for i, r in enumerate(regions)}
    ti = {t: i for i, t in enumerate(times)}
    out = np.full((len(times), len(regions), len(cols)), np.nan)

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    for r in rows:
        out[ti[float(r["time"])], ri[r["region"]]] = [num(r[c]) for c in cols]
    return out, cols


if MLOS2_CSV.exists():
    V, VCOLS = load_vertical(MLOS2_CSV)
    print(f"vertical.csv -> {V.shape}, {len(VCOLS)} named columns")
    idx = {c: i for i, c in enumerate(VCOLS)}
    ok = 0
    for ch, (name, lag) in enumerate(CHANNELS):
        b = V[lag:lag + T, :, idx[name]]
        agree = np.isclose(np.nan_to_num(b), N[..., ch], rtol=1e-6, atol=1e-6).mean()
        ok += agree > 0.999
        print(f"  npy[{ch:2d}] = {name:24s} lag {lag:2d}   agreement {100 * agree:6.2f}%")
    print(f"\\n{ok}/{len(CHANNELS)} channels identified exactly")
else:
    V = None
    print("vertical.csv not available; using the hard-coded map above")
'''
    ),
    md(
        """
### What this corrects

The lags reproduce the paper's §IV exactly — precipitation 12 weeks, minimum
NDVI 17, mean canopy 12 — which is independent confirmation that this array is
the one the paper describes.

Two corrections to `docs/DATA.md`:

* It lists **land-surface temperature**. There is no LST channel. The three
  temperature channels are `meanTair`, `minTair`, `maxTair` — 2 m air
  temperature from GLDAS, in Kelvin.
* `meanNdvi`, `maxNdvi` and `meanPsurf` exist in the source CSV but were
  **excluded** from the array. Only `minNdvi` survives, and only at lag 17.

Also worth stating plainly: the paper's §VII refers to "20 features at each
time-step". The released array has 11, and the five GNNs are run with
`use_disease_only=True`, so they see exactly **one** — the case channel.
"""
    ),
    md(
        """
## F2 — The unshifted array is reconstructible

`reproduction/REPRODUCIBILITY_MATRIX.md` originally recorded the ten "Unshifted
Dataset" rows of Table I as not reproducible, because no unshifted `.npy` is
released. That was wrong, and this is the correction.

Every channel in the shifted array is a lag-`L` slice of a CSV column. Taking
every column at `L = 0` instead reconstructs the unshifted array. The check below
first proves the round-trip on the *shifted* array — if that is exact, the
unshifted construction follows by the same map.
"""
    ),
    code(
        """
if V is not None:
    idx = {c: i for i, c in enumerate(VCOLS)}
    shifted = np.stack([V[lag:lag + T, :, idx[c]] for c, lag in CHANNELS], axis=-1)
    exact = np.allclose(np.nan_to_num(shifted), N, rtol=1e-6, atol=1e-6)
    print("released shifted npy == nan_to_num(reconstruction):", exact)

    unshifted = np.nan_to_num(np.stack([V[0:T, :, idx[c]] for c, _ in CHANNELS], axis=-1))
    print("unshifted array:", unshifted.shape)
    print("targets identical (cases are never shifted):",
          np.allclose(unshifted[..., CASES_IDX], N[..., CASES_IDX]))
    np.save(RESULTS / "sri_lanka_2013-2022_unshifted.npy", unshifted)
    print("saved ->", RESULTS / "sri_lanka_2013-2022_unshifted.npy")
else:
    print("skipped: needs vertical.csv")
"""
    ),
    md(
        """
This is a candidate, not a confirmed match to whatever file the authors used —
they never released one. It is validated the only way available: run the five
GNNs on it and compare against Table I's unshifted rows. That is a separate job;
what this cell establishes is that the input can be built at all.
"""
    ),
    md(
        """
## F3 — 4% of the weather is `0` where it should be missing

`evaluation.py` calls `np.nan_to_num` on load, and the released array already has
zero NaNs. The source CSV does not: the GLDAS columns are ~4% missing. Those gaps
became **zeros**.

For a Kelvin temperature that is not a benign default — 0 K sits 290 K away from
every real value.
"""
    ),
    code(
        """
print(f"{'channel':24s}{'zeros %':>9s}{'non-zero min':>14s}{'max':>10s}")
for i, name in enumerate(NAMES):
    v = N[..., i]
    nz = v[v != 0]
    lo = nz.min() if nz.size else 0.0
    print(f"{name:24s}{100 * (v == 0).mean():9.1f}{lo:14.3f}{v.max():10.3f}")

t = N[..., 0]
real = t[t != 0]
span = real.max() - real.min()
print()
print("meanTair, the clearest case:")
print(f"  std with zeros    {t.std():8.2f}   -> real {span:.1f} K spread = {span / t.std():.2f} sd")
print(f"  std without zeros {real.std():8.2f}   -> real {span:.1f} K spread = {span / real.std():.2f} sd")
print(f"  usable dynamic range compressed to {100 * real.std() / t.std():.1f}% ({t.std() / real.std():.0f}x)")
"""
    ),
    code(
        """
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].hist(t.ravel(), bins=60)
axes[0].set_title("meanTair as released (note the spike at 0)")
axes[0].set_xlabel("Kelvin")
axes[1].hist(real.ravel(), bins=60, color="tab:green")
axes[1].set_title("meanTair, zeros excluded")
axes[1].set_xlabel("Kelvin")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
**Why it matters, and why it did not affect the reproduction.** The five GNNs run
`use_disease_only=True` and never read these channels, so none of the reproduced
numbers is affected. But every contribution this project has proposed —
a physics-informed loss reading temperature and precipitation, a GAN conditioned
on meteorological covariates — would read them. Under z-scoring the real
variation occupies 3.8% of the scale while an artificial 0-vs-300 split takes the
rest.

**Fix before using covariates:** mask rather than zero-fill, or interpolate the
gaps (they are scattered, not one contiguous block), and normalise over observed
values only.
"""
    ),
    md(
        """
## F4 — Week 395 is a reporting artifact

One week carries 29% of its year's cases, across 18 districts at once. Epidemics
do not do that; reporting backlogs do.
"""
    ),
    code(
        """
national = cases.sum(1)
top = np.argsort(national)[::-1][:8]
print("largest national weeks:")
for i in top:
    print(f"  week {i:3d}  {national[i]:8.0f}")

k = 395
print(f"\\nweek {k} in context: {national[k - 3:k + 4].round(0)}")
ratio = national[k] / np.median(national[k - 5:k + 6])
print(f"  {ratio:.0f}x the local median")
hit = int((cases[k] > 100).sum())
print(f"  {hit} of {D} districts simultaneously above 100 cases")
"""
    ),
    code(
        """
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(national, lw=0.9)
ax.axvline(221, color="tab:red", ls="--", lw=1, label="2017 outbreak (week 221)")
ax.axvline(395, color="tab:orange", ls="--", lw=1, label="week 395 artifact")
ax.set_xlabel("week index"); ax.set_ylabel("national weekly cases")
ax.legend(); plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
## F5 — The 2017 outbreak sits on the wrong side of one CV split

This is the finding that explains the spread in Table I.

The authors' protocol takes nested prefixes of the series — 60%, 70%, 80%, 90%,
100% — and splits each 70/30. The largest outbreak in the record peaks at week
221. Where that falls relative to each split decides whether the model is
interpolating or extrapolating.
"""
    ),
    code(
        """
W = H = 3
peak_week = int(np.argmax(national))
rows = []
print(f"{'segment':>8s}{'weeks':>7s}{'train':>12s}{'test':>12s}"
      f"{'train cases':>13s}{'test cases':>12s}{'peak in':>9s}")
for frac in (0.6, 0.7, 0.8, 0.9, 1.0):
    n_weeks = int(T * frac)
    n_seg = n_weeks - W - H
    cut = W + int(n_seg * 0.7)
    tr, te = national[:cut].sum(), national[cut:n_weeks].sum()
    where = "train" if peak_week < cut else "TEST"
    rows.append({"segment": frac, "train_end": cut, "test_end": n_weeks,
                 "train_cases": tr, "test_cases": te, "peak_in": where})
    print(f"{frac:8.1f}{n_weeks:7d}{f'0-{cut}':>12s}{f'{cut}-{n_weeks}':>12s}"
          f"{tr:13.0f}{te:12.0f}{where:>9s}")

print(f"\\noutbreak peaks at week {peak_week} ({national[peak_week]:.0f} cases nationally)")
"""
    ),
    md(
        """
Segment 0.6 is asked to predict **199,952** test cases having trained on
**133,998** — and its test window contains the largest outbreak in the record,
a regime it has never seen. Every later segment has that outbreak in training.

This is visible directly in the authors' own per-segment output. From our
reproduction of STGAT, held-out MAE per segment:

| segment | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| Test MAE | **54.23** | 16.25 | 38.78 | 41.55 | 22.99 |

Segment 0.6 is 3.3× worse than segment 0.7. And it explains ASTGCN's reported
standard deviation in Table I — ±19.85 MAE on a mean of 33.68, which is not
model noise but one fold measuring a different problem from the other four.

**No architecture fixes this.** It is distribution shift, and it is baked into
the protocol.
"""
    ),
    md(
        """
## What actually caps accuracy

Two measurements, side by side.
"""
    ),
    code(
        """
print("=== the target's own history ===")
print(f"{'lag':>5s}{'r':>9s}{'r^2':>8s}")
for lag in (1, 2, 3, 4, 8, 12, 26, 52):
    r = np.corrcoef(cases[:-lag].ravel(), cases[lag:].ravel())[0, 1]
    print(f"{lag:5d}{r:9.3f}{r ** 2:8.3f}")

per_district = [np.corrcoef(cases[:-1, i], cases[1:, i])[0, 1]
                for i in range(D) if cases[:, i].std() > 0]
print(f"\\nper-district lag-1 r: min {min(per_district):.3f}, "
      f"median {np.median(per_district):.3f}, max {max(per_district):.3f}")
"""
    ),
    code(
        """
if V is not None:
    ci = VCOLS.index("cases")
    print("=== the covariates, at their own best lag ===")
    print(f"{'covariate':24s}{'best lag':>10s}{'|r|':>8s}{'r^2':>8s}")
    best = []
    for j, name in enumerate(VCOLS):
        if name == "cases":
            continue
        rs = []
        for lag in range(0, 30):
            x = V[:V.shape[0] - lag, :, j].ravel()
            y = V[lag:, :, ci].ravel()
            m = ~np.isnan(x) & ~np.isnan(y)
            rs.append(abs(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 1000 else 0.0)
        rs = np.array(rs)
        bl = int(np.argmax(rs))
        best.append((name, bl, rs[bl]))
        print(f"{name:24s}{bl:10d}{rs[bl]:8.3f}{rs[bl] ** 2:8.3f}")
    strongest = max(best, key=lambda t: t[2])
    print(f"\\nstrongest covariate anywhere: {strongest[0]} at lag {strongest[1]}, "
          f"|r|={strongest[2]:.3f} (r^2={strongest[2] ** 2:.3f})")
else:
    print("skipped: needs vertical.csv")
"""
    ),
    md(
        """
### The comparison that matters

| Predictor | best \\|r\\| | r² |
|---|---|---|
| cases at *t−1* | **0.921** | **0.848** |
| cases at *t−3* (the paper's horizon) | 0.817 | 0.667 |
| best covariate (`meanQair`, lag 4) | 0.145 | 0.021 |

The target's own recent history explains ~85% of variance at one week and ~67%
at three. The best meteorological covariate explains **2%**.

That is the cap. Three consequences follow, and each is testable:

1. **Persistence is a strong baseline by construction**, not by accident. The
   `crosscheck/` workspace measured this: naive last-value-carried-forward beats
   every model in Table I's cross-validated column.
2. **Adding covariates cannot help much linearly.** Any gain has to come from a
   nonlinear or district-specific relationship these pooled correlations would
   miss — which is a hypothesis worth testing, not an assumption to build on.
3. **The remaining headroom is in the tails**: outbreak onset and peak timing,
   where persistence necessarily fails, and which is exactly where segment 0.6
   shows the models failing too.

A caveat on method: these are *pooled linear* correlations. A covariate could
matter nonlinearly, or in a few districts, or through an interaction, and show
|r| ≈ 0.05 here. F3 is also a confound — the weather channels are 4% zero-filled,
which attenuates any correlation they might have. Both are reasons to re-measure
on cleaned data before concluding a covariate is useless.
"""
    ),
    md("## Distribution and spatial structure"),
    code(
        """
pct = np.percentile(cases, [0, 25, 50, 75, 90, 95, 99, 100])
print("case percentiles 0/25/50/75/90/95/99/100:", np.round(pct, 1))
print(f"mean {cases.mean():.1f}  std {cases.std():.1f}  zeros {100 * (cases == 0).mean():.1f}%")
skew = ((cases - cases.mean()) ** 3).mean() / cases.std() ** 3
print(f"skewness {skew:.1f}")

total = cases.sum()
order = np.sort(cases.ravel())[::-1]
for frac in (0.01, 0.05, 0.10):
    n = int(frac * cases.size)
    print(f"  top {100 * frac:4.1f}% of district-weeks hold {100 * order[:n].sum() / total:5.1f}% of cases")
"""
    ),
    code(
        """
totals = cases.sum(0)
order = np.argsort(totals)[::-1]
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].bar(range(D), totals[order])
axes[0].set_xticks(range(D))
axes[0].set_xticklabels([DISTRICTS[i] for i in order], rotation=90, fontsize=7)
axes[0].set_title("total cases by district, 2013-2022")
axes[1].hist(np.log1p(cases.ravel()), bins=60, color="tab:purple")
axes[1].set_title("log1p(cases) — the space the baseline trains in")
plt.tight_layout(); plt.show()

print("most affected:", [DISTRICTS[i] for i in order[:5]])
print("least affected:", [DISTRICTS[i] for i in order[-3:]])
print(f"Colombo alone: {100 * totals[order[0]] / total:.1f}% of all cases")
"""
    ),
    md(
        """
## F7 — The graph: two one-way edges, and one nearly isolated district

The paper says the adjacency was "organically built independently by each member
of the team based on district distances and cross-validated" (§III). It is
hand-made, and it shows.

Adjacency is a symmetric relation: if A borders B then B borders A. The released
list breaks that twice.
"""
    ),
    code(
        """
adj = json.loads(ADJ.read_text(encoding="utf-8"))
idx = {d: i for i, d in enumerate(DISTRICTS)}
A = np.zeros((D, D))
for d, nbrs in adj.items():
    for b in nbrs:
        A[idx[d], idx[b]] = 1.0
np.fill_diagonal(A, 0.0)

print(f"directed edges: {int(A.sum())}   density {A.mean():.3f}   symmetric: {np.allclose(A, A.T)}")
oneway = [(DISTRICTS[i], DISTRICTS[j])
          for i in range(D) for j in range(D) if A[i, j] and not A[j, i]]
print(f"one-way pairs: {len(oneway)}")
for a, b in oneway:
    print(f"   {a} -> {b}, but not back")

deg = A.sum(1)
print(f"\\ndegree: min {deg.min():.0f}  median {np.median(deg):.0f}  max {deg.max():.0f}")
for i in np.argsort(deg)[:3]:
    print(f"   least connected: {DISTRICTS[i]:14s} out={int(A[i].sum())} in={int(A[:, i].sum())}")
for i in np.argsort(deg)[-2:]:
    print(f"   most connected:  {DISTRICTS[i]:14s} out={int(A[i].sum())} in={int(A[:, i].sum())}")
"""
    ),
    md(
        """
`load_adjacency_matrix` in `evaluation.py` writes `A[i][j] = 1` for each listed
neighbour without symmetrising, so both one-way entries survive into the graph
and message passing is directional there. With 116 directed edges, two
inconsistencies is a small defect — but it is a defect, and an adaptive graph
that learns its own adjacency would not inherit it.

Jaffna having degree 1 matters more: it is a high-incidence district (5th by
total cases) that the graph can barely reach.
"""
    ),
    md(
        """
## F8 — Is the graph doing any work?

The case for a GNN over 25 independent time series rests on neighbouring
districts co-varying more than distant ones. That is testable directly.
"""
    ),
    code(
        """
from scipy import stats

log_cases = np.log1p(cases)
R = np.corrcoef(log_cases.T)
neigh = np.array([R[i, j] for i in range(D) for j in range(i + 1, D) if A[i, j]])
far = np.array([R[i, j] for i in range(D) for j in range(i + 1, D) if not A[i, j]])

print(f"neighbour pairs      n={len(neigh):3d}   mean r = {neigh.mean():.3f}")
print(f"non-neighbour pairs  n={len(far):3d}   mean r = {far.mean():.3f}")
print(f"difference                        {neigh.mean() - far.mean():+.3f}")
_, pval = stats.mannwhitneyu(neigh, far, alternative="greater")
print(f"Mann-Whitney U, one-sided         p = {pval:.4f}")

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(far, bins=25, alpha=0.6, label=f"non-neighbours (mean {far.mean():.2f})", density=True)
ax.hist(neigh, bins=25, alpha=0.6, label=f"neighbours (mean {neigh.mean():.2f})", density=True)
ax.set_xlabel("correlation of log1p(cases) between district pairs")
ax.legend(); plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
### The result cuts both ways

Neighbours **are** more correlated than non-neighbours, and significantly so
(p = 0.0002). The graph is not noise.

But look at the size of it. Neighbours correlate at 0.62; non-neighbours at
**0.55**. Every district in Sri Lanka moves together — a shared national
seasonality — and geographic adjacency adds only **+0.07** on top of that.

That is a direct argument for a **learned** adjacency. A fixed binary graph
spends all its capacity on the 57 neighbour pairs and assigns zero weight to the
243 non-neighbour pairs that already correlate at 0.55. An adaptive graph can
weight pairs by how they actually co-vary rather than by whether they share a
border.

It is also a caution: if most inter-district correlation is a common seasonal
signal rather than genuine spatial spread, then *any* graph — fixed or learned —
is modelling something a per-district seasonal term could capture more cheaply.
Worth testing before attributing gains to the graph.
"""
    ),
    md(
        """
## Seasonality

Dengue in Sri Lanka tracks the monsoons, so a seasonal profile is expected. It
matters here for a specific reason: it is the most likely source of the shared
inter-district correlation in F8.
"""
    ),
    code(
        """
national = cases.sum(1)
week_of_year = np.arange(T) % 52
profile = np.array([national[week_of_year == w].mean() for w in range(52)])

print(f"peak weeks-of-year:   {np.argsort(profile)[::-1][:5]}")
print(f"trough weeks-of-year: {np.argsort(profile)[:5]}")
print(f"peak/trough ratio:    {profile.max() / profile.min():.2f}")

fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(profile, marker="o", ms=3)
ax.set_xlabel("week of year"); ax.set_ylabel("mean national weekly cases")
ax.set_title("Seasonal profile, averaged over 2013-2022")
plt.tight_layout(); plt.show()
"""
    ),
    md(
        """
The profile is **bimodal** — a large peak around weeks 11–14 and a second around
week 31 — with a ~2.9× peak-to-trough ratio. Two monsoon-driven transmission
seasons rather than one.

None of the five reproduced GNNs is given any seasonal encoding: with
`use_disease_only=True` and a 3-week window, the model sees three consecutive
case counts and nothing that tells it where in the year it is. A 3-week window
cannot represent a 52-week cycle. That is a concrete, cheap gap — a
week-of-year feature costs nothing and is not in any model here.
"""
    ),
    md("## Verdict, and what it implies for the contributions"),
    code(
        """
findings = [
    ("F1", "11 channels identified by name and lag, 100% exact",
     "docs/DATA.md lists a land-surface-temperature channel that does not exist"),
    ("F2", "unshifted array is reconstructible from vertical.csv",
     "the 10 unshifted Table I rows may be reproducible after all"),
    ("F3", "4% of weather values are 0-filled; usable range compressed 26x",
     "must be fixed before any covariate-using contribution"),
    ("F4", "week 395 is a 19x spike across 18 of 25 districts at once",
     "affects any model trained or scored across it"),
    ("F5", "2017 outbreak is in TEST for segment 0.6, TRAIN for the rest",
     "explains ASTGCN's +-19.85 MAE and the cross-segment spread"),
    ("F6", "cases r^2=0.85 at lag 1; best covariate r^2=0.02",
     "persistence is strong by construction; headroom is in the tails"),
    ("F7", "adjacency has 2 one-way edges; Jaffna has degree 1",
     "hand-built graph carries defects a learned graph would not inherit"),
    ("F8", "neighbour r=0.62 vs non-neighbour r=0.55 (p=0.0002)",
     "graph is real but weak; argues for a learned adjacency"),
    ("F9", "bimodal seasonality, 2.9x peak/trough, no seasonal feature in any model",
     "a 3-week window cannot represent a 52-week cycle"),
]
import csv as _csv
with open(RESULTS / "eda_findings.csv", "w", newline="", encoding="utf-8") as fh:
    w = _csv.writer(fh)
    w.writerow(["id", "finding", "implication"])
    w.writerows(findings)
for fid, what, why in findings:
    print(f"{fid}  {what}\\n     -> {why}")
print("\\nwrote", RESULTS / "eda_findings.csv")
"""
    ),
    md(
        """
## Where this leaves the two proposed contributions

Neither is ruled out, but both need re-aiming based on the above.

**Physics-informed SEIR–SEI loss.** F6 is the argument *for* it: if the
covariates carry no linear signal and persistence already captures the
autocorrelation, then extra accuracy has to come from structure, not from more
features. A compartmental prior supplies structure. Two constraints from this
notebook: it must not read the 0-filled weather channels until F3 is fixed, and
`crosscheck/FINDINGS.md` F4.2–F4.4 records three defects in the source paper's
parameter table that must not be inherited.

**GAN augmentation.** F5 is the argument for it: the failure is a regime the
model never saw. Synthesising outbreak-like sequences targets exactly that gap.
But F4 warns that the data contains at least one non-epidemic spike, and a
generator trained to reproduce the empirical distribution will happily learn to
generate reporting artifacts. Clean first.

**The measurement to run before either.** Segment 0.6 is the only fold that tests
extrapolation to an unseen outbreak. If a contribution helps anywhere, it should
help there first. Scoring on the pooled five-segment average — as Table I does —
would dilute exactly the effect worth detecting.
"""
    ),
]
