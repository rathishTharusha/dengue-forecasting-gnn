# %% [markdown]
# ## 2. The data, rebuilt from the sources
#
# ### 2.1 Weekly cases
#
# The Epidemiology Unit publishes a Weekly Epidemiological Report (WER); Table 1 gives
# dengue cases per Regional Director of Health Services division. The source table is
# the parsed Table 1 rows of every report (553 reports, verified against the official
# PDFs district by district). The rebuild:
#
# * orders weeks by the report's own **volume and number** (volume N = year 1973 + N),
#   not by the parsed date, which is wrong for a handful of reports;
# * maps division labels to the 25 districts of the benchmark graph, excluding the
#   national total and the Kalmunai division (inside Ampara; the benchmark's config
#   excludes it, so the Ampara target is Ampara RDHS only);
# * puts every report on a regular grid of report numbers and leaves weeks with **no
#   report missing** — never filled, because a filled value is invented data and an
#   interpolation borrows the following week;
# * applies **one correction**, read from the published PDF itself (below).

# %%
import json
import re

import numpy as np
import pandas as pd

GRAPH = json.loads((SRC / "graph" / "sri_lanka_adj_list.json").read_text(encoding="utf-8"))
NAMES = sorted(GRAPH)                       # district order used everywhere
N = len(NAMES)

#: The dump's truncated or misspelt labels that are not a prefix of their district key.
LABEL_ALIASES = {"paha": "Gampaha", "monaragala": "Moneragala", "nuwara": "NuwaraEliya"}
#: The national total row and the Kalmunai division, excluded as the benchmark excludes them.
EXCLUDED_LABELS = ("srilanka", "94srilanka", "kalmune", "kalmunei", "kalmunai")


def canonical(label: str) -> str | None:
    key = str(label).lower().replace("-", "").replace(" ", "")
    if key in EXCLUDED_LABELS:
        return None
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]
    for n in NAMES:
        nk = n.lower().replace("-", "")
        if nk.startswith(key[:6]) or key.startswith(nk[:6]):
            return n
    return None


raw = pd.read_csv(SRC / "cases" / "output_Dengue Fever.csv", encoding="utf-8", encoding_errors="replace")
vol = raw["Source File"].astype(str).str.extract(r"(?i)vol[_ ]?(\d+)[_ ]?no[_ ]?(\d+)")
raw["year"] = vol[0].astype(int) + 1973
raw["week_no"] = vol[1].astype(int)
raw["parsed_date"] = pd.to_datetime(raw["TimeStampStart"], format="%d-%b-%Y %H:%M:%S",
                                    errors="coerce").dt.normalize()
raw["cases"] = pd.to_numeric(raw["Cases"], errors="coerce")
label = raw["Location Name"].astype(str).str.lower().str.replace("-", "").str.replace(" ", "")
national = raw[label.isin(["srilanka", "94srilanka"])].groupby(["year", "week_no"])["cases"].max()
kalmunai = raw[label.isin(["kalmune", "kalmunei", "kalmunai"])].groupby(["year", "week_no"])["cases"].max()
raw["district"] = raw["Location Name"].map(canonical)
dist = raw.dropna(subset=["district"])
assert (dist.groupby(["year", "week_no", "district"])["cases"].nunique() <= 1).all(), \
    "a report-district cell disagrees across source files"
table = dist.groupby(["year", "week_no", "district"]).agg(
    cases=("cases", "first"), parsed_date=("parsed_date", "min")).reset_index()
print(f"source table: {len(raw):,} rows -> {table[['year', 'week_no']].drop_duplicates().shape[0]} reports "
      f"x {table.district.nunique()} districts")

# %% [markdown]
# **The week-395 "spike" is a spreadsheet error, corrected from the same report.** In
# WER Vol 48 No 02 (26 Dec 2020 – 1 Jan 2021), Table 1's dengue row **A** (cases this
# week) has most cells equal to the sum of the two before them — a formula dragged
# across the row — and its districts sum to thousands against a printed national
# total of 35. Row **B** (cumulative for the year) is internally consistent, and in the
# first week of the year cumulative *is* weekly. The cell below reads both rows from
# the PDF and replaces row A with row B for that one report.

# %%
import pymupdf

WER_COLUMNS = ["Colombo", "Gampaha", "Kalutara", "Kandy", "Matale", "NuwaraEliya", "Galle",
               "Hambantota", "Matara", "Jaffna", "Kilinochchi", "Mannar", "Vavuniya",
               "Mullaitivu", "Batticaloa", "Ampara", "Trincomalee", "Kurunegala", "Puttalam",
               "Anuradhapura", "Polonnaruwa", "Badulla", "Moneragala", "Ratnapura", "Kegalle",
               "Kalmunai", "SRILANKA"]

page = pymupdf.open(SRC / "cases" / "vol_48_no_02-english_1.pdf")[2]
grid = page.find_tables().tables[0].extract()
header = next(r for r in grid if r and r[0] == "RDHS")
i = next(k for k, r in enumerate(grid) if r and r[0] and str(r[0]).startswith("Dengue"))
assert str(grid[i][1]) == "B" and str(grid[i + 1][1]) == "A", "unexpected row layout in the Dengue block"
assert len(header[2:]) == len(WER_COLUMNS)
row_b = [int(x) for x in grid[i][2:]]
row_a = [int(x) for x in grid[i + 1][2:]]
formula_cells = sum(1 for k in range(2, 26) if abs(row_a[k] - (row_a[k - 1] + row_a[k - 2])) <= 1)
assert sum(row_b[:25]) + row_b[25] == row_b[26], "row B does not add up; refusing to use it"
print(f"row A (printed 'cases this week'): districts sum {sum(row_a[:25]):,}, national cell {row_a[26]}, "
      f"{formula_cells}/24 cells = sum of the two before")
print(f"row B (year to date = week 1):     districts + Kalmunai = {sum(row_b[:25]) + row_b[25]} "
      f"= national {row_b[26]}")
CORRECTION = {(2021, 2): dict(zip(WER_COLUMNS[:25], row_b[:25]))}

# %%
def report_dates(tab: pd.DataFrame) -> pd.Series:
    """One start date per report; a parsed date whose year disagrees with the volume is re-derived."""
    per = tab.groupby(["year", "week_no"])["parsed_date"].min()
    ok = (per.dt.year == per.index.get_level_values("year")) | (
        (per.dt.month == 12) & (per.index.get_level_values("year") == per.dt.year + 1))
    good, keys, fixed = per[ok], list(per.index), per.copy()
    for i, k in enumerate(keys):
        if ok.loc[k]:
            continue
        for step in range(1, len(keys)):
            hit = next((j for j in (i - step, i + step) if 0 <= j < len(keys) and keys[j] in good.index), None)
            if hit is not None:
                fixed.loc[k] = good.loc[keys[hit]] + pd.Timedelta(weeks=i - hit)
                break
    return fixed


tab = table.copy()
for (y, n), repl in CORRECTION.items():
    mask = (tab.year == y) & (tab.week_no == n)
    for d, v in repl.items():
        tab.loc[mask & (tab.district == d), "cases"] = v
last_no = tab.groupby("year")["week_no"].max()
first_year, last_year = int(tab.year.min()), int(tab.year.max())
first_no = int(tab.loc[tab.year == first_year, "week_no"].min())
week_grid = []
for y in range(first_year, last_year + 1):
    n_weeks = max(52, int(last_no.get(y, 52)))
    start = first_no if y == first_year else 1
    stop = int(last_no[y]) if y == last_year else n_weeks
    week_grid.extend((y, n) for n in range(start, stop + 1))
wide = (tab.pivot_table(index=["year", "week_no"], columns="district", values="cases", aggfunc="first")
        .reindex(pd.MultiIndex.from_tuples(week_grid, names=["year", "week_no"])).reindex(columns=NAMES))
observed = wide.notna().all(axis=1).to_numpy()
dates = report_dates(tab).reindex(wide.index)
if dates.isna().any():
    dates = dates.interpolate()          # dates only: the case values of a missing week stay missing
if dates.isna().any():                   # a missing week at either end: step from the nearest known date
    known = dates.dropna()
    pos = {k: i for i, k in enumerate(wide.index)}
    for k in dates[dates.isna()].index:
        j = min(known.index, key=lambda kk: abs(pos[kk] - pos[k]))
        dates.loc[k] = known.loc[j] + pd.Timedelta(weeks=pos[k] - pos[j])
CASES = wide.to_numpy(dtype=np.float64)
MISSING = ~observed
WEEK_START = pd.Series(pd.to_datetime(dates.to_numpy()))
YEARS = np.array([k[0] for k in wide.index])
T = len(CASES)

check = wide.sum(axis=1).to_frame("districts").join(
    pd.DataFrame({"national": national, "kalmunai": kalmunai}))
check = check[observed & check.national.notna().to_numpy()]
gap = (check.districts + check.kalmunai.fillna(0) - check.national).abs()
print(f"rebuilt series: {T} weeks, {week_grid[0][0]}-W{week_grid[0][1]} .. {week_grid[-1][0]}-W{week_grid[-1][1]}, "
      f"{MISSING.sum()} weeks with no report (left missing)")
print(f"integrity: districts + Kalmunai equal the Unit's own national total in "
      f"{int((gap == 0).sum())} of {len(check)} reports")

# %% [markdown]
# ### 2.2 Weekly climate (ERA5)
#
# Six daily ERA5 variables per district, from the Open-Meteo archive at one interior
# point per district (the centroid if it lies inside the district's GADM polygon,
# otherwise a representative point). Each report week takes the mean of its 7 days,
# precipitation the 7-day total. **No lag is baked in**: the model reads climate only
# up to two weeks before the forecast week (ERA5 is published about 5 days late), and
# that rule is applied when windows are cut, not here.

# %%
CLIMATE_VARS = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
                "precipitation_sum", "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean"]
CLIMATE = np.full((T, N, len(CLIMATE_VARS)), np.nan)
for j, name in enumerate(NAMES):
    daily = pd.DataFrame(json.loads((SRC / "climate" / f"{name}.json").read_text(encoding="utf-8"))["daily"])
    daily["time"] = pd.to_datetime(daily["time"])
    daily = daily.set_index("time")
    for t, ws in enumerate(WEEK_START):
        span = daily.loc[ws: ws + pd.Timedelta(days=6)]
        for c, v in enumerate(CLIMATE_VARS):
            CLIMATE[t, j, c] = span[v].sum() if v == "precipitation_sum" else span[v].mean()
assert not np.isnan(CLIMATE).any(), "ERA5 does not cover every week and district"
print(f"climate: {CLIMATE.shape} (weeks, districts, variables) from {N} daily ERA5 series")

# %% [markdown]
# ### 2.3 Population
#
# The SEIR model needs each district's population. A mid-year estimate for year Y is
# published after year Y, so each week in year Y uses the **latest official figure for
# an earlier year**: the 2012 Census count for 2013-2014, and the DCS mid-year estimate
# for Y − 1 from 2015 on. Both are read from the official PDFs. Each Census total must
# equal male + female on the same line and the 25 must sum to the national census total.

# %%
DCS_TO_GRAPH = {"Nuwara-eliya": "NuwaraEliya", "Monaragala": "Moneragala"}
rows = []
for pg in pymupdf.open(SRC / "population" / "Mid-year_population_by_district_and_sex_2024.pdf"):
    for tbl in pg.find_tables().tables:
        g = tbl.extract()
        years = [(k, str(c)) for k, c in enumerate(g[0]) if c and str(c)[:4].isdigit()]
        for line in g[2:]:
            nm = (line[0] or "").strip()
            for col, lab in years:
                if nm and line[col] not in (None, ""):
                    rows.append({"district": DCS_TO_GRAPH.get(nm, nm), "year": int(lab[:4]),
                                 "thousands": int(str(line[col]).replace(",", ""))})
midyear = pd.DataFrame(rows).drop_duplicates(["district", "year"])
for y, g in midyear.groupby("year"):
    nat = int(g.loc[g.district == "Sri Lanka", "thousands"].iloc[0])
    assert abs(nat - int(g.loc[g.district != "Sri Lanka", "thousands"].sum())) <= 13, y
midyear = midyear[midyear.district != "Sri Lanka"].pivot(index="year", columns="district",
                                                         values="thousands")[NAMES] * 1000

DCS_SPELLING = {"Batticaloa": "Baticaloa", "Moneragala": "Monaragala"}
SEPARATE_A1 = {"Anuradhapura", "Moneragala", "Polonnaruwa"}
census = {}
for name in NAMES:
    sp = DCS_SPELLING.get(name, name)
    pdf = SRC / "population" / "census2012" / (f"{sp}_A1.pdf" if name in SEPARATE_A1 else f"{sp}.pdf")
    found = None
    for pg in pymupdf.open(pdf):
        text = pg.get_text().replace("\xa0", " ")
        for lab in ("District", re.escape(name), re.escape(sp)):
            for m in re.finditer(lab, text):
                nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text[m.end(): m.end() + 400])]
                if len(nums) >= 3 and nums[0] > 10_000 and nums[0] == nums[1] + nums[2]:
                    found = nums[0]
                    break
            if found:
                break
        if found:
            break
    assert found, f"{name}: no total = male + female line in {pdf.name}"
    census[name] = found
NATIONAL_CENSUS_2012 = 20_359_439   # DCS, Census of Population and Housing 2012 - Key Findings, p. 19
assert sum(census.values()) == NATIONAL_CENSUS_2012, "district census totals do not sum to the national total"

pop_rows = []
for y in YEARS:
    earlier = [yy for yy in midyear.index if yy < y]
    pop_rows.append(midyear.loc[max(earlier)].to_numpy(float) if earlier
                    else np.array([census[n] for n in NAMES], float))
POPULATION = np.stack(pop_rows)
print(f"population: census 2012 sums to {sum(census.values()):,}; mid-year estimates "
      f"{midyear.index.min()}-{midyear.index.max()}; each week uses an earlier year's figure")

# %% [markdown]
# ### 2.4 The no-future-information rule, and the evaluation protocol
#
# A forecast is made at the start of week *i* for weeks *i..i+2*. It may read cases up
# to week *i−1* and climate up to week *i−2*; population is already an earlier year's
# figure. **Folds** follow the frozen protocol: rolling origins at 0.55, 0.70 and 0.85 of
# the window list (and nine disjoint origins 0.40 + k/15 for confirmation), the 30
# windows before each origin for validation, the next 15% (or 1/15) for test, and
# normalisation statistics from training weeks only. Windows touching a missing week
# are dropped; fold boundaries are computed before dropping, so a gap never shifts them.

# %%
import torch

LAGS = {"cases": 1, "climate": 2}
WINDOW, HORIZON = 3, 3
ORIGINS = (0.55, 0.70, 0.85)
TEST_FRAC = 0.15
ORIGINS_9 = tuple(round(0.40 + k / 15.0, 4) for k in range(9))
TEST_FRAC_9 = 1.0 / 15.0

from dataclasses import dataclass, replace as dc_replace


@dataclass
class Fold:
    origin: float
    idx: dict
    mean: float
    std: float
    window: int = WINDOW


def build_folds(origins=ORIGINS, test_frac=TEST_FRAC, window=WINDOW) -> list[Fold]:
    bad = set(np.where(MISSING)[0].tolist())
    ids = list(range(window, T - HORIZON))
    clean = lambda i: not any(t in bad for t in range(i - window, i + HORIZON))  # noqa: E731
    folds = []
    for o in origins:
        cut, end = int(o * len(ids)), int(min(o + test_frac, 1.0) * len(ids))
        tr = [i for i in ids[: cut - 30] if clean(i)]
        va = [i for i in ids[cut - 30: cut] if clean(i)]
        te = [i for i in ids[cut:end] if clean(i)]
        hist = np.log1p(CASES[: ids[: cut - 30][-1] + 1])
        folds.append(Fold(o, {"train": np.array(tr), "val": np.array(va), "test": np.array(te)},
                          float(np.nanmean(hist)), float(np.nanstd(hist) + 1e-8), window))
    return folds


def seasonal_features(idx: np.ndarray) -> np.ndarray:
    """sin/cos of day-of-year at one and two cycles per year, (K, 4)."""
    ang = 2 * np.pi * WEEK_START.dt.dayofyear.to_numpy()[idx] / 365.25
    return np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], -1)


def with_history(fold: Fold, weeks: int) -> Fold:
    return dc_replace(fold, idx={k: v[v - weeks >= 0] for k, v in fold.idx.items()})


def climate_blocks(idx: np.ndarray, blocks, base: np.ndarray) -> np.ndarray:
    """Means of the climate over lag blocks (inclusive week ranges before the origin)."""
    if len(idx) and int(np.min(idx)) - max(b for _, b in blocks) < 0:
        raise ValueError("a window has no climate that far back; drop it with with_history()")
    out = []
    for a, b in blocks:
        assert a >= LAGS["climate"], "climate closer than the ERA5 release delay is not available"
        out.append(np.stack([base[idx - lag] for lag in range(a, b + 1)], 0).mean(0))
    return np.concatenate(out, -1)


def build_tensors(fold: Fold, split: str, use_climate=False, use_season=False, clim_blocks=()) -> dict:
    idx = fold.idx[split]
    x_raw = np.stack([CASES[i - fold.window: i].T for i in idx])
    y_raw = np.stack([CASES[i: i + HORIZON].T for i in idx])
    p_raw = np.repeat(CASES[idx - 1][:, :, None], HORIZON, axis=2)
    z = lambda a: (np.log1p(a) - fold.mean) / fold.std  # noqa: E731
    feats = [z(x_raw)]
    if use_climate:
        lo = LAGS["climate"]
        cut = lambda ids: np.stack([np.moveaxis(CLIMATE[i - lo - 2: i - lo + 1], 0, 1) for i in ids])  # noqa: E731
        cl, ref = cut(idx), cut(fold.idx["train"])
        m, s = ref.mean((0, 1, 2)), ref.std((0, 1, 2)) + 1e-8
        feats.append(((cl - m) / s).reshape(len(idx), N, -1))
    if clim_blocks:
        blk, ref = climate_blocks(idx, clim_blocks, CLIMATE), climate_blocks(fold.idx["train"], clim_blocks, CLIMATE)
        feats.append((blk - ref.mean((0, 1))) / (ref.std((0, 1)) + 1e-8))
    if use_season:
        feats.append(np.repeat(seasonal_features(idx)[:, None, :], N, axis=1))
    t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    return {"x": t(np.concatenate(feats, -1)), "y_raw": t(y_raw), "p_raw": t(p_raw), "y_z": t(z(y_raw)),
            "p_z": t(z(p_raw)), "x_raw": t(x_raw), "pop": t(POPULATION[idx - 1]), "idx": idx}


def rmse(pred, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(truth)) ** 2)))


def score(pred, truth) -> dict:
    out = {"RMSE": rmse(pred, truth), "MAE": float(np.mean(np.abs(pred - truth)))}
    for h in range(truth.shape[-1]):
        out[f"RMSE_h{h + 1}"] = rmse(pred[..., h], truth[..., h])
    return out


# The district graph: the benchmark's adjacency list with self-loops. Edges are listed
# row-major (this order matters to DCRNN, section 4).
_a = np.eye(N, dtype=np.float32)
for d, nbrs in GRAPH.items():
    for o in nbrs:
        if o in NAMES:
            _a[NAMES.index(d), NAMES.index(o)] = 1.0
_src, _dst = np.nonzero(_a)
EDGE_INDEX = torch.tensor(np.stack([_src, _dst]), dtype=torch.long)
FIXED = torch.tensor(_a / _a.sum(1, keepdims=True), dtype=torch.float32)   # row-normalised
print(f"graph: {N} districts, {int(_a.sum()) - N} directed edges plus self-loops, "
      f"{int((_a != _a.T).sum())} one-way entries")

for f in build_folds():
    print(f"origin {f.origin}: train {len(f.idx['train'])}, val {len(f.idx['val'])}, test {len(f.idx['test'])} "
          f"windows; test weeks {WEEK_START[f.idx['test'][0]]:%Y-%m-%d} .. {WEEK_START[f.idx['test'][-1]]:%Y-%m-%d}")
