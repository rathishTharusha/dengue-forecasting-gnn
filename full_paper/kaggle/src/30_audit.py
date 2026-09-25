# %% [markdown]
# ## 3. Why not the benchmark array?
#
# Prior work, including the benchmark and our own first phase, used the processed array
# `sri_lanka_2013-2022_shifted.npy` from the benchmark authors' repository: 459 rows
# (weeks) × 25 districts × 11 channels (5 weather channels at lag 0, the cases, and 5
# channels documented as lagged 12 or 17 weeks). The array has no date axis. This
# section dates it and measures what that shows.

# %%
ARR = np.load(SRC / "benchmark" / "sri_lanka_2013-2022_shifted.npy", allow_pickle=True).astype(float)
arr_cases = np.nan_to_num(ARR[..., 5])
print(f"benchmark array: {ARR.shape} (rows, districts, channels)")

# Date every array row by matching its 25 district counts exactly against the report
# weeks. The array was built from the published row A, so the comparison uses the
# report values before our one correction.
published = (table.pivot_table(index=["year", "week_no"], columns="district", values="cases", aggfunc="first")
             .reindex(pd.MultiIndex.from_tuples(week_grid, names=["year", "week_no"])).reindex(columns=NAMES)
             .to_numpy(float))
lookup: dict[bytes, list[int]] = {}
for t in range(T):
    if not np.isnan(published[t]).any():
        lookup.setdefault(published[t].tobytes(), []).append(t)
row_week = np.array([lookup.get(arr_cases[k].tobytes(), [-1])[0] for k in range(len(ARR))])
matched = row_week >= 0
row_date = pd.Series([WEEK_START[t] if t >= 0 else pd.NaT for t in row_week])
row_year = row_date.dt.year

dups = [k for k in range(len(ARR) - 1) if arr_cases[k].sum() > 0 and np.array_equal(arr_cases[k], arr_cases[k + 1])]
print(f"rows matched exactly to a published report: {matched.sum()} of {len(ARR)}; unmatched rows: "
      f"{np.flatnonzero(~matched).tolist()}")
print(f"identical consecutive rows (one week stored twice): {dups}")
print(f"rows 0-47 are dated {row_date[:48].min():%Y-%m-%d} .. {row_date[:48].max():%Y-%m-%d}; "
      f"rows 51 onward run {row_date[51:].min():%Y-%m-%d} .. {row_date[51:].max():%Y-%m-%d}")
span = (WEEK_START >= row_date[51:].min()) & (WEEK_START <= row_date[51:].max()) & pd.Series(~MISSING)
used = set(row_week[matched].tolist())
print(f"reports inside the array's 2013-2022 span that never reached it: "
      f"{sum(1 for t in np.flatnonzero(span.to_numpy()) if t not in used)}")

# The benchmark protocol's folds on the array (rolling origins over its row order).
ids = list(range(3, len(ARR) - 3))
for o in ORIGINS:
    cut, end = int(o * len(ids)), int(min(o + 0.15, 1.0) * len(ids))
    tr, te = ids[: cut - 30], ids[cut:end]
    print(f"  benchmark fold {o}: {sum(1 for r in tr if row_year[r] == 2023)} rows from 2023 in TRAINING; "
          f"test rows dated {row_date[te].min():%Y-%m} .. {row_date[te].max():%Y-%m}")

# %% [markdown]
# **The climate columns are on a different timeline, and the "lagged" ones point into
# the future.** The weather channels were not reordered with the cases, so they are
# dated separately: the cell searches for the calendar offset at which the array's
# temperature best matches ERA5 temperature. It then tests the precipitation channel
# documented as "lag 12" — which should hold rain from 12 weeks *earlier* — at every
# shift from −20 to +20 weeks.

# %%
era_daily = {}
for name in NAMES:
    d = pd.DataFrame(json.loads((SRC / "climate" / f"{name}.json").read_text(encoding="utf-8"))["daily"])
    era_daily[name] = d.assign(time=pd.to_datetime(d["time"])).set_index("time")
no_jaffna = [j for j, n in enumerate(NAMES) if n != "Jaffna"]     # the array's Jaffna weather is all zero


def era_weekly(var: str, starts: pd.DatetimeIndex, total: bool = False) -> np.ndarray:
    out = np.full((len(starts), N), np.nan)
    for j, name in enumerate(NAMES):
        s = era_daily[name][var]
        roll = s.rolling(7).sum() if total else s.rolling(7).mean()     # value at day t covers t-6..t
        out[:, j] = roll.reindex(starts + pd.Timedelta(days=6)).to_numpy()
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


arr_temp = ARR[:, no_jaffna, 0].mean(1)
best = None
for off in range(-120, 121):          # candidate first-row dates around the dating implied by the reports
    d0 = pd.Timestamp("2013-05-17") + pd.Timedelta(days=off)
    starts = pd.DatetimeIndex([d0 + pd.Timedelta(weeks=k) for k in range(len(ARR))])
    r = corr(arr_temp, era_weekly("temperature_2m_mean", starts)[:, no_jaffna].mean(1))
    if best is None or r > best[0]:
        best = (r, d0)
r_best, D0 = best
clim_starts = pd.DatetimeIndex([D0 + pd.Timedelta(weeks=k) for k in range(len(ARR))])
case_starts = pd.DatetimeIndex([row_date[k] if matched[k] else pd.NaT for k in range(len(ARR))])
r_case = corr(arr_temp[matched], era_weekly("temperature_2m_mean",
                                           case_starts[matched])[:, no_jaffna].mean(1))
print(f"array temperature vs ERA5: best at climate row 0 = {D0:%Y-%m-%d} (r = {r_best:.3f}); "
      f"on the dates the CASE rows carry, r = {r_case:.3f}")

prec = ARR[:, no_jaffna, 7].ravel()
shifts = list(range(-20, 21))
r_shift = [corr(prec, era_weekly("precipitation_sum", clim_starts + pd.Timedelta(weeks=s), total=True)
                [:, no_jaffna].ravel()) for s in shifts]
peak = shifts[int(np.nanargmax(r_shift))]
print(f"'lag 12' precipitation channel vs ERA5 rain {'later' if peak > 0 else 'earlier'} by k weeks: "
      f"best at k = {peak:+d} (r = {max(r_shift):.3f}); at the documented k = -12, r = {r_shift[shifts.index(-12)]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].plot(case_starts, arr_temp, ".", ms=3, label="array temperature, at the case rows' dates")
axes[0].plot(clim_starts, arr_temp, "-", lw=0.8, label="array temperature, re-dated")
axes[0].legend(fontsize=7); axes[0].set_title("the climate columns were not reordered with the cases")
axes[1].bar(shifts, r_shift, color=["#e76f51" if s > 0 else "#9aa5b1" for s in shifts])
axes[1].axvline(-12, color="k", ls="--", lw=1)
axes[1].set_xlabel("ERA5 rainfall shifted by k weeks (negative = earlier, as documented)")
axes[1].set_ylabel("correlation"); axes[1].set_title("the 'lag 12' rain channel holds FUTURE rain")
fig.tight_layout(); fig.savefig(OUT / "audit.png", dpi=150); plt.show()

# %% [markdown]
# **The largest spike is the spreadsheet error.** Row 395 of the array carries the
# printed row A of WER Vol 48 No 02, not the true count.

# %%
k395 = int(np.flatnonzero(row_week == int(np.flatnonzero(
    (np.array([k[0] for k in wide.index]) == 2021) & (np.array([k[1] for k in wide.index]) == 2))[0]))[0])
print(f"array row {k395}: districts sum {arr_cases[k395].sum():,.0f} "
      f"(printed row A {sum(row_a[:25]):,}); the corrected week sums to {sum(row_b[:25]):,}")

# The persistence floor on the array, under the benchmark protocol, with and without
# the windows that touch the spreadsheet error.
floors = []
for o in ORIGINS:
    cut, end = int(o * len(ids)), int(min(o + 0.15, 1.0) * len(ids))
    te = ids[cut:end]
    err = np.stack([np.repeat(arr_cases[i - 1][:, None], 3, 1) - arr_cases[i:i + 3].T for i in te])
    touch = np.array([k395 in range(i - 3, i + 3) for i in te])
    floors.append((rmse(err, 0), rmse(err[~touch], 0)))
ARRAY_FLOOR = np.mean(floors, axis=0)
print(f"persistence on the array (mean over the 3 origins): {ARRAY_FLOOR[0]:.2f} over all test windows, "
      f"{ARRAY_FLOOR[1]:.2f} without the windows touching row {k395}")
