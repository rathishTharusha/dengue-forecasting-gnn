"""Audit the time axis of the processed array, and draw what the audit finds.

The processed array ``sri_lanka_2013-2022_shifted.npy`` has no date axis, and
``docs/DATA.md`` describes it as "459 consecutive weeks spanning 2013-2022".
``calendar_index.csv`` (from ``build_seir_inputs.py calendar``) dates every row
by matching its 25-district case vector to a named Epidemiology Unit report.
This script re-tests that result with checks that do not reuse the matching:

A  duplicate and unmatched rows
B  yearly sums against officially published national totals
C  whether the case series jumps at the 2023 -> 2013 splice
D  whether the climate channels follow the same timeline as the cases
   (annual-cycle fit in row order vs case calendar, and where the annual
   temperature peaks land in the cases' calendar)
E  source reports that exist but never reached the array
F  where the misplaced rows sit in the frozen evaluation protocol

Nothing here modifies the array.

Run::

    python analysis/_build/audit_array_order.py \
        --raw "datasets/output_Dengue Fever.csv"

Writes ``analysis/results/array_audit/audit.json`` and five figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
NPY = REPO / "notebooks" / "baseline" / "sri_lanka_2013-2022_shifted.npy"
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
CAL = REPO / "data" / "external" / "calendar_index.csv"
OUT = REPO / "analysis" / "results" / "array_audit"

#: Official national totals: 2017 from Emerg Infect Dis 2020 (doi:10.3201/eid2604.190435);
#: 2018-2019 from PLoS NTD 2021 (doi:10.1371/journal.pntd.0009624).
OFFICIAL_TOTALS = {2017: 186101, 2018: 51659, 2019: 105049}

# Reference palette (dataviz skill, light mode), validated with validate_palette.js.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#d9d8d3"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
        "lines.linewidth": 2.0,
    })


def harmonic_r2(y: np.ndarray, t_weeks: np.ndarray) -> float:
    """R^2 of a two-harmonic annual cycle fitted against a time axis in weeks."""
    w = 2 * np.pi * t_weeks / 52.1775
    x = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(1 - (y - x @ beta).var() / y.var())


def run_checks(raw_csv: Path) -> tuple[dict, dict]:
    arr = np.load(NPY, allow_pickle=True).astype(float)
    names = sorted(json.loads(ADJ.read_text(encoding="utf-8")))
    cal = pd.read_csv(CAL, parse_dates=["report_week_start"])
    cases = np.nan_to_num(arr[..., 5])
    exact = cal["abs_diff"].eq(0).to_numpy()
    no_jaffna = [i for i, n in enumerate(names) if n != "Jaffna"]  # GLDAS all-zero there

    out: dict = {}

    # A -- duplicates and unmatched rows
    identical = [k for k in range(len(cases) - 1)
                 if cases[k].sum() > 0 and np.array_equal(cases[k], cases[k + 1])]
    out["A_unmatched_rows"] = cal.loc[~cal.abs_diff.eq(0), "array_week"].tolist()
    out["A_identical_consecutive_rows"] = [[k, k + 1] for k in identical]

    # B -- official totals
    b = {}
    for year, total in OFFICIAL_TOTALS.items():
        rows = cal.loc[cal.report_year.eq(year) & cal.abs_diff.eq(0), "array_week"].to_numpy()
        b[year] = {"weeks_in_array": int(len(rows)), "array_sum": float(cases[rows].sum()),
                   "official": total, "share": float(cases[rows].sum() / total)}
    out["B_official_totals"] = b

    # C -- splice
    change = np.abs(np.diff(np.log1p(cases), axis=0)).mean(axis=1)
    out["C_splice_change_row47_48"] = float(change[47])
    out["C_splice_percentile"] = float((change < change[47]).mean() * 100)
    out["C_largest_change_inside_2023"] = float(change[:47].max())

    # D -- climate timeline
    row_t = np.arange(len(cases), dtype=float)
    cal_t = ((cal["report_week_start"] - pd.Timestamp("2013-01-01")).dt.days / 7).to_numpy()
    channels = {0: "meanTair", 1: "minTair", 3: "meanQair", 4: "meanSoilmoi",
                7: "meanPrecip (lag 12)", 10: "minNdvi (lag 17)"}
    d = {}
    for c, label in channels.items():
        y = arr[:, no_jaffna, c].mean(axis=1)
        m = exact & np.isfinite(y)
        d[label] = {"r2_row_order": harmonic_r2(y[m], row_t[m]),
                    "r2_case_calendar": harmonic_r2(y[m], cal_t[m])}
    out["D_annual_cycle_r2"] = d

    temp_c = arr[:, no_jaffna, 0].mean(axis=1) - 273.15
    smooth = pd.Series(temp_c).rolling(7, center=True, min_periods=1).mean().to_numpy()
    peaks, _ = find_peaks(smooth, distance=40, prominence=0.4)
    exact_rows = np.flatnonzero(exact)

    def case_date_at(k: int) -> tuple[pd.Timestamp, int]:
        """Date of the cases in row k, taken from the nearest exactly-matched row.

        A row with no exact report match only has a nearest-report guess for a date,
        so an unmatched peak row is dated from its closest exact neighbour instead,
        stepped by one week per row.
        """
        j = int(exact_rows[np.argmin(np.abs(exact_rows - k))])
        return cal.loc[j, "report_week_start"] + pd.Timedelta(weeks=int(k - j)), j

    peak_rows = []
    for k in peaks:
        date, j = case_date_at(int(k))
        peak_rows.append({"row": int(k), "case_date": date.date().isoformat(),
                          "day_of_year": int(date.dayofyear), "dated_from_row": j})
    out["D_temperature_peaks"] = peak_rows
    out["D_rows_between_peaks"] = np.diff(peaks).tolist()
    body_doy = [p["day_of_year"] for p in peak_rows if p["row"] >= 51]
    out["D_peak_day_of_year_sd"] = float(np.std(body_doy))

    # Drift: how far the cases run ahead of the weather, measured two independent
    # ways. (1) Report calendar only: case weeks elapsed minus rows elapsed.
    # (2) Climate only: where each annual temperature peak lands in case time. A
    # peak's day-of-year gives drift only modulo a year, so it is unwrapped under
    # one assumption -- dropped weeks can only push case time forward -- with 2.5
    # weeks of slack for peak-detection noise. Both are relative to the first peak
    # row; the absolute offset at that row is not measurable this way.
    body_peaks = [p for p in peak_rows if p["row"] >= 51]
    ref_row, ref_date = body_peaks[0]["row"], pd.Timestamp(body_peaks[0]["case_date"])
    ref_doy = body_peaks[0]["day_of_year"]
    drift, prev = [], None
    for p in body_peaks:
        shift = (p["day_of_year"] - ref_doy) / 7.0
        cands = [shift + 52.1775 * m for m in range(-1, 3)]
        val = shift if prev is None else min(c for c in cands if c >= prev - 2.5)
        cal_drift = (pd.Timestamp(p["case_date"]) - ref_date).days / 7 - (p["row"] - ref_row)
        drift.append({"row": p["row"], "from_climate_weeks": float(val),
                      "from_calendar_weeks": float(cal_drift)})
        prev = val
    out["D_drift_reference_row"] = ref_row
    out["D_drift"] = drift
    out["D_drift_mean_abs_disagreement_weeks"] = float(
        np.mean([abs(d["from_climate_weeks"] - d["from_calendar_weeks"]) for d in drift]))

    # E -- reports in the source that never reached the array
    dump = pd.read_csv(raw_csv, encoding="utf-8", encoding_errors="replace")
    dump["start"] = pd.to_datetime(dump["TimeStampStart"], format="%d-%b-%Y %H:%M:%S",
                                   errors="coerce").dt.normalize()
    used = set(cal.loc[exact, "report_week_start"])
    lo, hi = cal.loc[exact & (cal.array_week >= 51), "report_week_start"].agg(["min", "max"])
    in_span = sorted({w for w in dump["start"].dropna() if lo <= w <= hi})
    dropped = [w for w in in_span if w not in used]
    out["E_reports_in_source_span"] = len(in_span)
    out["E_reports_used"] = int(sum(1 for w in in_span if w in used))
    out["E_reports_dropped"] = len(dropped)

    # F -- the frozen protocol's folds
    import adaptive as base

    folds = base.build_folds(cases, 3, 3)
    f = []
    for fold in folds:
        tr, va, te = fold.train_index, fold.val_index, fold.test_index
        te_dates = cal.loc[te, "report_week_start"]
        f.append({"origin": fold.origin, "train_rows": [int(tr.min()), int(tr.max())],
                  "val_rows": [int(va.min()), int(va.max())],
                  "test_rows": [int(te.min()), int(te.max())],
                  "rows_2023_in_train": int(np.sum(tr <= 47)),
                  "test_calendar": [te_dates.min().date().isoformat(),
                                    te_dates.max().date().isoformat()]})
    out["F_folds"] = f

    extras = {"arr": arr, "cal": cal, "cases": cases, "temp_c": temp_c, "smooth": smooth,
              "peaks": peaks, "dropped": dropped, "dump_weeks": set(dump["start"].dropna()),
              "folds": folds, "exact": exact}
    return out, extras


# --------------------------------------------------------------------------- figures

def fig_row_to_date(x: dict, path: Path) -> None:
    cal, exact = x["cal"], x["exact"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    rows = cal["array_week"].to_numpy()
    dates = cal["report_week_start"]
    is2023 = cal["report_year"].eq(2023).to_numpy() & exact

    ref = pd.Timestamp("2013-01-05") + pd.to_timedelta(rows * 7, unit="D")
    ax.plot(rows, ref, color=MUTED, linewidth=1.0, zorder=1)
    ax.text(300, ref[300] - pd.Timedelta(days=320), "if the rows were consecutive weeks from 2013",
            color=INK_2, fontsize=9, rotation=17)

    ax.plot(rows[exact], dates[exact], color=NEUTRAL, linewidth=0.8, zorder=2)
    ax.scatter(rows[exact & ~is2023], dates[exact & ~is2023], s=16, color=BLUE, zorder=3,
               edgecolors=SURFACE, linewidths=0.6, label="2013–2022 report (exact match)")
    ax.scatter(rows[is2023], dates[is2023], s=16, color=ORANGE, zorder=3,
               edgecolors=SURFACE, linewidths=0.6, label="2023 report (exact match)")
    ax.scatter(rows[~exact], dates[~exact], s=34, marker="x", color=INK_2, zorder=4,
               linewidths=1.2, label="no exact report match (date unreliable)")

    ax.annotate("rows 0–47 are 2023\n(report Vol 50)", xy=(24, pd.Timestamp("2023-06-01")),
                xytext=(70, pd.Timestamp("2022-06-01")), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ax.annotate("row 49: series restarts\nat May 2013", xy=(51, pd.Timestamp("2013-05-15")),
                xytext=(95, pd.Timestamp("2013-02-01")), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})
    ax.annotate("row 395 spike:\nweek of 2020-12-26", xy=(395, pd.Timestamp("2020-12-26")),
                xytext=(318, pd.Timestamp("2022-05-01")), color=INK, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})

    ax.set_xlim(-5, 465)
    ax.set_xlabel("row in sri_lanka_2013-2022_shifted.npy")
    ax.set_ylabel("date of the weekly report those cases come from")
    ax.yaxis.set_major_locator(mdates.YearLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("Rows are not in date order: 2023 comes first, then 2013–2022 with gaps")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_two_timelines(x: dict, path: Path) -> None:
    cases, temp = x["cases"], x["temp_c"]
    rows = np.arange(len(cases))
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.35})
    for ax in (a, b):
        ax.axvspan(-0.5, 47.5, color=ORANGE, alpha=0.12, linewidth=0)
        ax.axvline(48, color=INK_2, linewidth=0.8)

    a.plot(rows, np.maximum(cases.sum(axis=1), 1), color=BLUE, linewidth=1.4)
    a.set_yscale("log")
    a.set_ylabel("national weekly cases (log)")
    a.set_title("Cases: the 2023 block sits in front of 2013, and row 395 spikes")
    a.text(24, 4500, "2023", color=INK, ha="center", fontsize=9)
    a.annotate("row 395", xy=(395, cases[395].sum()), xytext=(350, 5200), fontsize=9, color=INK,
               arrowprops={"arrowstyle": "-", "color": INK_2, "linewidth": 0.8})

    b.plot(rows, temp, color=BLUE, linewidth=1.4)
    b.set_ylabel("mean air temperature (°C)")
    b.set_title("Climate: one smooth annual cycle per ~52 rows, no break at row 48")
    for k in x["peaks"]:
        b.plot(k, x["smooth"][k], "o", color=BLUE, markersize=6, markeredgecolor=SURFACE,
               markeredgewidth=1.2)
    b.set_xlabel("row in the array (Jaffna excluded from the temperature mean: no GLDAS data)")
    b.set_xlim(-5, 465)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_climate_alignment(audit: dict, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})

    d = audit["D_annual_cycle_r2"]
    labels = list(d)
    yy = np.arange(len(labels))
    h = 0.36
    a.barh(yy - h / 2 - 0.02, [d[k]["r2_row_order"] for k in labels], height=h, color=BLUE,
           label="fitted against row order")
    a.barh(yy + h / 2 + 0.02, [d[k]["r2_case_calendar"] for k in labels], height=h, color=ORANGE,
           label="fitted against the cases' dates")
    for i, k in enumerate(labels):
        a.text(d[k]["r2_row_order"] + 0.01, i - h / 2 - 0.02, f"{d[k]['r2_row_order']:.2f}",
               va="center", fontsize=8, color=INK_2)
        a.text(d[k]["r2_case_calendar"] + 0.01, i + h / 2 + 0.02, f"{d[k]['r2_case_calendar']:.2f}",
               va="center", fontsize=8, color=INK_2)
    a.set_yticks(yy, labels)
    a.invert_yaxis()
    a.set_xlim(0, 1)
    a.set_xlabel("R² of an annual cycle")
    a.set_title("Climate follows row order, not the case dates")
    a.legend(loc="lower right")
    a.grid(axis="y", visible=False)

    ref = audit["D_drift_reference_row"]
    cal = pd.read_csv(CAL, parse_dates=["report_week_start"])
    body = cal[(cal.array_week >= ref) & cal.abs_diff.eq(0)]
    # Leave out reports whose dump date contradicts their own volume number (row 101):
    # a parsing error in the dump, not a property of the array.
    yr = body["report_week_start"].dt.year
    body = body[(body["report_year"] == yr)
                | ((body["report_week_start"].dt.month == 12) & (body["report_year"] == yr + 1))]
    ref_date = pd.Timestamp(next(p["case_date"] for p in audit["D_temperature_peaks"]
                                 if p["row"] == ref))
    line =(body["report_week_start"] - ref_date).dt.days / 7 - (body["array_week"] - ref)
    b.plot(body["array_week"], line, color=BLUE, linewidth=1.6,
           label="from the report dates (cases only)")
    dr = audit["D_drift"]
    b.scatter([d["row"] for d in dr], [d["from_climate_weeks"] for d in dr], s=46, color=ORANGE,
              zorder=3, edgecolors=SURFACE, linewidths=1.2,
              label="from temperature peaks (climate only)")
    b.axhline(52.18, color=MUTED, linewidth=0.9)
    b.text(ref + 4, 54.5, "one full year", fontsize=8, color=INK_2)
    b.set_ylim(-5, 66)
    b.set_xlabel("row in the array")
    b.set_ylabel(f"weeks the cases run ahead of the weather (vs row {ref})")
    b.set_title("Two independent measurements agree")
    b.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_coverage(x: dict, path: Path) -> None:
    cal, exact, dropped, dump_weeks = x["cal"], x["exact"], set(x["dropped"]), x["dump_weeks"]
    years = list(range(2013, 2024))
    grid = np.zeros((len(years), 53))          # 0 absent, 1 in source only, 2 in array
    for w in dump_weeks:
        if w.year in years:
            grid[years.index(w.year), min(w.isocalendar().week, 53) - 1] = max(
                grid[years.index(w.year), min(w.isocalendar().week, 53) - 1], 1)
    for w in cal.loc[exact, "report_week_start"]:
        if w.year in years:
            grid[years.index(w.year), min(w.isocalendar().week, 53) - 1] = 2
    for w in dropped:
        if w.year in years and grid[years.index(w.year), min(w.isocalendar().week, 53) - 1] < 2:
            grid[years.index(w.year), min(w.isocalendar().week, 53) - 1] = 1

    fig, ax = plt.subplots(figsize=(11, 4.4))
    cmap = ListedColormap([NEUTRAL, ORANGE, BLUE])
    ax.pcolormesh(np.arange(54) + 0.5, np.arange(len(years) + 1) - 0.5, grid, cmap=cmap,
                  vmin=0, vmax=2, edgecolors=SURFACE, linewidth=1.5)
    ax.set_yticks(range(len(years)), [str(y) for y in years])
    ax.invert_yaxis()
    ax.set_xticks([1, 10, 20, 30, 40, 52])
    ax.set_xlabel("ISO week of the report")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="in the array"),
                       Patch(color=ORANGE, label="in the source reports, not in the array"),
                       Patch(color=NEUTRAL, label="no report in the source")],
              loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    ax.set_title("Which weeks the array actually contains")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_folds(audit: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 3.6))
    folds = audit["F_folds"]
    for i, f in enumerate(folds):
        for (lo, hi), color, label in ((f["train_rows"], BLUE, "train"),
                                       (f["val_rows"], AQUA, "val"),
                                       (f["test_rows"], ORANGE, "test")):
            ax.barh(i, hi - lo + 1, left=lo, height=0.5, color=color, edgecolor=SURFACE, linewidth=2)
            if hi - lo > 25:
                ax.text((lo + hi) / 2, i, label, ha="center", va="center", fontsize=9, color=SURFACE,
                        fontweight="bold")
        ax.text(f["test_rows"][1] + 6, i,
                f"tests {f['test_calendar'][0][:7]} → {f['test_calendar'][1][:7]}",
                va="center", fontsize=9, color=INK)
    ax.add_patch(plt.Rectangle((2.5, -0.45), 45, len(folds) - 0.1, fill=False, edgecolor=INK,
                               linewidth=1.2, zorder=5))
    ax.text(52, -0.72, "rows 3–47 are 2023 reports, inside every training set", ha="left",
            va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(folds)), [f"origin {f['origin']:.2f}" for f in folds])
    ax.set_xlim(0, 560)
    ax.set_ylim(len(folds) - 0.4, -0.95)
    ax.set_xlabel("row in the array")
    ax.grid(axis="y", visible=False)
    ax.set_title("The frozen protocol trains on 2023 rows to forecast 2018–2022")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    style()
    OUT.mkdir(parents=True, exist_ok=True)

    audit, x = run_checks(args.raw)
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fig_row_to_date(x, OUT / "1_row_to_date.png")
    fig_two_timelines(x, OUT / "2_two_timelines.png")
    fig_climate_alignment(audit, OUT / "3_climate_alignment.png")
    fig_coverage(x, OUT / "4_week_coverage.png")
    fig_folds(audit, OUT / "5_protocol_folds.png")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
