"""Build the LaTeX tables and the generated evidence rows from saved result files.

Reads: seirgnn2/results/frozen9_plus_audit.json (protocol P9),
full_paper/outputs/kaggle_run/{runs.jsonl,results.json} (protocols P3 and PK9),
analysis/results/improved_sweep.json and paper/ieee/results/interval_baseline.json.
Writes: paper/ieee/tables/*.tex and paper/ieee/results/evidence_generated.md.

Values that exist only in a markdown log (EVIDENCE.md) are written in this file as
constants and tagged LOG-ONLY with their evidence ID.
"""
from __future__ import annotations

import collections

import numpy as np
from common import KAGGLE, P9_FILE, RES, load_json, load_jsonl, write_tex
from common import project_stats as ps

EVID: list[tuple[str, str, str, str, str]] = []   # (id, what, value, protocol, source)
_next = [100]


def ev(what: str, value: str, protocol: str, source: str, fixed: str | None = None) -> str:
    if fixed:
        i = fixed
    else:
        i = f"EV-{_next[0]}"
        _next[0] += 1
    EVID.append((i, what, value, protocol, source))
    return i


def f2(x: float) -> str:
    return f"{x:.2f}"


def signed(x: float) -> str:
    return f"{x:+.2f}".replace("-", "$-$")


# ---------------------------------------------------------------- Table: P9 main
P9_ROWS = [  # (arm name in file, display name)
    ("persistence", "Persistence (last week)"),
    ("gcn+direct", "GCN, direct"),
    ("gcn+residual", "GCN, residual (baseline GNN)"),
    ("LSTM+direct", "LSTM, direct"),
    ("ASTGCN+direct", "ASTGCN, direct"),
    ("AAGCN+direct", "AAGCN, direct"),
    ("STGAT+direct", "STGAT, direct"),
    ("ASTGCN+residual", "ASTGCN, residual"),
    ("adaptive_gwn+residual", "Adaptive GCN, residual"),
    ("gcn+foi", "GCN, SEIR, free rate"),
    ("gcn+foi_res", "GCN, gated SEIR, free rate"),
    ("adaptive_gwn+foi anchor", "Adaptive GCN, SEIR, anchored"),
    ("adaptive_gwn+foi_res mass", "Adaptive GCN, gated SEIR, mass-action"),
    ("gcn+foi_res anchor", "GCN, gated SEIR, anchored"),
    ("adaptive_gwn+foi_res anchor", "Adaptive GCN, gated SEIR, anchored"),
    ("adaptive_gwn+foi_res anchor E0enc", "Adaptive GCN, gated SEIR, anchored, learned $E_0$"),
]


def p9() -> None:
    rows = load_json(P9_FILE)
    test = {r["arm"]: r for r in ps.compare(rows, "RMSE", "persistence", "origin")}
    val = {r["arm"]: r for r in ps.compare(rows, "val_RMSE", "persistence", "origin")}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    lines = []
    for name, label in P9_ROWS:
        rs = by[name]

        def m(k, rs=rs):
            return float(np.mean([r[k] for r in rs]))

        sd = float(np.std([r["RMSE"] for r in rs], ddof=1))
        base = [f2(m("val_RMSE")), f2(m("RMSE_h1")), f2(m("RMSE_h2")), f2(m("RMSE_h3")),
                f"{f2(m('RMSE'))} $\\pm$ {f2(sd)}"]
        if name == "persistence":
            cells = base + ["--", "--", "--"]
            i = ev("Persistence, 9 origins: val / h1 / h2 / h3 / test (mean ± sd over 9 origins)",
                   f"{m('val_RMSE'):.4f} / {m('RMSE_h1'):.4f} / {m('RMSE_h2'):.4f} / {m('RMSE_h3'):.4f} / "
                   f"{m('RMSE'):.4f} ± {sd:.4f}", "P9", "frozen9_plus_audit.json (derived)")
        else:
            t, v = test[name], val[name]
            cells = base + [signed(t["delta"]), f"{t['wins']}/{t['n']}", f"{t['p_adj']:.3f}"]
            i = ev(f"{label}: val / h1 / h2 / h3 / test ± sd (27 runs); Δtest, wins, p_adj(test); Δval, p_adj(val)",
                   f"{m('val_RMSE'):.4f} / {m('RMSE_h1'):.4f} / {m('RMSE_h2'):.4f} / {m('RMSE_h3'):.4f} / "
                   f"{m('RMSE'):.4f} ± {sd:.4f}; {t['delta']:+.4f}, {t['wins']}/{t['n']}, "
                   f"{t['p_adj']:.4f}; {v['delta']:+.4f}, {v['p_adj']:.4f}", "P9",
                   "frozen9_plus_audit.json via seirgnn2/stats.py (BH over 30 arms)",
                   fixed="EV-190" if name == "adaptive_gwn+foi_res mass" else None)
        lines.append(f"{label} & " + " & ".join(cells) + f" \\\\ % {i}")
    head = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Main results on the nine-origin protocol (rebuilt data, 25 districts, window 3, "
        "horizon 3, 3 seeds per origin). RMSE in cases per district-week. Horizon columns and "
        "test RMSE are means over 27 runs; $\\pm$ is the standard deviation over those runs and is "
        "dominated by differences between origins. $\\Delta$ is test RMSE minus persistence, averaged over "
        "origins (negative is better); Wins counts origins out of nine; $p_{\\mathrm{adj}}$ is the "
        "Benjamini-Hochberg adjusted exact sign-flip $p$-value of the test difference on the nine origin means, "
        "over all 30 arms compared. Val is validation RMSE, used for selection.}\n\\label{tab:main}\n"
        "\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n"
        "Model & Val & $h{=}1$ & $h{=}2$ & $h{=}3$ & Test RMSE & $\\Delta$ & Wins & $p_{\\mathrm{adj}}$ \\\\\n"
        "\\midrule\n")
    body = lines[0] + "\n\\midrule\n" + "\n".join(lines[1:])
    write_tex("main_p9.tex", head + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")


# ---------------------------------------------------------------- Tables from P3 (Kaggle run)
def p3() -> None:
    runs = [r for r in load_jsonl(KAGGLE / "runs.jsonl") if r["origin_set"] == "three"]
    by = collections.defaultdict(list)
    for r in runs:
        by[r["name"]].append(r)
    res = load_json(KAGGLE / "results.json")
    pers = res["persistence_3"]

    def stat(name):
        rs = by[name]

        def g(k):
            return float(np.mean([r[k] for r in rs]))

        return dict(val=g("val_RMSE"), h1=g("RMSE_h1"), h2=g("RMSE_h2"), h3=g("RMSE_h3"), test=g("RMSE"),
                    sd=float(np.std([r["RMSE"] for r in rs], ddof=1)), n=len(rs))

    def paired_vs(a, b):
        ka = {(r["origin"], r["seed"]): r["RMSE"] for r in by[a]}
        kb = {(r["origin"], r["seed"]): r["RMSE"] for r in by[b]}
        d = np.array([ka[k] - kb[k] for k in ka])
        return float(d.mean()), int((d < 0).sum()), len(d)

    rows = [
        ("GCN, residual (graph control)", "graph=gcn"),
        ("No graph, residual", "graph=none"),
        ("Adaptive GCN (simple), residual", "graph=adaptive"),
        ("AAGCN, direct", "AAGCN+direct"), ("AAGCN, gated", "AAGCN+gated"),
        ("AAGCN, SEIR, free rate", "AAGCN+foi"), ("AAGCN, gated SEIR, free rate", "AAGCN+foi_res"),
        ("STGAT, direct", "STGAT+direct"), ("STGAT, gated SEIR, free rate", "STGAT+foi_res"),
        ("DCRNN, direct", "DCRNN+direct"), ("DCRNN, gated SEIR, free rate", "DCRNN+foi_res"),
        ("AAGCN, direct, NB + season (B)", "B"),
        ("AAGCN, direct, squared error + season", "B, squared error"),
    ]
    # Persistence per horizon exists only in the log (EV-020); constants copied and tagged.
    ph = (29.6185, 35.1310, 42.1140)
    lines = []
    i = ev("Persistence, 3 origins: val / h1 / h2 / h3 / test ± sd (3 origins)",
           f"{pers['val']:.4f} / {ph[0]} / {ph[1]} / {ph[2]} / {pers['test']:.4f} ± 12.4088",
           "P3", "results.json persistence_3 (val, test); horizons LOG-ONLY from EV-020")
    lines.append(f"Persistence (last week) & {f2(pers['val'])} & {f2(ph[0])} & {f2(ph[1])} & {f2(ph[2])} & "
                 f"{f2(pers['test'])} $\\pm$ 12.41 & -- & -- \\\\ % {i} (EV-020)")
    lines.append("\\midrule")
    for label, name in rows:
        s = stat(name)
        if name == "graph=gcn":
            d = w = n = None
        else:
            d, w, n = paired_vs(name, "graph=gcn")
        dcell = "--" if d is None else signed(d)
        wcell = "--" if d is None else f"{w}/{n}"
        i = ev(f"{label}: val / h1 / h2 / h3 / test ± sd (9 runs); Δ vs graph=gcn, wins of 9 paired runs",
               f"{s['val']:.4f} / {s['h1']:.4f} / {s['h2']:.4f} / {s['h3']:.4f} / {s['test']:.4f} ± {s['sd']:.4f}; "
               + ("n/a" if d is None else f"{d:+.4f}, {w}/{n}"), "P3", "kaggle_run/runs.jsonl (derived)")
        lines.append(f"{label} & {f2(s['val'])} & {f2(s['h1'])} & {f2(s['h2'])} & {f2(s['h3'])} & "
                     f"{f2(s['test'])} $\\pm$ {f2(s['sd'])} & {dcell} & {wcell} \\\\ % {i}")
        if name == "graph=adaptive":
            lines.append("\\midrule")
    head = (
        "\\begin{table*}[t]\n\\centering\n"
        "\\caption{Indicative ablation on the three-origin protocol (rebuilt data, origins 0.55, 0.70, 0.85, "
        "3 seeds, 9 runs per row). With three origins an exact sign-flip test cannot return $p<0.25$, so "
        "none of these gaps is tested. RMSE in cases per district-week; $\\pm$ is the standard deviation over "
        "the nine runs. $\\Delta$ is test RMSE minus the GCN control (graph layer, residual head), averaged over the "
        "nine matched runs; Wins counts runs out of nine. Do not compare these numbers with Table~\\ref{tab:main}: "
        "the test weeks differ.}\n\\label{tab:p3}\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{@{}lrrrrrrr@{}}\n\\toprule\n"
        "Model & Val & $h{=}1$ & $h{=}2$ & $h{=}3$ & Test RMSE & $\\Delta$ & Wins \\\\\n\\midrule\n")
    write_tex("ablation_p3.tex", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

    # -------- what did not help (changes to B, P3)
    neg = [
        ("TimeGAN augmentation", "G2 TimeGAN"), ("SEIR-simulator pre-training", "G3 SEIR pre-training"),
        ("Climate lags 2--4 weeks", "K1 climate lags 2-4"), ("Climate lags 2--13 weeks", "K2 climate lags 2-13"),
        ("Climate lags 2--25 weeks", "K3 climate lags 2-25"), ("Spatial penalty", "P1 spatial penalty"),
        ("Metapopulation SEIR head", "P4 metapopulation SEIR"),
        ("District seasonal curves", "A2 district seasonal curves"),
        ("Half of the training windows", "B, 50% of training"),
        ("RevIN normalization", "R1 RevIN"), ("District embedding", "R2 district embedding"),
    ]
    bs = stat("B")
    i = ev("B: val, test", f"{bs['val']:.4f}, {bs['test']:.4f}", "P3", "kaggle_run/runs.jsonl (derived)")
    lines = [f"B itself & {f2(bs['val'])} & {f2(bs['test'])} & -- & -- \\\\ % {i}"]
    for label, name in neg:
        s = stat(name)
        d, w, n = paired_vs(name, "B")
        i = ev(f"{label} vs B (AAGCN direct NB season): val, test, Δtest vs B, wins of 9 paired runs",
               f"{s['val']:.4f}, {s['test']:.4f}, {d:+.4f}, {w}/{n}; B val {bs['val']:.4f} test {bs['test']:.4f}",
               "P3", "kaggle_run/runs.jsonl (derived)")
        lines.append(f"{label} & {f2(s['val'])} & {f2(s['test'])} & {signed(d)} & {w}/{n} \\\\ % {i}")
    head = (
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Changes to model B (AAGCN, direct head, negative-binomial likelihood, seasonal features) "
        "tested on the three-origin protocol, 9 runs per row, indicative only. "
        "$\\Delta$ is the test RMSE change against B (positive is worse); Wins counts matched runs out of nine.}\n"
        "\\label{tab:neg}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        "\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nChange & Val & Test & $\\Delta$ & Wins \\\\\n\\midrule\n")
    write_tex("negative_p3.tex", head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # -------- protocols table
    leg = res["array_floor"]
    p9p = [r for r in load_json(P9_FILE) if r["name"] == "persistence"]
    p9t = float(np.mean([r["RMSE"] for r in p9p]))
    i1 = ev("Persistence test RMSE by protocol: P3, P9, PK9, legacy all windows / without week 395",
            f"{pers['test']:.4f}, {p9t:.4f}, {res['persistence_9']['test']:.4f}, "
            f"{leg['all_windows']:.4f} / {leg['without_row_395']:.4f}", "all",
            "results.json; frozen9_plus_audit.json")
    t = ("\\begin{table}[t]\n\\centering\n\\caption{Evaluation protocols, 3 seeds each. Persistence test RMSE "
         "(cases per district-week) differs between protocols because the test weeks differ, so numbers from "
         "different rows are never compared. Legacy array: with / without the week-395 error.}\n"
         "\\label{tab:protocols}\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
         "\\begin{tabular}{@{}lccr@{}}\n\\toprule\n"
         "Protocol & Origins & Test share & Persistence \\\\\n\\midrule\n"
         f"Nine origins (main) & 0.50 to 0.90 by 0.05 & 5\\% & {f2(p9t)} \\\\ % {i1}\n"
         f"Three origins & 0.55, 0.70, 0.85 & 15\\% & {f2(pers['test'])} \\\\ % {i1}\n"
         f"Legacy array & 0.55, 0.70, 0.85 & 15\\% & {f2(leg['all_windows'])} / "
         f"{f2(leg['without_row_395'])} \\\\ % {i1}\n"
         "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_tex("protocols.tex", t)


def intervals() -> None:
    d = load_json(RES / "interval_baseline.json")
    g = d["gaussian_head"]
    lines = []
    for kind, lab in (("additive", "Persistence $\\pm$ residual quantiles"),
                      ("scaled", "Persistence $\\pm$ scaled residual quantiles")):
        i = ev(f"Interval baseline {kind}: coverage, mean width (legacy folds, mean of 3 folds)",
               f"{d[kind]['coverage']:.4f}, {d[kind]['width']:.2f}", "PL",
               "paper/ieee/results/interval_baseline.json (derived by scripts/interval_baseline.py)")
        lines.append(f"{lab} & {d[kind]['coverage'] * 100:.1f} & {d[kind]['width']:.0f} \\\\ % {i}")
    lines.append("\\midrule")
    for a in ("STGAT", "A3TGCN", "ASTGCN", "DCRNN", "AAGCN"):
        i = ev(f"Gaussian head {a}: PICP, MPIW (mean of 9 runs)", f"{g[a]['coverage']:.4f}, {g[a]['width']:.2f}",
               "PL", "analysis/results/improved_sweep.json")
        lines.append(f"{a}, Gaussian head & {g[a]['coverage'] * 100:.1f} & {g[a]['width']:.0f} \\\\ % {i}")
    t = ("\\begin{table}[t]\n\\centering\n\\caption{Nominal 95\\% intervals on the legacy-array folds "
         "(three folds, 68 test windows each). Coverage is the share of district-week targets inside the "
         "interval; width is the mean upper minus lower bound in cases. The first two rows need no network.}\n"
         "\\label{tab:interval}\n\\footnotesize\n\\begin{tabular}{@{}lrr@{}}\n\\toprule\n"
         "Interval & Coverage (\\%) & Width \\\\\n\\midrule\n" + "\n".join(lines) +
         "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    write_tex("intervals.tex", t)


def derived() -> None:
    """Counts and gaps quoted in the text, each recorded as an evidence row."""
    import csv

    rows = load_json(P9_FILE)
    for metric, lab in (("val_RMSE", "validation"), ("RMSE", "test")):
        t = ps.compare(rows, metric, "persistence", "origin")
        better = sum(r["p_adj"] < 0.05 and r["delta"] < 0 for r in t)
        worse = sum(r["p_adj"] < 0.05 and r["delta"] > 0 for r in t)
        ev(f"Arms with BH-adjusted p < 0.05 against persistence on {lab} RMSE: better, worse, of arms compared",
           f"{better}, {worse}, of {len(t)}", "P9", "frozen9_plus_audit.json via seirgnn2/stats.py")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)

    def h(name, k):
        return float(np.mean([r[k] for r in by[name]]))

    for name in ("gcn+direct", "adaptive_gwn+foi_res anchor", "adaptive_gwn+foi_res anchor E0enc",
                 "gcn+foi_res anchor", "adaptive_gwn+residual"):
        gaps = [h("persistence", f"RMSE_h{k}") - h(name, f"RMSE_h{k}") for k in (1, 2, 3)]
        ev(f"Persistence minus {name} test RMSE at h=1, 2, 3 (positive = model better)",
           ", ".join(f"{g:+.4f}" for g in gaps), "P9", "frozen9_plus_audit.json (arithmetic)")
    with open(KAGGLE / "rescue.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ev(f"Rescue control, {r['encoder']}: SEIR minus gated head, val delta (wins), test delta (wins)",
               f"{float(r['SEIR - gated (val)']):+.4f} ({r['wins']}), {float(r['SEIR - gated (test)']):+.4f} "
               f"({r['test wins']}); share of repair from anchor and gate {float(r['share of repair from anchor+gate']):.3f}",
               "P3", "kaggle_run/rescue.csv")
    fv = load_json(RES / "figure_values.json")
    d, v = fv["fig_data"], fv["fig_valtest"]
    ev("Legacy array national cases: week 395, week 394, week 396, largest other week",
       f"{d['legacy_week395']:.0f}, {d['legacy_week394']:.0f}, {d['legacy_week396']:.0f}, {d['legacy_max_other']:.0f}",
       "PL", "notebooks/baseline/sri_lanka_2013-2022_shifted.npy channel 5 (summed over districts)")
    ev("Rebuilt series: weeks, weeks with no report, first and last week start date",
       f"{d['rebuilt_weeks']}, {d['rebuilt_missing']}, {d['rebuilt_first']} to {d['rebuilt_last']}",
       "rebuilt", "data/corrected/rebuilt_cases.npy, rebuilt_index.csv")
    ev("Three-origin configurations: number, number below persistence on validation, of those above persistence on "
       "test; lowest-validation configuration (val, test)",
       f"{v['n_configs']}, {v['n_below_persistence_val']}, {v['of_those_test_above_persistence']}; "
       f"{v['lowest_val']} ({v['lowest_val_point'][0]:.2f}, {v['lowest_val_point'][1]:.2f})", "P3",
       "kaggle_run/runs.jsonl, results.json (figure_values.json)")
    gaps = [h("persistence", f"RMSE_h{k}") - h("gcn+residual", f"RMSE_h{k}") for k in (1, 2, 3)]
    ev("Persistence minus gcn+residual (baseline GNN) test RMSE at h=1, 2, 3 (positive = model better)",
       ", ".join(f"{g:+.4f}" for g in gaps), "P9", "frozen9_plus_audit.json (arithmetic)")


def write_evidence() -> None:
    md = ["| ID | What | Value | Protocol | Source | Status |", "|---|---|---|---|---|---|"]
    for i, what, val, prot, src in EVID:
        st = "LOG-ONLY (horizons)" if "LOG-ONLY" in src else "VERIFIED (derived by script)"
        md.append(f"| {i} | {what} | {val} | {prot} | {src} | {st} |")
    (RES / "evidence_generated.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def update_evidence_md() -> None:
    """Keep paper/EVIDENCE.md Part C in step with the rows the scripts produce."""
    path = RES.parent.parent / "EVIDENCE.md"
    text = path.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED -->", "<!-- END GENERATED -->"
    nl = chr(10)
    block = (begin + nl + nl + "# Part C. Rows derived by paper/ieee/scripts (IEEE draft)" + nl + nl
             + "Generated by `paper/ieee/scripts/make_tables.py` from the saved files named in each row. "
             "Do not edit by hand; rerun the script. Persistence per horizon for the three-origin protocol is "
             "copied from EV-020 (LOG-ONLY)." + nl + nl
             + (RES / "evidence_generated.md").read_text(encoding="utf-8") + nl + end)
    if begin in text:
        text = text[: text.index(begin)] + block + text[text.index(end) + len(end):]
    else:
        text = text.rstrip() + nl + nl + block + nl
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    p9()
    p3()
    intervals()
    derived()
    write_evidence()
    update_evidence_md()
    print(f"wrote tables and {len(EVID)} evidence rows")
