"""Read-only result analysis and vector figures for the development manuscript.

Run from any directory: python paper/ieee/scripts/publication.py
No new-period case data are read. No model is trained. Dates for Fig. 1 use the
weekly grid verified by ac89792; historical model scores are never relabeled
as scores produced using that fix. The original project sign-flip/BH tests are
reused unchanged. Confidence intervals are descriptive t intervals over origins.
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import t
from common import FIG, RES, REPO, TAB, P9_FILE, LEGACY, REBUILT, load_json, project_stats as ps
from historical_tables import P9_ROWS
from review_analysis import holm

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42,
                     "axes.linewidth": .6, "lines.linewidth": .9})
BLUE, ORANGE = "#0072B2", "#D55E00"


def save(fig, name):
    # Fixed physical page size; tight crops would silently change final font size.
    fig.savefig(FIG / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def data_plot():
    legacy = np.load(LEGACY)[:, :, 5].astype(float).sum(1)
    cases = np.load(REBUILT)[:, :, 0].astype(float)
    dates = pd.date_range("2013-06-15", periods=len(cases), freq="7D")
    total = np.where(np.isnan(cases).all(1), np.nan, np.nansum(cases, axis=1))
    fig, axs = plt.subplots(2, 1, figsize=(3.5, 3.25))
    fig.subplots_adjust(left=.23, right=.96, bottom=.15, top=.93, hspace=.73)
    axs[0].plot(np.arange(len(legacy)), legacy / 1000, color=BLUE)
    axs[0].axvline(395, color="black", ls="--", lw=.7)
    axs[0].annotate("Source-table\nformula error", xy=(395, legacy[395]/1000),
                    xytext=(175, 8.4), fontsize=9, arrowprops={"arrowstyle":"->", "lw":.6})
    axs[0].set_xlabel("Legacy array week index")
    axs[1].plot(dates, total / 1000, color=BLUE)
    axs[1].set_xlabel("Report week start")
    for label, ax in zip(("(a) Legacy", "(b) Rebuilt"), axs):
        ax.set_ylabel("Weekly cases\n(thousands)")
        ax.text(0, 1.06, label, transform=ax.transAxes, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
    return_fig = {"first":str(dates[0].date()), "last":str(dates[-1].date()),
                  "weeks":len(cases), "missing":int(np.isnan(cases).all(1).sum()),
                  "legacy_error_total":float(legacy[395])}
    save(fig, "fig_data.pdf")
    return return_fig


def architecture():
    from drawio_diagrams import architecture as export_architecture
    export_architecture()


def protocol():
    from drawio_diagrams import protocol as export_protocol
    export_protocol()


def statistics():
    rows=load_json(P9_FILE); by=collections.defaultdict(list)
    for r in rows:by[r["name"]].append(r)
    comp={r["arm"]:r for r in ps.compare(rows,"RMSE","persistence","origin")}
    hp=holm([v["p"] for v in comp.values()])
    for v,p in zip(comp.values(),hp): v["p_holm"]=p
    ref=ps.cells(rows,"RMSE")["persistence"]; cells=ps.cells(rows,"RMSE")
    out={}
    for name, rs in by.items():
        d=ps.paired(cells[name],ref,"origin")
        hw=float(t.ppf(.975,len(d)-1)*np.std(d,ddof=1)/np.sqrt(len(d)))
        seed_sd=[np.std([r["RMSE"] for r in rs if r["origin"]==o],ddof=1)
                 for o in sorted({r["origin"] for r in rs})] if name!="persistence" else [0.]
        base_by={o:np.mean([r["RMSE"] for r in by["persistence"] if r["origin"]==o]) for o in {r["origin"] for r in rs}}
        out[name]={"rmse":float(np.mean([r["RMSE"] for r in rs])),"mae":float(np.mean([r["MAE"] for r in rs])),
                   **{f"rmse_h{k}":float(np.mean([r[f"RMSE_h{k}"] for r in rs])) for k in (1,2,3)},
                   "delta":float(d.mean()),"ci":[float(d.mean()-hw),float(d.mean()+hw)],
                   "skill":float(np.mean([1-r["RMSE"]/base_by[r["origin"]] for r in rs])),
                   "mean_seed_sd":float(np.mean(seed_sd)),"origins":len(d),"rows":len(rs),
                   "p_adj":comp[name]["p_adj"] if name in comp else None,
                   "p_raw":comp[name]["p"] if name in comp else None,
                   "p_holm":comp[name]["p_holm"] if name in comp else None}
    counts={}
    for metric in ("RMSE","val_RMSE"):
        c=ps.compare(rows,metric,"persistence","origin")
        counts[metric]={"better":sum(r["delta"]<0 and r["p_adj"]<.05 for r in c),
                        "worse":sum(r["delta"]>0 and r["p_adj"]<.05 for r in c),"family":len(c)}
    holm_counts={}
    for metric in ("RMSE","val_RMSE"):
        c=ps.compare(rows,metric,"persistence","origin");p=holm([r["p"] for r in c])
        holm_counts[metric]={"better":sum(r["delta"]<0 and v<.05 for r,v in zip(c,p)),"worse":sum(r["delta"]>0 and v<.05 for r,v in zip(c,p)),"family":len(c)}
    manifest={"holm_counts":holm_counts,"source":"seirgnn2/results/frozen9_plus_audit.json","sha256":hashlib.sha256(P9_FILE.read_bytes()).hexdigest(),
              "status":"historical development; original graph","counts":counts,"arms":out}
    (RES/"development_statistics.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    ev=["\n# Part E. Codex publication audit (historical development only)",
        "Derived by `paper/ieee/scripts/publication.py`; source hash and unrounded values in `paper/ieee/results/development_statistics.json`. Confidence intervals use nine seed-averaged origin differences and a Student t critical value; serial dependence and retrospective selection limit interpretation.",
        "| ID | What | Value | Protocol | Source | Status |","|---|---|---|---|---|---|"]
    for i,(name,label) in enumerate(P9_ROWS,200):
        s=out[name]; ev.append(f"| EV-{i} | {label}: RMSE, MAE, paired difference, 95% CI, skill, mean within-origin seed SD | {s['rmse']:.6f}, {s['mae']:.6f}, {s['delta']:.6f}, [{s['ci'][0]:.6f}, {s['ci'][1]:.6f}], {s['skill']:.6f}, {s['mean_seed_sd']:.6f} | P9 development, 9 origins | frozen9_plus_audit.json | VERIFIED (derived) |")
    ev.append("| EV-216 | Repaired report calendar: start, end, weeks, missing, districts | 2013-06-15, 2024-02-24, 559, 7, 25 | Rebuilt calendar; not rerun scores | ac89792:data/corrected/rebuilt_index.csv | VERIFIED |")
    ev.append("| EV-217 | Effective reporting and spatial import: rho_eff = rho exp(q); u_i = log(max((A I)_i, 1e-9)); lambda = lambda_base exp(0.1 beta (u_i - mean(u))); E multiplier = 2 sigmoid(g) with recovered-mass compensation | As stated | Stored decoder implementation | seirgnn2/models.py:325-373 at ef22ed3 | VERIFIED (code) |")
    ev.append("| EV-218 | Shared target weeks: train/validation and validation/test boundaries | 2, 2 at each of 9 origins | P9 historical | seirgnn2/core.py::build_folds, direct target-set intersection; partition_overlap.json | VERIFIED (derived) |")
    ev.append("| EV-219 | Recomputed Holm correction on 30 historical arms, origin unit; zero improvements and zero degradations at 0.05 on validation and test; adaptive anchored E0 test p_raw=0.19921875, p_BH=0.576171875, p_Holm=1 | As stated | P9 audit reanalysis, not new experiments | review_analysis.holm; development_statistics.json | VERIFIED |")
    ev.append("| EV-220 | Legacy source sensitivity: exclusion of six test windows touching row 395 on origin 0.85; mean of three origin RMSEs 44.7953 to 29.5210 (34.1 percent reduction); last-fold 68.62 to 22.80 | As stated | Legacy diagnostic, not retraining | review_audit_results.json; 30_audit.py | VERIFIED |")
    ev.append("| EV-221 | Corrected graph: 57 undirected shared borders, 114 directed neighbor entries, 25 loader self-loops = 139; historical 116 + 25 = 141 | As stated | GADM 4.1 stored border test | 3878e70; review_audit_results.json | VERIFIED |")
    ev.append("| EV-222 | Legacy row 395, published source report: row A has 15/24 recurrence matches and 7165 district sum versus national 35; row B plus Kalmunai totals 351 | As stated | WER Vol48 No02, 2020-12-26 to 2021-01-01 | report_corrections.json; primary report | VERIFIED |")
    ev.append("| EV-223 | Seroprevalence assumption: 68.2 percent in 1689 suburban Colombo participants, not a national susceptibility estimate | As stated | Source-study scope | Jeewandara 2015, doi:10.1371/journal.pone.0144799 | VERIFIED |")
    ev.append("| EV-224 | Training decimal values: lr 0.003, weight decay 0.0001; early-stop improvement threshold 1e-6, no minimum epochs | As stated | Historical P9 code | train.run_fold; core.EarlyStop | VERIFIED |")
    ev.append("| EV-225 | Horizon endpoints h1/h3: persistence 23.47/33.26; GCN residual 23.24/32.47 | As stated | Historical P9 | generated main_p9.tex; frozen9_plus_audit.json | VERIFIED |")
    p=REPO/"paper/EVIDENCE.md";txt=p.read_text(encoding="utf-8").split("\n# Part E. Codex publication audit")[0]
    p.write_text(txt.rstrip()+"\n"+"\n".join(ev)+"\n",encoding="utf-8")
    lines=[r"\begin{table*}[t]",r"\centering",
           r"\caption{Historical development comparison on the rebuilt series, using the original graph. RMSE and MAE are in cases per district-week, averaged over nine origins and three seeds. Horizon columns report RMSE. $\Delta$ is the seed-averaged paired RMSE difference from persistence; brackets give a descriptive 95\% t interval over origins. $p_{\rm adj}$ uses the unchanged exact sign-flip test with Holm adjustment over thirty arms. Retrospective selection and temporal dependence limit inference. No comparison is significant under Holm. Historical BH values are archived separately.}",
           r"\label{tab:main}",r"\small\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{@{}p{2.5in}rrrrrr@{}}",r"\toprule",
           r"Model & $h=1$ & $h=2$ & $h=3$ & RMSE & MAE & $\Delta$ [95\% CI]; $p_{\rm adj}$ \\",r"\midrule"]
    for i,(name,label) in enumerate(P9_ROWS,200):
        s=out[name];h=[np.mean([r[f"RMSE_h{k}"] for r in by[name]]) for k in (1,2,3)]
        last="--" if name=="persistence" else f"{s['delta']:+.2f} [{s['ci'][0]:+.2f}, {s['ci'][1]:+.2f}]; {s['p_holm']:.3f}"
        lines.append(label+" & "+" & ".join(f"{v:.2f}" for v in h+[s["rmse"],s["mae"]])+" & "+last+f" \\\\ % EV-{i}\n")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
    (TAB/"main_p9.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")
    selected=[("gcn+residual","GCN, residual\n(baseline GNN)"),("adaptive_gwn+residual","Adaptive GCN,\nresidual"),
              ("gcn+foi_res anchor","GCN, gated SEIR,\nanchored"),("adaptive_gwn+foi_res anchor E0enc","Adaptive GCN, gated SEIR,\nanchored, learned $E_0$")]
    fig,ax=plt.subplots(figsize=(3.5,2.75));fig.subplots_adjust(left=.56,right=.94,bottom=.25,top=.95)
    for y,(name,label) in enumerate(selected[::-1]):
        s=out[name];ax.errorbar(s["delta"],y,xerr=[[s["delta"]-s["ci"][0]],[s["ci"][1]-s["delta"]]],fmt="o",color=BLUE,capsize=3,ms=4,lw=1)
    ax.set_yticks(range(4),[l for _,l in selected[::-1]]);ax.axvline(0,color="black",ls="--",lw=.7)
    ax.set_xlabel("RMSE difference\nfrom persistence (cases)");ax.set_xticks([-3,-1,1]);ax.set_ylim(-.55,3.55)
    ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
    save(fig,"fig_forest.pdf")
    # Separate descriptive summary preserves the seed spread rather than
    # confusing initialization variation with uncertainty between origins.
    pd.DataFrame([{"arm":name,**{k:v for k,v in s.items() if k!="ci"},
                   "ci_lower":s["ci"][0],"ci_upper":s["ci"][1]} for name,s in out.items()]).to_csv(RES/"development_statistics.csv",index=False)
    supp=[r"\begin{table}[!ht]\centering",r"\caption{Additional descriptive statistics for the historical development comparison. Skill is the mean over origins of one minus model RMSE divided by persistence RMSE. Seed SD is the mean within-origin standard deviation over neural seeds; persistence is deterministic. Neither statistic is a significance test.}",
          r"\label{tab:seed}\small",r"\begin{tabular}{@{}p{1.8in}rr@{}}\toprule",r"Model & Skill & Seed SD \\\midrule"]
    for i,(name,label) in enumerate(P9_ROWS,200):
        ss=out[name];supp.append(f"{label} & {ss['skill']:+.3f} & {ss['mean_seed_sd']:.3f} \\\\ % EV-{i}")
    supp += [r"\bottomrule\end{tabular}\end{table}"]
    (TAB/"seed_variability.tex").write_text("\n".join(supp)+"\n",encoding="utf-8")
    return manifest


def main():
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    # Mirror core.build_folds' start-index boundaries without importing training.
    c=np.load(REBUILT)[...,0]; missing=set(np.flatnonzero(np.isnan(c).any(1)).tolist())
    ids=list(range(3,len(c)-3)); overlaps=[]
    for origin in np.arange(.5,.901,.05):
        cut=int(origin*len(ids));end=int(min(origin+.05,1)*len(ids))
        target_sets=[]
        for starts in (ids[:cut-30],ids[cut-30:cut],ids[cut:end]):
            clean=[i for i in starts if not any(k in missing for k in range(i-3,i+3))]
            target_sets.append({i+h for i in clean for h in range(3)})
        overlaps.append({"origin":round(float(origin),2),"train_validation":sorted(target_sets[0]&target_sets[1]),
                         "validation_test":sorted(target_sets[1]&target_sets[2])})
    assert all(len(o["train_validation"])==len(o["validation_test"])==2 for o in overlaps)
    (RES/"partition_overlap.json").write_text(json.dumps(overlaps,indent=2)+"\n",encoding="utf-8")
    data=data_plot();architecture();protocol();stats=statistics()
    (RES/"publication_manifest.json").write_text(json.dumps({"data_plot":data,"figure_widths_inches":{"fig_data":3.5,"fig_arch":7.16,"fig_forest":3.5,"evaluation_protocol":3.5},"inputs":{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LEGACY,REBUILT,P9_FILE]}},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"data":data,"counts":stats["counts"]},indent=2))


if __name__=="__main__":
    main()
