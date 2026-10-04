# Writing notes for the IEEE draft (paper/ieee/)

Compiled with `latexmk -pdf main.tex` (MiKTeX, IEEEtran). No errors, no overfull boxes, no undefined references or citations. Body is 6 pages, references start on page 7. The folder compiles on Overleaf as-is (`main.tex`, `refs.bib`, `figures/*.pdf`, `tables/*.tex`; the scripts are not needed to compile).

Self-checks: `python paper/ieee/scripts/check_paper.py` reports 0 problems. It checks that every decimal or percentage number in a prose line appears (after rounding) in an evidence row named on that line, that every cited EV id exists, and it searches for banned words and em dashes. Integers such as 25 districts, 559 weeks, 27 runs and 69 configurations are not checked by the script; I checked them by hand against EV-163, EV-101, EV-116 and EV-164.

## Assumptions I made (please confirm or correct)

1. **D8 = option A.** Re-running was not allowed in this session, so Limitations states that the static graph has two non-real edges. If you want option B, the nine-origin arms that use the static graph need a re-run first.
2. **Venue, authors, page limit.** D1 still had placeholders. The title block says `[AUTHOR NAMES]` and the acknowledgment has a TODO. I assumed 6 pages of body plus references on extra pages.
3. **"Original GNN baseline" = GCN encoder with a direct head** (Table II first learned row). The plan's open question 9(e) was not answered. In the three-origin table the control is the GCN with a residual head, because that is the configuration the project's graph screen used.
4. **Seeds and spread.** "$\pm$" is the standard deviation over runs (27 or 9), which mostly reflects differences between origins. The paired gap and its p-value are the inference. Captions say so.
5. **Contribution 3 wording.** See "Things the plan got wrong or could not support", item 1.
6. **STGAT citation.** The original STGAT paper (Huang et al., ICCV 2019) is in `literature/manifest.csv` but not on the approved list in PAPER_PLAN.md, so it is not cited. The five encoders are cited through Weng et al. plus the four original papers on the list (ASTGCN, A3T-GCN, AAGCN, DCRNN). Add Huang et al. if you approve it.
7. **Acknowledgment text.** The AI-use statement is my wording. Edit it so it states exactly how your group used the tool.

## Things the plan or evidence got wrong, or could not support (found while writing)

1. **The plan's story overstates one causal link.** The plan says the population-scaled force of infection takes the bare SEIR head from +12 RMSE to within 1.3 of persistence. The two arms in the nine-origin table differ in the graph as well as the rate (GCN free rate +12.23; adaptive graph anchored +1.26). No bare-head arm pairs the two rates on one graph. The paper says this plainly. For the gated head a clean pair exists (GCN, free 28.43 vs anchored 27.62, neither significant). I did not change the story, only the wording.
2. **EVIDENCE.md miscount, now corrected.** The ledger said 13 of 30 arms are significant on validation. The correct count is 7 better and 8 worse (EV-149). Test: 0 better, 8 worse (EV-150).
3. **Rescue control reverses between validation and test.** On validation the simulator adds nothing for STGAT, A3T-GCN and DCRNN (+0.66, 0.00, -0.68). On test the SEIR head is lower than the gated head by 0.62, 1.66 and 3.50. The paper reports both and credits the repair to the anchor only on validation, which was the pre-set rule. A reader may reasonably ask for a stronger statement either way.
4. **The scatter figure's lowest-validation point is P1 (spatial penalty, 15.45), not model B (15.51).** The plan and some ledger rows describe B as the validation pick. In the paper B is "the lowest validation RMSE in Table III", which is true for that table. The figure labels the point without a name.
5. **Interval comparison (D7) came out against the neural intervals.** A persistence-residual interval covers 96.0% (width 99) and a scaled version covers 96.3% (width 74), while the Gaussian-head intervals cover 94.7% to 96.2% with widths 123 to 250. The paper therefore does not claim calibration as a contribution. Caveat: this uses the legacy-array folds (the only place the Gaussian-head results exist), which include the backlog week in the third fold.
6. **Week labels conflict.** The Kaggle notebook output says the rebuilt series runs 2013-W26 to 2024-W10, but `data/corrected/rebuilt_index.csv` has week starts from 2013-05-15 to 2024-02-24. The paper only says 559 weeks and does not give week numbers. Please check which is right.
7. **The legacy 19x spike claim in CLAUDE.md** is not repeated. The paper quotes the measured national totals instead (7165 vs 396 and 414, EV-162).
8. **Not copied from the ACM short paper:** the skewness and top-1% figures, the 440 deaths figure, and the "alpha converges to exactly zero in 108 runs" claim (see paper/ACM_CHECK.md, A5, A6, A8).

## Numbers that rest on a log or a document, not a saved output file

- EV-060, EV-061: the force-of-infection regression (0.35 to 0.38 up to 0.59 to 0.67; 39% to 41% of cells pinned). Cited in Discussion as "diagnostics". Saving the script output would remove this caveat.
- EV-050: entropy 0.469 is in the saved JSON; the "1.000 before the fix" comparison is from the log. The paper states 1.000 as the value for a uniform matrix, not as a measured earlier run.
- EV-185: the example edges come from the audit sheet (the JSON field `adj_top` holds the same strings, but I did not parse it).
- EV-184: the statement that six arms were hand-picked is from the experiment log.
- EV-115 horizon values for three-origin persistence: copied from the log (EV-020), tagged in the generator.
- EV-068, EV-069, EV-080, EV-083, EV-086, EV-089: older runs or documents, cited as "earlier experiment code" or "legacy array".

## Choices to cut for space

- `tables/intervals.tex` (the interval table) and `figures/fig_forest.pdf` (per-origin gaps) are generated but not included, to keep the body at 6 pages. Add them back if the limit is relaxed.
- The Kipf and Cola-GNN citations were dropped from the text for space. Their entries stay in `refs.bib` but are not cited, so they do not print. `krishnapriyan2021failure`, `raissi2019pinn` and `phaijoo2018sensitivity` are cited once each.

## Places a human should read closely

- Section IV-C and Eq. (6): the anchored force of infection, checked line by line against `seirgnn2/models.py` (comment next to the equation). Please confirm the notation for $S_i$ (susceptible fraction at the start of the window).
- Section IV-A: the sentence about the embedding initialization and weight decay. The ablation of the exemption alone was never run, and the sentence says so.
- Section VI-B: the gated-versus-SEIR paragraph (validation and test disagree) and the sentence that credits the repair to the anchor.
- Section VII-A: the three "we believe" statements (graph encodes outbreak timing, small variance left, validation-test disagreement). None was tested.
- Section V-B: "One author chose the six hand-picked arms". Name or rephrase as your group prefers.
- Abstract and Conclusion: the recommendation sentence is ours, not a result.

## Reproducing the tables and figures

```
python paper/ieee/scripts/interval_baseline.py   # writes results/interval_baseline.json (legacy folds)
python paper/ieee/scripts/make_figures.py        # figures/*.pdf and results/figure_values.json
python paper/ieee/scripts/make_tables.py         # tables/*.tex, results/evidence_generated.md, EVIDENCE.md Part C
python paper/ieee/scripts/make_bib.py            # refs.bib from the approved list
python paper/ieee/scripts/check_paper.py         # number, evidence-id, banned-word and em-dash check
cd paper/ieee && latexmk -pdf main.tex
```

`make_tables.py` rewrites Part C of `paper/EVIDENCE.md` (between the BEGIN/END GENERATED markers). Part D (code constants) is written by hand.

## Update after your answers (round 2)

- **Baseline GNN (item 2).** ADR 0001 and the v2 notebook define the original baseline as a GCN that predicts a residual over persistence on log1p targets (the v1 plain GCN on absolute counts lost to persistence). The paper now calls **GCN with a residual head** the baseline GNN (Table II, EV-102, per-horizon gaps EV-165) and retrains it under the new protocol; the direct-head GCN is shown as a separate row. Hyperparameters differ from the old notebook (legacy array, other learning rate), so the paper says "design" and "retrain", not "the original numbers".
- **Contribution 3, validation vs test (items 3, 4).** The Abstract, Introduction (contribution 3) and Conclusion now say: on validation the repair is explained by the anchor, on test the SEIR version is lower, and the anchored bare head is 1.26 cases worse than persistence with the scaling effect untested for the bare head. Nothing stronger than Section VI-B.
- **STGAT (item 6).** The code (`analysis/lib/reproduced.py::_STGAT`) is Weng et al.'s dengue adaptation (graph-attention layer, two LSTM layers) of Huang et al., ICCV 2019. The paper cites Huang et al. and says ours is the adaptation, not the pedestrian model. Added to PAPER_PLAN.md section 7 and `refs.bib`. Author given names are NOT in the repo, so the bib entry lists surnames only (NEEDS CHECK).
- **Counts (item 7).** All places agree: validation 7 better and 8 worse, test 0 better and 8 worse (EV-149, EV-150).
- **Intervals (item 8).** The text, the abstract and the check script now state that the interval comparison uses the legacy-array folds, which include the backlog week.
- **Placeholders (item 1).** Authors are `[NAMES, AFFILIATION, EMAILS]`, venue and page limit are in a TODO comment. With "6 + references" the draft fits (6 pages of body, references on page 7). With "6 including references" about 0.9 page must be cut.
- **Week index (item 5): NOT changed, waiting for you.** See the message to the user. The paper still says only "559 weeks" and Fig. 1b still plots the index dates.
- **Codex review (item 3, second block).** Codex is not installed here, so it was not run. `paper/ieee/REVIEW_PROMPT.md` holds the prompt with the `paper/ieee/` paths and your extra focus items.
