# Prompt for the independent (Codex) review

Codex is not installed on the machine where this draft was written, so the review has not been run. Paste the text below into Codex from the repository root, on branch `paper/ieee-draft`. It writes `paper/ieee/REVIEW_CODEX.md`.

---

You are an independent reviewer of a research paper draft written by another AI. Your job is to find errors, not to rewrite the paper. Do not edit paper/ieee/main.tex or any existing file. Write all findings to paper/ieee/REVIEW_CODEX.md.

Read first: paper/STYLE_GUIDE.md, paper/PAPER_PLAN.md, paper/EVIDENCE.md, paper/ieee/main.tex, paper/ieee/refs.bib.

CHECK 1: NUMBERS (most important)
For every number in main.tex, find its evidence ID. Then open the ORIGINAL source named in EVIDENCE.md (git show <commit>:<path>), not just the ledger, and confirm the value. Recompute derived numbers (means, standard deviations, percentage changes) from the raw outputs where possible. Flag any number without an evidence ID.

CHECK 2: CLAIMS VS EVIDENCE
Flag claims such as "improves" or "outperforms" that the numbers do not support, gains smaller than the seed spread presented as wins, comparisons between runs with different protocols, and any model not compared with the persistence baseline.

CHECK 3: METHODS VS CODE
Compare every equation, architecture description and hyperparameter in the paper with the actual code and configs in src/ and the notebooks. Report each mismatch with file:line.

CHECK 4: WRITING
Flag banned words, em dashes, and sentences that sound machine-written or are hard to follow. Quote each one and suggest a plainer version. Flag terms used before they are explained, inconsistent model names, and internal project words (Phase, Stage, EXP, ADR).

CHECK 5: FORMAT AND REFERENCES
IEEE format problems, broken \ref or \cite, unused, missing or incomplete bib entries, figures unreadable at column width, and the page limit.

OUTPUT
For each issue: ID, severity (BLOCKER = wrong number or false claim; MAJOR = misleading claim or method mismatch; MINOR = style or format), location (section and line in main.tex), what is wrong, evidence (file:line or the command you ran), and a suggested fix.
End with a count per severity and your overall judgment: is the paper accurate and ready for human review?

Extra focus for this paper:
- Check that the Abstract, Introduction and Conclusion claim nothing stronger than the Results section, especially contribution 3 and the validation vs test results.
- Recount the 30 arms yourself: how many are better or worse than persistence on validation and on test, after BH correction.
- Confirm that every legacy-array result (adaptive graph, WGAN-GP, interval comparison) is labelled as such and never shares a table with 9-origin results.
- Recompute the persistence-residual interval numbers (96.0% coverage at width 99, scaled 96.3% at 74) from the script output.
- Read paper/ieee/WRITING_NOTES.md and check every number listed there that rests on a log rather than a saved file.
