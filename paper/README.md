# Phase 2 short paper

**Due:** Sunday 23 Aug 2026, 11:59 PM · **Limit:** 4 pages excluding references ·
**Template:** ACM `sigconf`

Execution plan and positioning: [`../docs/PHASE2_PLAN.md`](../docs/PHASE2_PLAN.md).
Read §1 (Positioning) before writing anything.

## Build

```bash
make              # main.pdf              — clean, for conference submission
make highlighted  # main-highlighted.pdf  — colour-coded, for grading
make pages        # page count — run this constantly
```

Needs `acmart`. Locally: `apt-get install texlive-publishers` (Debian/Ubuntu) or
`tlmgr install acmart`. On Overleaf it ships by default — select the *ACM Conference
Proceedings Primary Article* template and point it at `main.tex`.

## Two versions, one source

The brief requires both a clean version and a colour-highlighted version showing each
member's contributions. **Do not maintain two copies of the text.** Wrap your own
contributions in the per-member macros:

```latex
\cA{Text Tharusha wrote.}
\cB{Text Praveen D wrote.}
```

`\cA`–`\cF` map to the six members (see the colour definitions at the top of `main.tex` —
**assign these at the Day 0 standup**). In the clean build they expand to nothing; in the
highlighted build they colour the text.

Where the implementer and the writer differ, the brief requires a margin note naming both:

```latex
\cC{The adaptive adjacency is defined as...}\credit{Implemented by Bimsara; written by Janith.}
```

Track these **as you go**. Reconstructing six people's contributions on Sunday night is how
teams lose marks on a criterion that costs nothing to satisfy.

## Writing order

Sections are ordered by when they can be written, not by where they appear:

| Order | Section | Depends on |
|---|---|---|
| 1 | `01_introduction` | nothing — write Day 0 |
| 2 | `02_related_work` | nothing — write Day 0–1 |
| 3 | `04_experimental_setup` | nothing (protocol is frozen) — write Day 3 |
| 4 | `03_framework` | the code that actually ran — write Day 2–3 |
| 5 | `05_results` | frozen numbers — write Day 4 |
| 6 | `06_discussion_conclusion` | results — write Day 4 |
| 7 | `00_abstract` | everything — **write last**, Day 5 |

Each file opens with its word budget. **The whole paper is ~1,500 words of body text** —
this was measured by compiling the template at a range of word counts, not estimated. With
2 figures, 2 tables and ~6 equations, 1,600 words is the hard ceiling; 1,700 spills to a
fifth page.

Run `make pages` after every writing session. Going over means cutting from Related Work and
Experimental Setup — **never from Results.**

## Non-negotiables

- Every number traces to a CSV in `results/` and an entry in `docs/EXPERIMENT_LOG.md`.
- Tables are generated from CSVs by script, not typed by hand.
- §3 describes the loss we **actually trained with**, not the one in the proposal.
- Results freeze Friday 6 PM.
