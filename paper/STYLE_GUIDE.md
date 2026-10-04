# Writing style guide (for every author, human or AI)

Goal: a reader understands each paragraph on the first read. Use the correct technical terms, but explain them in plain words. The paper should read like careful researchers wrote it.

Audience: an ML researcher who knows little epidemiology, and an epidemiologist who knows little about graph neural networks.

Sentences
- One idea per sentence. Most sentences 12-22 words; some shorter, a few longer when the logic needs it.
- Active voice with "we": "We train the model on 2013-2019 data", not "The model was trained".
- Present tense for what the paper shows and how the model works. Past tense for what we did in experiments.
- American spelling (IEEE style), used consistently.

Terms
- Use the right technical term, then explain it once in plain words where it first appears. Example: "We use rolling-origin cross-validation: we train on all weeks before a cut-off date, test on the weeks after it, then move the cut-off forward and repeat."
- One name per thing. Each model variant has one name, used the same way in text, tables and figures. Define each abbreviation once.
- Never use internal project words in the paper (Phase 2, Stage D, EXP-012, ADR, branch names). Use method names.
- After every equation, one or two sentences saying what it means and why it is there.

Claims
- Numbers instead of adjectives: "MAE fell from X to Y (Z% lower)", not "greatly improved".
- Use "significant" only with a statistical test and its result.
- If a gain is small or within the spread across seeds, say so.
- Report what did not work, plainly: "This did not improve accuracy. We believe the reason is ..."
- Every number comes from paper/EVIDENCE.md. Put each sentence that contains a number on its own line in the .tex file and end that line with a comment such as % EV-017 (LaTeX ignores single line breaks).
- Every citation comes from the approved reference list. Never invent a reference.

Do not use:
delve, leverage, utilize, harness, novel, groundbreaking, cutting-edge, state-of-the-art (unless we beat a published result on the same data), robust (unless we tested robustness), seamless, comprehensive, holistic, pivotal, crucial, vital role, paramount, intricate, nuanced, landscape, realm, tapestry, paradigm, synergy, unlock, empower, showcase, underscore, shed light on, pave the way, a testament to, in today's world, in recent years, it is worth noting, it is important to note, notably, furthermore, moreover, additionally (at sentence start), plays a key role.

Also avoid:
- Em dashes. Use a comma, brackets or a new sentence.
- Lists of three adjectives ("accurate, efficient and scalable").
- "Not only X but also Y".
- Rhetorical questions.
- A last sentence that only repeats the paragraph.
- Stacked hedges ("may potentially suggest"). One hedge is enough.
- Bullet lists in the body. The only list allowed is the contributions list in the Introduction.

Example
Bad: "Leveraging a novel physics-informed paradigm, our robust framework seamlessly integrates epidemiological dynamics, significantly enhancing forecasting performance."
Good: "We add a loss term that penalizes forecasts which break a simple SEIR model of disease spread. At a 1-week horizon this lowered MAE from X to Y. At 4 weeks it made no difference."

Final test: read each paragraph aloud. If it sounds like a press release, rewrite it.

IEEE format
- \documentclass[conference]{IEEEtran}; packages: cite, amsmath, amssymb, graphicx, booktabs.
- Abstract 150-250 words, no citations, no equations. Then 4-6 Index Terms in alphabetical order.
- Table captions above tables, figure captions below figures. Refer to them as Table I and Fig. 1.
- Numeric citations [1] in order of first use, via BibTeX (IEEEtran style) in paper/refs.bib.
- Results as mean ± standard deviation, stating the number of seeds and folds.
- The "Acknowledgment" section includes a statement of how AI tools were used, as IEEE policy requires.
