# IEEE reference audit

Checked 2026-10-05. Only existing approved entries were retained; no new paper was invented or added. The ACM bibliography is unchanged. The IEEE generator applies the checked fields in `paper/ieee/scripts/reference_corrections.json` after copying the approved source entries.

## Primary-source metadata checks and corrections

| Key | Check/correction | Primary record |
|---|---|---|
| tissera2020severe | Added DOI; title, year, volume and pages agree | [CDC article](https://wwwnc.cdc.gov/eid/article/26/4/19-0435_article) |
| weng2024graph | Added verified DOI; full metadata still needs author confirmation where inaccessible | [IEEE record](https://doi.org/10.1109/BigData62323.2024.10825335) |
| wu2019graphwavenet | Added DOI and pages 1907–1913; author list agrees | [IJCAI proceedings](https://www.ijcai.org/proceedings/2019/264) |
| rodriguez2023einns | Added volume, issue, pages and DOI; authors agree | [AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/26690) |
| wang2022causalgnn | Added volume, issue, pages and DOI; authors agree | [AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/21479) |
| guo2019astgcn | Added DOI; authors, pages and year agree | [AAAI citation](https://ojs.aaai.org/index.php/AAAI/citationstylelanguage/get/acm-sig-proceedings?publicationId=2344&submissionId=3881) |
| raissi2019pinn | Replaced abbreviated title with full title and added DOI | [Publisher article](https://www.sciencedirect.com/science/article/pii/S0021999118307125) |
| salinas2020deepar | Added publisher DOI; published author-list check remains open | [Publisher record](https://www.sciencedirect.com/science/article/pii/S0169207019301888), [author preprint](https://arxiv.org/abs/1704.04110) |
| bracher2021evaluating | Added DOI; authors, year and article number agree | [PLOS citation](https://journals.plos.org/ploscompbiol/article/citation?id=10.1371%2Fjournal.pcbi.1008618) |
| huang2019stgat | Prior verified DOI and authors preserved; manuscript clearly distinguishes the dengue adaptation | Prior primary-source check in PAPER_PLAN.md |
| liu2025seirlstm | Prior verified DOI/metadata preserved | Prior primary-source check in PAPER_PLAN.md |
| epid2021wer0248 | Added citation to the already approved source report in Data; URL rendering supported with the url package | Source report and `data/external/report_corrections.json` |

## Structural checks

`paper/ieee/scripts/audit_publication.py` checks duplicate keys, missing cited entries, unused keys, missing DOI/URL fields, and unresolved LaTeX citations/references. Unused approved entries are retained as source material and do not print in the manuscript. Missing DOI is reported, not guessed. The detailed machine-readable result is `paper/ieee/results/publication_checks.json`.

## Remaining verification

The publisher DeepAR page could not be fully opened. The available author preprint has three authors and the stored entry reproduces that list, but the complete published-version author list requires checking. This is flagged rather than filled by inference.

Other approved references without a fresh full-document check retain their previous provenance and do not receive a blanket VERIFIED label. The Phaijoo reference has an unusual stored issue `a`; it is no longer cited in this manuscript and requires checking before reuse. Weng author metadata and the broad DengueGNN mechanism statement need a final source-document read by the authors. Approvals for primary-source entries in the earlier plan are preserved; census publication dates must not be inferred from PDF creation dates alone.

The manuscript does not cite EINNs as a graph network specifically, does not equate the pedestrian STGAT paper with the dengue adaptation, and does not use the PINN failure paper to explain an untested causal mechanism in our runs.
