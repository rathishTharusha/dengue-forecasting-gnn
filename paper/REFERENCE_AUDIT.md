# Final reference audit ? 5 October 2026

The active IEEE/root bibliography now contains exactly the 15 cited references. Original source bibliographies and the earlier ACM bibliography are preserved. Bibliographic checks cover author order, title, venue/year, volume/issue/pages where applicable, and DOI/URL. Publisher-deposited records are saved in `ieee/results/reference_metadata.json`; errors on unused records in that file are not silently marked verified.

| Active key | Metadata/support check | Correction or status |
|---|---|---|
| tissera2020severe | CDC publisher record and Crossref DOI 10.3201/eid2604.190435 | Authors/order, title, 26(4), 682?691, 2020 agree; supports Sri Lankan outbreak motivation. |
| weng2024graph | IEEE-deposited Crossref DOI 10.1109/BigData62323.2024.10825335; local author manuscript first page/full comparison text | **Added eighth author Mahi Pasarkar and pages 4448?4456.** Local manuscript lists seven authors and says final code will be provided: it is an earlier version, so published metadata takes precedence. No persistence baseline found in its compared model set; statistical baselines ARE present, now explicitly acknowledged. |
| liu2025seirlstm | [PLOS article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013540), Crossref | Six authors, 21(9), e1013540, 2025 agree. Methods explicitly give gamma=1/week, omega=7/10/week and eleven-fold under-ascertainment scaling. These motivate assumptions, not their nationwide validation. |
| huang2019stgat | IEEE-deposited Crossref DOI 10.1109/ICCV.2019.00637 | Five authors, ICCV 2019, pages 6271?6280 verified. Manuscript distinguishes the dengue adaptation. |
| bai2021a3tgcn | Publisher-deposited Crossref DOI 10.3390/ijgi10070485 | **Corrected Jiandong Bai, Yujiao Song and Zhixiang Hou; added Haifeng Li.** Seven authors; ISPRS IJGI 10(7), 485, 2021. Previous names Jizhou/Yanyan/Huang were incorrect. |
| guo2019astgcn | AAAI/Crossref DOI 10.1609/aaai.v33i01.3301922 | Five authors, 33(01), 922?929, 2019 agree. |
| li2018dcrnn | [Author record](https://arxiv.org/abs/1707.01926), local primary PDF | Four authors, title and ICLR 2018 agree; URL retained, no invented page range. |
| shi2019aagcn | [Author record](https://arxiv.org/abs/1805.07694), local primary PDF | Four authors, CVPR 2019, 12026?12035; cited as basis of adapted graph encoder, not exact dengue implementation. |
| gulmohamed2026denguegnn | [Nature publisher article](https://www.nature.com/articles/s41598-026-43073-y), Crossref | Four authors, Scientific Reports 16, article 10584, 2026; issue 1 added. Broad unsupported contrast about its mechanism removed. |
| wu2019graphwavenet | IJCAI proceedings/Crossref DOI 10.24963/ijcai.2019/264 | Five authors and pages 1907?1913 agree. |
| rodriguez2023einns | AAAI/Crossref DOI 10.1609/aaai.v37i12.26690 | Five authors, 37(12), 14453?14460, 2023 agree. Cited for epidemiological neural structure, not specifically a graph architecture. |
| wang2022causalgnn | AAAI/Crossref DOI 10.1609/aaai.v36i11.21479 | Six authors, 36(11), 12191?12199, 2022 agree. |
| cao2023mepognn | [Author record](https://arxiv.org/abs/2306.14857) | Six authors and 2023 extended-preprint title agree. Explicitly retained as arXiv extended version; not silently substituted with a different conference version. |
| epid2021wer0248 | Original official URL and repository extraction/correction evidence | Vol.48 No.02, table period December 26, 2020?January 1, 2021. Fresh web retrieval returns HTTP 403; reproducibility uses saved report-correction evidence and source-audit records. Human authors should retain the source PDF with the submission archive. |
| jeewandara2015sero | [PLOS article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0144799), Crossref | Eleven authors, 10(12), e0144799, 2015 agree. The 68.2% result is from 1689 suburban Colombo participants, not a national susceptible fraction. |

## Disclosure policy

[IEEE's AI author guidance](https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/) requires identification of tools, affected sections and the level of assistance in Acknowledgment. The disclosure identifies Claude Code and Codex, drafting throughout the text, source/statistical checks and figure/table generation. It does not falsely claim completed human approval. Author signoff and any additional ICITR-specific instructions remain human responsibilities.

## Structural checks

Zero duplicate keys, unresolved citations, missing cited entries or unused active entries. Every active entry has DOI or URL. The generator applies checked corrections and synchronizes `paper/refs.bib`. Removed unused entries remain available in historical source bibliographies; their removal does not imply they are incorrect or invalidate the separate archived ACM paper.
