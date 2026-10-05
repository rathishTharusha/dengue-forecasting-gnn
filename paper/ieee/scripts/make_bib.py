"""Copy the approved reference entries (PAPER_PLAN.md section 7) from full_paper/overleaf/refs.bib."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "full_paper" / "overleaf" / "refs.bib"
OUT = Path(__file__).resolve().parents[1] / "refs.bib"
KEYS = ["tissera2020severe", "weng2024graph", "gulmohamed2026denguegnn", "liu2025seirlstm",
        "phaijoo2018sensitivity", "guo2019astgcn", "bai2021a3tgcn", "shi2019aagcn", "li2018dcrnn",
        "wu2019graphwavenet", "kipf2017gcn", "rodriguez2023einns", "wang2022causalgnn", "cao2023mepognn",
        "deng2020colagnn", "raissi2019pinn", "krishnapriyan2021failure", "yoon2019timegan", "kim2022revin",
        "shao2022stid", "salinas2020deepar", "bracher2021evaluating"]
# Added after the user approved it (PAPER_PLAN.md section 7); metadata from literature/manifest.csv only.
EXTRA = r"""@inproceedings{huang2019stgat,
  title     = {{STGAT}: Modeling Spatial-Temporal Interactions for Human Trajectory Prediction},
  author    = {Huang, Yingfan and Bi, Huikun and Li, Zhaoxin and Mao, Tianlu and Wang, Zhaoqi},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {6271--6280},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.00637}
}

@misc{epid2021wer0248,
  author       = {{Epidemiology Unit, Ministry of Health, Nutrition and Indigenous Medicine, Sri Lanka}},
  title        = {Weekly Epidemiological Report, Vol. 48, No. 02, 26th December 2020 to 1st January 2021},
  year         = {2021},
  howpublished = {\url{https://www.epid.gov.lk/storage/post/pdfs/vol_48_no_02-english_1.pdf}}
}

@article{jeewandara2015sero,
  title   = {Change in Dengue and {Japanese} Encephalitis Seroprevalence Rates in {Sri Lanka}},
  author  = {Jeewandara, Chandima and Gomes, Laksiri and Paranavitane, S. A. and Tantirimudalige, Mihiri
             and Panapitiya, Sumedha Sandaruwan and Jayewardene, Amitha and Fernando, Samitha
             and Fernando, R. H. and Prathapan, Shamini and Ogg, Graham S. and Malavige, Gathsaurie Neelika},
  journal = {PLOS ONE},
  volume  = {10},
  number  = {12},
  pages   = {e0144799},
  year    = {2015},
  doi     = {10.1371/journal.pone.0144799}
}

@book{dcs2015census,
  author       = {{Department of Census and Statistics, Sri Lanka}},
  title        = {Census of Population and Housing 2012, Key Findings},
  publisher    = {Ministry of Finance and Planning, supported by UNFPA},
  year         = {2015},
  note         = {ISBN 978-955-577-906-7}
}

@misc{dcs2024midyear,
  author       = {{Department of Census and Statistics, Sri Lanka}},
  title        = {Mid-year population by district and sex, 2014 to 2024},
  year         = {2024},
  howpublished = {\url{https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/Mid-year_population_by_district_and_sex_2024.pdf}}
}

@article{hersbach2020era5,
  title   = {The {ERA5} global reanalysis},
  author  = {Hersbach, Hans and Bell, Bill and Berrisford, Paul and others},
  journal = {Quarterly Journal of the Royal Meteorological Society},
  volume  = {146},
  number  = {730},
  pages   = {1999--2049},
  year    = {2020},
  doi     = {10.1002/qj.3803}
}

@misc{zippenfenig2024openmeteo,
  author = {Zippenfenig, Patrick},
  title  = {Open-Meteo.com Weather {API}},
  year   = {2024},
  doi    = {10.5281/zenodo.7970649},
  note   = {Zenodo}
}

@misc{didan2021mod13q1,
  author       = {Didan, Kamel},
  title        = {{MODIS/Terra} Vegetation Indices 16-Day {L3} Global 250m {SIN} Grid {V061}},
  year         = {2021},
  publisher    = {NASA Land Processes Distributed Active Archive Center},
  doi          = {10.5067/MODIS/MOD13Q1.061}
}

@misc{gadm41,
  author       = {{GADM}},
  title        = {{GADM} database of Global Administrative Areas, version 4.1},
  howpublished = {\url{https://gadm.org}},
  note         = {Accessed 5 October 2026}
}"""  # NEEDS CHECK: given names of the authors
entries = {m.group(2): e for e in re.split(r"\n(?=@)", SRC.read_text(encoding="utf-8"))
           if (m := re.match(r"@(\w+)\{([^,]+),", e))}
missing = [k for k in KEYS if k not in entries]
assert not missing, missing
OUT.write_text("\n\n".join([entries[k].strip() for k in KEYS] + [EXTRA]) + "\n", encoding="utf-8")
print("wrote", len(KEYS), "entries")
