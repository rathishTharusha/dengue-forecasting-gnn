"""District totals from the 2012 Census, read from the official DCS district reports.

The HDX workbook used earlier sums its GN-division rows to 20,266,363 -- 93,076
short of the official census total of 20,359,439, and 0.1-0.7% short in every
district. So district totals come from the Department of Census and Statistics'
own per-district reports (Table A1, "Population by divisional secretariat
division, sex and sector"), one PDF per district.

Each extracted total must equal male + female on the same line, and the 25
districts must sum exactly to the national total printed in the DCS publication
"Census of Population and Housing 2012 - Key Findings" (p. 19). The build fails
otherwise.

Download the PDFs (git-ignored) into ``data/raw/dcs/census2012/`` from::

    http://www.statistics.gov.lk/pophousat/cph2011/Pages/Activities/Reports/District/<Name>.pdf

``<Name>`` is the district, with the DCS spellings ``Baticaloa`` and
``Monaragala``. Three districts publish Table A1 as a separate file instead:
``.../District/Anuradhapura/A1.pdf``, ``.../Monaragala/A1.pdf`` and
``.../Polonnaruwa/A1.pdf``.

Run::

    python analysis/_build/build_census_2012.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
ADJ = REPO / "notebooks" / "baseline" / "sri_lanka_adj_list.json"
PDFS = REPO / "data" / "raw" / "dcs" / "census2012"
OUT = REPO / "data" / "external" / "census_2012_dcs_district_totals.csv"
BASE = "http://www.statistics.gov.lk/pophousat/cph2011/Pages/Activities/Reports/District/"

NATIONAL_TOTAL_2012 = 20_359_439
DCS_SPELLING = {"Batticaloa": "Baticaloa", "Moneragala": "Monaragala"}
SEPARATE_A1 = {"Anuradhapura", "Moneragala", "Polonnaruwa"}


def source(name: str) -> tuple[Path, str]:
    spelled = DCS_SPELLING.get(name, name)
    if name in SEPARATE_A1:
        return PDFS / f"{spelled}_A1.pdf", f"{BASE}{spelled}/A1.pdf"
    return PDFS / f"{spelled}.pdf", f"{BASE}{spelled}.pdf"


def extract(path: Path, name: str) -> tuple[int, int, int, int]:
    """(total, male, female, page) from the first line labelled with the district."""
    import pymupdf

    labels = [r"District", re.escape(name), re.escape(DCS_SPELLING.get(name, name))]
    for page_no, page in enumerate(pymupdf.open(path), start=1):
        text = page.get_text().replace("\xa0", " ")
        for label in labels:
            for m in re.finditer(label, text):
                nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]*", text[m.end(): m.end() + 400])]
                if len(nums) >= 3 and nums[0] > 10_000 and nums[0] == nums[1] + nums[2]:
                    return nums[0], nums[1], nums[2], page_no
    raise SystemExit(f"{name}: no total = male + female line found in {path.name}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for name in sorted(json.loads(ADJ.read_text(encoding="utf-8"))):
        path, url = source(name)
        if not path.exists():
            raise SystemExit(f"missing {path}; download it from {url}")
        total, male, female, page = extract(path, name)
        rows.append({"district": name, "population_2012": total, "male": male, "female": female,
                     "pdf_page": page, "source_url": url})
    df = pd.DataFrame(rows)
    total = int(df["population_2012"].sum())
    if total != NATIONAL_TOTAL_2012:
        raise SystemExit(f"districts sum to {total:,}, not the official {NATIONAL_TOTAL_2012:,}")
    df.to_csv(OUT, index=False)
    print(df[["district", "population_2012", "male", "female"]].to_string(index=False))
    print(f"\n25 districts, each total = male + female, summing exactly to {NATIONAL_TOTAL_2012:,}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
