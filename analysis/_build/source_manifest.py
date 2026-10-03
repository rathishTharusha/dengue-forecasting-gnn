"""Checksums for every raw source file, so a manual re-download can be verified.

Anyone can download the sources by hand (``docs/DATA_PROVENANCE.md`` gives every
URL) and run::

    python analysis/_build/source_manifest.py --check

which compares their files against ``data/external/source_manifest.csv``. A
matching SHA-256 proves the file is byte-for-byte the one this project used.

Two sources are API responses rather than static files -- ERA5 via Open-Meteo
and MODIS via ORNL. Their JSON carries server metadata (e.g. generation time) that
changes on every request, and ERA5's most recent weeks can be revised, so they
are checked by **values**, not bytes: re-run their builders and diff the weekly
tables, e.g. ``python analysis/_build/fetch_era5_climate.py weekly`` then
``git diff --stat data/external/era5_weekly_by_district.csv``.

Run::

    python analysis/_build/source_manifest.py            # (re)write the manifest
    python analysis/_build/source_manifest.py --check    # verify files on disk
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "raw"
MANIFEST = REPO / "data" / "external" / "source_manifest.csv"

#: Static files, by folder, with where each came from.
SOURCES = {
    "dcs/census2012": "DCS Census 2012 district reports -- "
                      "http://www.statistics.gov.lk/pophousat/cph2011/Pages/Activities/Reports/District/",
    "dcs/Mid-year_population": "DCS mid-year population estimates by district 2014-2024 -- "
                               "https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/"
                               "Mid-year_population_by_district_and_sex_2024.pdf",
    "dcs/census2012_key_findings.pdf": "DCS Census 2012 Key Findings -- "
                                        "https://srilanka.unfpa.org/sites/default/files/pub-pdf/Census-2012.pdf",
    "hdx": "2012 Census GN-level workbook (used for over-60 share only) -- "
           "https://data.humdata.org/dataset/sri-lanka-census-of-population-and-housing-2012",
    "seroprev": "Jeewandara et al. preprint JATS XML and Table 1 image -- "
                "https://www.medrxiv.org/content/10.1101/2023.04.23.23288986v1",
    "disease_modeling_MLOS2/Data": "Authors' repository sample files and GADM 4.1 boundaries -- "
                                   "https://github.com/MLOpenSourceOpenScience/disease_modeling_MLOS2",
    "wer": "Epidemiology Unit Weekly Epidemiological Reports -- https://www.epid.gov.lk/storage/post/pdfs/",
}
DCS_POPULATION_PDF_URL = ("https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/"
                          "Mid-year_population_by_district_and_sex_2024.pdf")
SKIP_SUFFIXES = {".tmp"}
API_FOLDERS = ("era5_openmeteo", "modis_ndvi")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(rel: str) -> str:
    for prefix, text in SOURCES.items():
        if rel.startswith(prefix):
            return text
    return ""


def scan() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.rglob("*")):
        if not path.is_file() or path.suffix in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(RAW).as_posix()
        if rel.startswith(API_FOLDERS) or "/.git" in rel or rel.startswith("disease_modeling_MLOS2/.git"):
            continue
        if rel.startswith("disease_modeling_MLOS2/") and not rel.startswith("disease_modeling_MLOS2/Data/"):
            continue
        rows.append({"file": rel, "bytes": path.stat().st_size, "sha256": sha256(path),
                     "source": describe(rel)})
    return pd.DataFrame(rows)


def check() -> int:
    manifest = pd.read_csv(MANIFEST)
    ok = bad = absent = 0
    for rec in manifest.itertuples(index=False):
        path = RAW / rec.file
        if not path.exists():
            absent += 1
            continue
        if sha256(path) == rec.sha256:
            ok += 1
        else:
            bad += 1
            print(f"DIFFERENT  {rec.file}")
    print(f"static files: {ok} identical, {bad} different, {absent} not downloaded")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.check:
        return check()
    df = scan()
    df.to_csv(MANIFEST, index=False)
    print(f"wrote {len(df)} entries -> {MANIFEST}")
    print(df.groupby(df.file.str.split("/").str[0])["bytes"].agg(["count", "sum"]).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
