"""Assemble the Kaggle dataset: every input file exactly as retrieved from its source.

Nothing in the dataset is processed by us. Each file is downloaded from the
place it was originally obtained (or, for the one file that has no public URL,
copied unmodified from where it was received), and its SHA-256 is checked
against ``data/external/source_manifest.csv`` wherever the manifest recorded
one when the file was first retrieved. ``SOURCES.csv`` in the dataset lists
every file with its URL, size, hash and retrieval date, and the notebook
re-verifies those hashes before it reads anything.

Run from the repository root::

    python full_paper/kaggle/fetch_sources.py            # -> full_paper/kaggle/dataset/

Then upload ``full_paper/kaggle/dataset/`` as a (private) Kaggle dataset.

Sources
-------
* Weekly dengue cases: the machine-readable table parsed from the Epidemiology
  Unit's Weekly Epidemiological Reports (``output_Dengue Fever.csv``, received
  from the benchmark authors; verified against all 553 official PDFs, see
  ``docs/DATA_PROVENANCE.md``). No public URL; copied byte for byte.
* WER Vol 48 No 02, the one report whose printed table is corrected: the
  Epidemiology Unit's server returns HTTP 500 at the time of writing, so the
  Internet Archive's capture of the same URL is used; its hash equals the one
  recorded when the file was first downloaded from epid.gov.lk.
* ERA5 daily climate: Open-Meteo Historical Weather API, one interior point per
  district computed from the GADM 4.1 boundaries in the authors' repository.
* Population: Department of Census and Statistics -- mid-year estimates by
  district (2014-2024) and the 2012 Census district reports (Table A1).
* District graph, GADM boundaries, district config and the benchmark array: the
  authors' repository, MLOpenSourceOpenScience/disease_modeling_MLOS2.
* PyMuPDF wheel (for reading the PDFs offline on Kaggle): PyPI.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "full_paper" / "kaggle" / "dataset"
MANIFEST = REPO / "data" / "external" / "source_manifest.csv"

AUTHORS = "https://raw.githubusercontent.com/MLOpenSourceOpenScience/disease_modeling_MLOS2/main"
DCS_CENSUS = "http://www.statistics.gov.lk/pophousat/cph2011/Pages/Activities/Reports/District/"
DCS_MIDYEAR = ("https://www.statistics.gov.lk/Resource/en/Population/Vital_Statistics/"
               "Mid-year_population_by_district_and_sex_2024.pdf")
WER_0202 = ("https://web.archive.org/web/20260627214326id_/"
            "https://www.epid.gov.lk/storage/post/pdfs/vol_48_no_02-english_1.pdf")
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
ERA5_DAILY = ["temperature_2m_mean", "temperature_2m_min", "temperature_2m_max",
              "precipitation_sum", "relative_humidity_2m_mean", "soil_moisture_0_to_7cm_mean"]
ERA5_START, ERA5_END = "2013-01-01", "2024-03-31"
PYMUPDF = "1.28.2"
UA = {"User-Agent": "Mozilla/5.0 (research data retrieval; dengue-forecasting-gnn)"}

#: DCS file names differ from the graph's district keys for two districts, and three
#: districts publish Table A1 as a separate file.
DCS_SPELLING = {"Batticaloa": "Baticaloa", "Moneragala": "Monaragala"}
SEPARATE_A1 = {"Anuradhapura", "Moneragala", "Polonnaruwa"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest() -> dict[str, str]:
    with MANIFEST.open(encoding="utf-8") as f:
        return {r["file"]: r["sha256"] for r in csv.DictReader(f)}


def download(url: str, dest: Path, tries: int = 6) -> bool:
    """Fetch ``url`` to ``dest`` unless already there; returns whether it downloaded."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    for k in range(tries):
        try:
            # The Internet Archive refuses browser-like agents from scripts; it accepts curl's.
            headers = {"User-Agent": "curl/8.5.0"} if "web.archive.org" in url else UA
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                data = r.read()
            dest.write_bytes(data)
            return True
        except Exception as exc:  # noqa: BLE001 -- retried, then re-raised
            if k == tries - 1:
                raise
            # Open-Meteo answers 429 when a long request exceeds the free tier's rate.
            wait = 75 if getattr(exc, "code", None) == 429 else 10 * (k + 1)
            print(f"    retry {k + 1} in {wait}s after {type(exc).__name__}: {exc}", flush=True)
            time.sleep(wait)


def district_points(gadm: Path, names: list[str]) -> dict[str, tuple[float, float]]:
    """Centroid if it lies inside the district, else a representative point (as fetched originally)."""
    from shapely.geometry import shape

    pts = {}
    for ft in json.loads(gadm.read_text(encoding="utf-8"))["features"]:
        name = ft["properties"]["NAME_1"]
        if name in names:
            geom = shape(ft["geometry"])
            p = geom.centroid if geom.contains(geom.centroid) else geom.representative_point()
            pts[name] = (round(p.y, 4), round(p.x, 4))
    missing = set(names) - set(pts)
    if missing:
        raise SystemExit(f"no GADM polygon for {sorted(missing)}")
    return dict(sorted(pts.items()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    known = manifest()
    today = dt.date.today().isoformat()
    rows: list[dict] = []

    def record(rel: str, url: str, note: str = "", manifest_key: str | None = None) -> None:
        path = OUT / rel
        h = sha256(path)
        expected = known.get(manifest_key) if manifest_key else None
        if expected and expected != h:
            raise SystemExit(f"{rel}: sha256 {h} does not match the manifest's {expected}")
        rows.append({"file": rel, "bytes": path.stat().st_size, "sha256": h, "source": url,
                     "retrieved": today, "manifest_match": "yes" if expected else "not recorded",
                     "note": note})
        print(f"  {rel:62s} {path.stat().st_size:>10,d}  {'manifest OK' if expected else ''}", flush=True)

    print("authors' repository")
    for rel, src in (("graph/sri_lanka_adj_list.json", "Models/sri_lanka_adj_list.json"),
                     ("graph/gadm41_LKA_1.json", "Data/Countries/gadm41_LKA_1.json"),
                     ("graph/disease_config.json", "Configs/disease_config.json"),
                     ("benchmark/sri_lanka_2013-2022_shifted.npy",
                      "Data/Datasets/sri_lanka_2013-2022_shifted.npy")):
        download(f"{AUTHORS}/{src}", OUT / rel)
        record(rel, f"{AUTHORS}/{src}")

    print("weekly dengue cases")
    src = REPO / "datasets" / "output_Dengue Fever.csv"
    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT / "cases" / "output_Dengue Fever.csv")
    record("cases/output_Dengue Fever.csv", "received from the benchmark authors (no public URL)",
           "parsed WER Table 1 dengue rows; verified against all 553 official PDFs")
    download(WER_0202, OUT / "cases" / "vol_48_no_02-english_1.pdf")
    record("cases/vol_48_no_02-english_1.pdf", WER_0202,
           "Internet Archive capture of the Epidemiology Unit URL (origin returns HTTP 500)",
           "wer/vol_48_no_02-english_1.pdf")

    print("population")
    download(DCS_MIDYEAR, OUT / "population" / "Mid-year_population_by_district_and_sex_2024.pdf")
    record("population/Mid-year_population_by_district_and_sex_2024.pdf", DCS_MIDYEAR, "",
           "dcs/Mid-year_population_by_district_and_sex_2024.pdf")
    names = sorted(json.loads((OUT / "graph" / "sri_lanka_adj_list.json").read_text(encoding="utf-8")))
    for name in names:
        spelled = DCS_SPELLING.get(name, name)
        if name in SEPARATE_A1:
            fname, url = f"{spelled}_A1.pdf", f"{DCS_CENSUS}{spelled}/A1.pdf"
        else:
            fname, url = f"{spelled}.pdf", f"{DCS_CENSUS}{spelled}.pdf"
        download(url, OUT / "population" / "census2012" / fname)
        record(f"population/census2012/{fname}", url, "", f"dcs/census2012/{fname}")

    print("ERA5 daily climate (Open-Meteo)")
    points = district_points(OUT / "graph" / "gadm41_LKA_1.json", names)
    (OUT / "climate").mkdir(parents=True, exist_ok=True)
    for name, (lat, lon) in points.items():
        q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": ERA5_START,
                                    "end_date": ERA5_END, "daily": ",".join(ERA5_DAILY),
                                    "models": "era5", "timezone": "Asia/Colombo"})
        url = f"{OPEN_METEO}?{q}"
        fetched = download(url, OUT / "climate" / f"{name}.json")
        record(f"climate/{name}.json", url, f"district point ({lat}, {lon}) from GADM 4.1")
        if fetched:
            time.sleep(25)   # an 11-year, 6-variable query counts as ~175 calls on the free tier

    print("PyMuPDF wheel")
    wheels = OUT / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "download", f"pymupdf=={PYMUPDF}", "--no-deps",
                    "--only-binary=:all:", "--platform", "manylinux_2_28_x86_64",
                    "--python-version", "3.11", "-d", str(wheels)], check=True,
                   capture_output=True)
    for whl in sorted(wheels.glob("*.whl")):
        record(f"wheels/{whl.name}", f"https://pypi.org/project/PyMuPDF/{PYMUPDF}/",
               "installed offline by the notebook to read the PDFs")

    with (OUT / "SOURCES.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "dataset-metadata.json").write_text(json.dumps({
        "title": "Dengue physics GNN - original sources",
        "id": "tharushaperera16/dengue-physics-gnn-sources",
        "licenses": [{"name": "other"}],
    }, indent=1), encoding="utf-8")
    print(f"\n{len(rows)} files, {sum(r['bytes'] for r in rows) / 1e6:.1f} MB -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
