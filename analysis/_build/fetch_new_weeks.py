"""Download the weekly epidemiological reports after the last week of the rebuilt series.

The series ends at Vol. 51 No. 10 (week starting 2024-02-24). The Epidemiology Unit lists later
reports on https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report.
File names carry a hash, so the listing is scraped; a browser User-Agent is required (a plain
request gets HTTP 403). Every file is stored under ``data/raw/wer_new`` (git-ignored) and its
SHA-256 and URL are written to the committed manifest ``data/external/wer_new_manifest.csv``.

Rules fixed before any parsing (see ``parse_new_weeks.py``): nothing here reads a case count.

    python analysis/_build/fetch_new_weeks.py
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RAW = REPO / "data" / "raw" / "wer_new"
MANIFEST = REPO / "data" / "external" / "wer_new_manifest.csv"
LISTING = "https://www.epid.gov.lk/weekly-epidemiological-report/weekly-epidemiological-report"
BASE = "https://www.epid.gov.lk"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
LAST_VOLUME, LAST_NUMBER = 51, 10          # last report already in the rebuilt series


def get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


def listing() -> list[tuple[int, int, str]]:
    html = get(LISTING).decode("utf-8", errors="ignore")
    out = []
    for url, vol, no in re.findall(r'["\']([^"\']*?vol[_ ]?(\d+)[_ ]no[_ ]?(\d+)[^"\']*?\.pdf)["\']', html, flags=re.I):
        out.append((int(vol), int(no), url if url.startswith("http") else BASE + url))
    return sorted(set(out))


def is_new(vol: int, no: int) -> bool:
    return (vol, no) > (LAST_VOLUME, LAST_NUMBER)


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    items = [x for x in listing() if is_new(x[0], x[1]) and "english" in x[2].lower()]
    print(f"{len(items)} English report files after Vol {LAST_VOLUME} No {LAST_NUMBER}")
    for vol, no, url in items:
        path = RAW / f"vol{vol}_no{no:02d}__{Path(url).name}"
        if not path.exists():
            path.write_bytes(get(url))
        b = path.read_bytes()
        if b[:4] != b"%PDF":
            raise SystemExit(f"{url} is not a PDF")
        rows.append({"volume": vol, "number": no, "url": url, "file": path.name,
                     "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
