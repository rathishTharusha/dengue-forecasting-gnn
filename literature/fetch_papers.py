"""Rebuild ``literature/pdfs/`` from ``manifest.csv``.

The PDFs are third-party copyrighted works, so they are git-ignored (same rule
as ``papers/``) and this script is how a teammate gets them: every row carries
the canonical URL. Rows marked ``local:`` are copied from a file already in the
repository instead of downloaded.

Each file is checked to be a *complete* PDF: it must start with ``%PDF`` and
carry the ``%%EOF`` trailer near its end. Publishers sometimes answer an
automated request with an HTML challenge page, and a dropped connection leaves a
truncated file that still starts with ``%PDF`` -- both would otherwise sit under
a ``.pdf`` name looking fine until someone opens them. Outcomes, sizes and
SHA-256 hashes go to ``fetch_log.csv`` so a changed or missing file is visible.

    python literature/fetch_papers.py            # fetch anything missing
    python literature/fetch_papers.py --force    # re-fetch everything
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PDFS = HERE / "pdfs"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def is_complete_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF") and b"%%EOF" in data[-2048:]


def fetch(url: str, dest: Path) -> str:
    if url.startswith("local:"):
        src = REPO / url.removeprefix("local:")
        if not src.exists():
            return f"missing local source {src.relative_to(REPO)}"
        shutil.copyfile(src, dest)
        return "copied"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf,*/*"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read()
    if not is_complete_pdf(body):
        return (f"not a complete PDF ({len(body)} bytes, starts {body[:15]!r}) "
                "-- fetch manually")
    dest.write_bytes(body)
    return "downloaded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    PDFS.mkdir(exist_ok=True)

    rows = list(csv.DictReader((HERE / "manifest.csv").open(encoding="utf-8")))
    log = []
    for row in rows:
        dest = PDFS / row["file"]
        if dest.exists() and not is_complete_pdf(dest.read_bytes()):
            dest.unlink()  # truncated or not a PDF: never keep it
        if dest.exists() and not args.force:
            status = "present"
        else:
            try:
                status = fetch(row["url"], dest)
            except Exception as exc:
                status = f"failed: {type(exc).__name__}: {exc}"
            time.sleep(1.0)  # be polite to arXiv and publishers
        ok = dest.exists() and is_complete_pdf(dest.read_bytes())
        size = dest.stat().st_size if ok else 0
        sha = hashlib.sha256(dest.read_bytes()).hexdigest()[:16] if ok else ""
        log.append({"key": row["key"], "file": row["file"], "status": status,
                    "bytes": size, "sha256_16": sha, "url": row["url"]})
        print(f"{'OK ' if ok else '-- '} {row['key']:28s} {size / 1e6:6.2f} MB  {status}")

    with (HERE / "fetch_log.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    have = sum(1 for r in log if r["bytes"])
    print(f"\n{have}/{len(log)} PDFs present in {PDFS.relative_to(REPO)}")


if __name__ == "__main__":
    main()
