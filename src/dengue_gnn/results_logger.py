"""Append experiment results to a CSV, loudly.

Rewritten during the Phase-2 review (finding F10). The previous version fell back
to ``0.0`` for every metric whose column was missing, and to ``1.0`` for the
learned gate -- a value that reads as "the model used pure geography" rather than
"this was never recorded". A logging failure that writes a plausible wrong number
into the file a paper cites is worse than one that crashes, because nothing
downstream can tell the difference.

Everything here either writes the real value or raises.

Rows are written in long format -- one record per ``(config, fold, seed,
horizon)`` -- so that mean +/- std and paired significance tests remain possible
after the fact. See ``results/README.md``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["append_rows", "read_rows"]


def append_rows(
    rows: Iterable[Mapping[str, object]],
    csv_path: str | Path,
    stamp: bool = True,
) -> Path:
    """Append result records to ``csv_path``, creating it if needed.

    Args:
        rows: Records to write. Every record must have identical keys -- a
            ragged batch is a bug in the caller, not something to paper over.
        csv_path: Destination CSV.
        stamp: Add a UTC ``timestamp`` column to each row.

    Returns:
        The resolved path written to.

    Raises:
        ValueError: If ``rows`` is empty, records disagree on their keys, or the
            existing file's header does not match the records being appended.
    """
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("refusing to write an empty result set")

    if stamp:
        now = datetime.now(timezone.utc).isoformat()
        for r in rows:
            r.setdefault("timestamp", now)

    fields = list(rows[0].keys())
    for i, r in enumerate(rows[1:], start=1):
        if list(r.keys()) != fields:
            missing = set(fields) - set(r)
            extra = set(r) - set(fields)
            raise ValueError(
                f"row {i} has inconsistent keys (missing={sorted(missing)}, "
                f"extra={sorted(extra)}); every row must share one schema"
            )

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    if exists:
        with open(path, newline="", encoding="utf-8") as fh:
            existing = next(csv.reader(fh), None)
        if existing != fields:
            raise ValueError(
                f"{path} has header {existing} but these rows have {fields}. "
                "Appending would silently misalign every column. Write to a new "
                "file, or migrate the existing one deliberately."
            )

    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

    return path.resolve()


def read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Read a results CSV back as a list of dicts.

    Values come back as strings; cast at the point of use so a malformed cell
    surfaces where it is interpreted rather than silently becoming ``0.0``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"no results file at {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
