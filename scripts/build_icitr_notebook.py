"""Build the single local ICITR walkthrough, including its source snapshots.

Run ``python scripts/build_icitr_notebook.py``. The resulting notebook does not
import this builder or its cell-source helpers when opened by a reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "full_paper/icitr/reproduce_all_results.ipynb"


def percent_cells(source: str) -> list:
    cells = []
    kind = None
    buf = []

    def flush():
        if kind is None or not any(line.strip() for line in buf):
            return
        text = "\n".join(buf).strip("\n")
        if kind == "markdown":
            text = "\n".join(
                line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
                for line in text.splitlines()
            )
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            cells.append(nbformat.v4.new_code_cell(text))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def source_snapshots() -> dict[str, str]:
    paths = list((REPO / "full_paper/kaggle/src").glob("*.py"))
    paths += list((REPO / "seirgnn2").glob("*.py"))
    paths += list((REPO / "analysis/lib").glob("*.py"))
    paths += list((REPO / "src/dengue_gnn").glob("*.py"))
    paths += [
        REPO / "analysis/_build/population_vintages.py",
        REPO / "scripts/verify_paper_numbers.py",
    ]
    return {p.relative_to(REPO).as_posix(): p.read_text(encoding="utf-8") for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the default audit cells")
    args = parser.parse_args()
    from icitr_array_audit_cells import array_audit_cells
    from icitr_prospective_cells import prospective_cells

    cells = percent_cells((REPO / "scripts/icitr_notebook_walkthrough.py").read_text("utf-8"))
    snapshots = source_snapshots()
    hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in snapshots.items()}
    payload = nbformat.v4.new_code_cell(
        "# Embedded source snapshots. These are implementation code, not experimental results.\n"
        "# Collapsed to keep the walkthrough readable; expand to inspect any definition.\n"
        f"MODULE_SOURCES = {snapshots!r}\nSOURCE_SHA256 = {hashes!r}\n"
    )
    historical_graph = subprocess.run(
        ["git", "show", "3878e70^:notebooks/baseline/sri_lanka_adj_list.json"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    payload.source += f"HISTORICAL_GRAPH = {json.loads(historical_graph)!r}\n"
    payload.metadata.update(jupyter={"source_hidden": True}, tags=["source-snapshots"])
    # Insert after the setup cell, before a cell first reads an implementation.
    insert_at = next(i for i, c in enumerate(cells) if "SNAPSHOT_INSERTION_POINT" in c.source)
    cells[insert_at] = payload
    audit_at = next(
        i for i, c in enumerate(cells) if c.cell_type == "markdown" and c.source.startswith("## 4.")
    )
    cells[audit_at:audit_at] = [nbformat.from_dict(c) for c in array_audit_cells()]
    cells.extend(nbformat.from_dict(c) for c in prospective_cells())
    cells.extend(percent_cells((REPO / "scripts/icitr_notebook_training.py").read_text("utf-8")))
    for index, cell in enumerate(cells):
        cell.id = f"c{index:03d}-" + hashlib.sha256(cell.source.encode()).hexdigest()[:10]
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_cell_{index}", "exec")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "icitr_walkthrough": {"source_commit": commit, "default_mode": "saved-result audit"},
        },
    )
    nbformat.validate(notebook)
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(
            notebook,
            timeout=600,
            kernel_name="python3",
            resources={"metadata": {"path": str(REPO)}},
        ).execute()
    nbformat.write(notebook, OUT)
    print(f"Wrote {OUT.relative_to(REPO)} ({len(cells)} cells; executed={args.execute})")


if __name__ == "__main__":
    main()
