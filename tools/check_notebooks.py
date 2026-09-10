"""Sanity-check every tracked notebook before it reaches main.

Catches the three failure modes that actually bite this project:

1. A notebook that is not valid JSON / nbformat -- usually a Google Drive sync
   conflict or a botched merge, and it silently refuses to open in Colab.
2. Committed cell outputs whose execution counts are out of order, which means
   the notebook was not run top-to-bottom and its results prove nothing.
3. ``QUICK_TEST = True`` left switched on -- those numbers are intentionally
   degraded (fewer districts, fewer epochs) and must never reach the report.

Run: ``python tools/check_notebooks.py``
Exit code 1 on any error. Warnings do not fail the build.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
#: Every notebook tree in the repo. The three added workspaces are equally
#: capable of arriving corrupted from a sync or with QUICK_TEST left enabled,
#: and their notebooks are generated, so a cell-source escaping slip produces
#: an .ipynb that looks fine in review and dies on execution.
NOTEBOOK_GLOBS = (
    "notebooks/**/*.ipynb",
    "crosscheck/notebooks/**/*.ipynb",
    "reproduction/kaggle/kernels/**/*.ipynb",
    "analysis/notebooks/**/*.ipynb",
)


def check_notebook(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for one notebook."""
    errors: list[str] = []
    warnings: list[str] = []
    rel = path.relative_to(REPO_ROOT).as_posix()

    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{rel}: not valid JSON ({exc})"], []
    except UnicodeDecodeError as exc:
        return [f"{rel}: not valid UTF-8 ({exc})"], []

    if nb.get("nbformat") != 4:
        errors.append(f"{rel}: nbformat is {nb.get('nbformat')!r}, expected 4")

    cells = nb.get("cells")
    if not isinstance(cells, list):
        return [f"{rel}: missing or malformed 'cells' list"], warnings
    if not cells:
        warnings.append(f"{rel}: notebook has no cells")

    last_count = 0
    out_of_order = False
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict) or "cell_type" not in cell:
            errors.append(f"{rel}: cell {i} is malformed")
            continue
        if cell["cell_type"] != "code":
            continue

        source = "".join(cell.get("source", []))

        # Generated notebooks can carry an escaping slip that only shows at
        # execution time, after the expensive cells above have already run.
        try:
            compile(source, f"{rel}:cell{i}", "exec")
        except SyntaxError as exc:
            errors.append(f"{rel}: cell {i} is not valid Python ({exc.msg}, line {exc.lineno})")
        if "QUICK_TEST = True" in source.replace("QUICK_TEST=True", "QUICK_TEST = True"):
            warnings.append(f"{rel}: cell {i} has QUICK_TEST enabled -- degraded numbers")

        count = cell.get("execution_count")
        if isinstance(count, int):
            if count < last_count:
                out_of_order = True
            last_count = count

    if out_of_order:
        warnings.append(f"{rel}: execution counts are out of order -- restart & run all")

    return errors, warnings


def main() -> int:
    notebooks = sorted({path for glob in NOTEBOOK_GLOBS for path in REPO_ROOT.glob(glob)})
    notebooks = [
        p for p in notebooks if ".ipynb_checkpoints" not in p.parts and ".venv" not in p.parts
    ]

    if not notebooks:
        print("no notebooks found -- nothing to check")
        return 0

    all_errors: list[str] = []
    all_warnings: list[str] = []
    for path in notebooks:
        errors, warnings = check_notebook(path)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    for warning in all_warnings:
        print(f"WARN  {warning}")
    for error in all_errors:
        print(f"ERROR {error}")

    print(
        f"\nchecked {len(notebooks)} notebook(s): "
        f"{len(all_errors)} error(s), {len(all_warnings)} warning(s)"
    )
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
