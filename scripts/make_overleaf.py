"""Assemble an Overleaf-ready bundle from paper/.

Overleaf needs a self-contained copy. That copy is *generated*, never hand-assembled:
a bundle maintained by copying files across drifts from the paper, which causes
regressions.

Regenerate after any change to paper/:

    python paper/_build/figures.py    # refresh figures and tables_generated.tex first
    python scripts/make_overleaf.py

Writes overleaf/ and overleaf.zip.
"""

from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER = REPO / "paper"
OUT = REPO / "overleaf"
ZIP = REPO / "overleaf.zip"

#: Files copied verbatim, relative to paper/.
REQUIRED = [
    "main.tex",
    "refs.bib",
    "ACM-Reference-Format.bst",
    "tables_generated.tex",
    "sections/00_abstract.tex",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_framework.tex",
    "sections/04_experimental_setup.tex",
    "sections/05_results.tex",
    "sections/06_discussion_conclusion.tex",
]

# Overleaf compiles one "main document" chosen in project settings. This stub
# gives the colour-coded build without duplicating a single line of text.
HIGHLIGHTED_STUB = r"""% Overleaf: set this as the main document (Menu -> Main document) to build the
% contribution-highlighted version. It defines one flag and defers entirely to
% main.tex -- there is only ever one copy of the text.
\def\HIGHLIGHTON{1}
\input{main.tex}
"""

README = """# {title}

Overleaf-ready bundle. **Generated — do not edit here.**

Edit `paper/` in the repository and regenerate:

```bash
python paper/_build/figures.py
python scripts/make_overleaf.py
```

Anything changed inside this folder is lost on the next regeneration.

## Uploading

1. Overleaf → **New Project → Upload Project** → select `overleaf.zip`.
2. Compiler is **pdfLaTeX** (Menu → Compiler). This is the default.
3. Main document is **`main.tex`**. This is the default.
4. Recompile. The bibliography needs two passes; Overleaf handles that itself.

## The two builds

| Main document | Produces |
|---|---|
| `main.tex` | clean version, for submission |
| `main-highlighted.tex` | contributions colour-coded per member, for grading |

Switch between them under **Menu → Main document**. Both read the same section
files, so the text exists once.

## Page limit

**4 pages excluding references.** At the time of generation the body ended on
page 4 with the bibliography spilling to page 5, which is compliant. Check the
body end page after any edit rather than counting words.

## Contents

```
{tree}
```

## Notes

- `tables_generated.tex` is written by `paper/_build/figures.py`. Do not retype values into it; regenerate.
- `acmart` is installed on Overleaf, so no `.cls` is bundled.
- `ACM-Reference-Format.bst` is included because the bundled version is what the repository builds against.
"""


def main() -> int:
    if not PAPER.is_dir():
        print(f"error: {PAPER} not found", file=sys.stderr)
        return 1

    missing = [f for f in REQUIRED if not (PAPER / f).exists()]
    if missing:
        print("error: required files missing from paper/:", file=sys.stderr)
        for f in missing:
            print(f"  {f}", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copied: list[str] = []
    for rel in REQUIRED:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / rel, dst)
        copied.append(rel)

    # Copy all PDF figures present in paper/figures/
    fig_dir = PAPER / "figures"
    if fig_dir.is_dir():
        for fig_pdf in sorted(fig_dir.glob("*.pdf")):
            rel = f"figures/{fig_pdf.name}"
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig_pdf, dst)
            copied.append(rel)

    (OUT / "main-highlighted.tex").write_text(HIGHLIGHTED_STUB, encoding="utf-8")
    copied.append("main-highlighted.tex")

    title = _title_of(PAPER / "main.tex")
    (OUT / "README.md").write_text(
        README.format(
            title=title,
            tree="\n".join(sorted(copied)),
        ),
        encoding="utf-8",
    )
    copied.append("README.md")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(OUT))

    print(f"wrote {OUT.relative_to(REPO)}/ ({len(copied)} files)")
    for rel in sorted(copied):
        print(f"    {rel}")
    print(f"wrote {ZIP.relative_to(REPO)} ({ZIP.stat().st_size / 1024:.0f} KB)")
    print("\nUpload overleaf.zip via Overleaf -> New Project -> Upload Project.")
    return 0


def _title_of(main_tex: Path) -> str:
    """Pull \\title{...} out of main.tex for the bundle README heading."""
    text = main_tex.read_text(encoding="utf-8")
    m = re.search(r"^\\title\{(.+?)\}", text, re.MULTILINE | re.DOTALL)
    if not m:
        return "Paper"
    return re.sub(r"\s+", " ", m.group(1).replace("\\\\", " ")).strip()


if __name__ == "__main__":
    raise SystemExit(main())
