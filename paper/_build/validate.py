"""Static checks on the paper sources, for machines without a TeX toolchain.

The build runs in Docker (see ``paper/Makefile``) because six machines with six
TeX distributions produce six different PDFs. That is the right call and it means
most contributors cannot compile locally, so the errors a compile would have
caught -- an undefined macro, a citation key that is not in the bibliography, a
figure that was never generated -- surface only for whoever runs the container.

These checks catch that class of error in a second and with no dependencies:

* brace balance per file, and matched ``\\begin``/``\\end`` environments;
* every ``\\cite`` key present in ``refs.bib``;
* every ``\\ref`` resolved by a ``\\label``;
* every ``\\includegraphics`` target present on disk;
* every custom macro used is defined in ``main.tex``.

They do **not** replace a compile. Page count, float placement and overfull boxes
still need the container.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent
SECTIONS = sorted((PAPER / "sections").glob("*.tex"))
FILES = [PAPER / "main.tex", *SECTIONS, PAPER / "tables_generated.tex"]

#: acmart and the standard packages provide these; only project macros are checked.
KNOWN_PREFIXES = ("begin", "end", "section", "subsection", "paragraph", "label",
                  "ref", "cite", "includegraphics", "caption", "centering", "input",
                  "textbf", "emph", "texttt", "text", "item", "itemsep", "toprule",
                  "midrule", "bottomrule", "multicolumn", "small", "linewidth",
                  "newcommand", "renewcommand", "newif", "usepackage", "definecolor",
                  "documentclass", "author", "affiliation", "institution", "city",
                  "country", "email", "title", "keywords", "maketitle", "balance",
                  "bibliographystyle", "bibliography", "setlength", "settopmatter",
                  "setcopyright", "pagestyle", "footnotetextcopyrightpermission",
                  "let", "undefined", "ifdefined", "else", "fi", "sethlcolor", "hl",
                  "todo", "quad", "times", "approx", "rightarrow", "ge", "le",
                  "sum", "log", "exp", "mathbb", "mathbf", "mathrm", "hat", "top",
                  "sim", "S", "%", "&", "_", "#", "$", "{", "}", "\\", " ")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    problems: list[str] = []
    bodies = {p: strip_comments(read(p)) for p in FILES}

    # --- braces and environments
    for path, body in bodies.items():
        opens = len(re.findall(r"(?<!\\)\{", body))
        closes = len(re.findall(r"(?<!\\)\}", body))
        if opens != closes:
            problems.append(f"{path.name}: {opens} '{{' vs {closes} '}}'")
        begins = re.findall(r"\\begin\{(\w+\*?)\}", body)
        ends = re.findall(r"\\end\{(\w+\*?)\}", body)
        for env in set(begins) | set(ends):
            if begins.count(env) != ends.count(env):
                problems.append(f"{path.name}: {begins.count(env)} \\begin{{{env}}} "
                                f"vs {ends.count(env)} \\end{{{env}}}")

    joined = "\n".join(bodies.values())

    # --- citations
    keys = set(re.findall(r"@\w+\{([^,]+),", read(PAPER / "refs.bib")))
    cited = {k.strip() for group in re.findall(r"\\cite\{([^}]*)\}", joined)
             for k in group.split(",")}
    for key in sorted(cited - keys):
        problems.append(f"citation not in refs.bib: {key}")

    # --- labels and refs
    labels = set(re.findall(r"\\label\{([^}]*)\}", joined))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", joined))
    for key in sorted(refs - labels):
        problems.append(f"\\ref with no \\label: {key}")

    # --- figures
    for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", joined):
        if not any((PAPER / f"{target}{ext}").exists() for ext in (".pdf", ".png", "")):
            problems.append(f"missing figure: {target}")

    # --- project macros
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", read(PAPER / "main.tex")))
    used = set(re.findall(r"\\([A-Za-z]+)", joined))
    unknown = {m for m in used
               if m not in defined and not m.startswith(KNOWN_PREFIXES)}
    # Anything still unresolved is very likely a real typo in a project macro.
    for macro in sorted(m for m in unknown if m in {"modelname", "Aadp", "Afix",
                                                    "Ablend", "Ldata", "Lreg",
                                                    "Lcons", "Lsmooth", "Rhat", "R"}):
        if macro not in defined:
            problems.append(f"project macro used but not defined: \\{macro}")

    words = len(re.findall(r"\b[A-Za-z][A-Za-z'-]+\b",
                           re.sub(r"\\[A-Za-z]+|\{|\}", " ",
                                  "\n".join(bodies[p] for p in SECTIONS))))
    print(f"sections: {len(SECTIONS)}   body words: ~{words}")
    print(f"citations used: {len(cited)}   labels: {len(labels)}   refs: {len(refs)}")
    figures = sorted((PAPER / "figures").glob("*.pdf"))
    print(f"figures on disk: {', '.join(f.stem for f in figures)}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall static checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
