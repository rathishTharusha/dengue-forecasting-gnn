"""Self-check for main.tex.

1. every decimal or percentage number in a sentence is found (after rounding) in one of the
   evidence rows named in that line's trailing comment;
2. every sentence line with such a number carries an EV comment, and every cited EV id exists;
3. banned words and em dashes (STYLE_GUIDE.md);
4. unresolved \\ref / \\cite are reported by LaTeX itself (see main.log), checked here too.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

IEEE = Path(__file__).resolve().parents[1]
TEX = (IEEE / "main.tex").read_text(encoding="utf-8")
EVID = (IEEE.parent / "EVIDENCE.md").read_text(encoding="utf-8")

BANNED = ["delve", "leverage", "utilize", "harness", "novel", "groundbreaking", "cutting-edge",
          "state-of-the-art", "robust", "seamless", "comprehensive", "holistic", "pivotal", "crucial",
          "vital role", "paramount", "intricate", "nuanced", "landscape", "realm", "tapestry", "paradigm",
          "synergy", "unlock", "empower", "showcase", "underscore", "shed light on", "pave the way",
          "a testament to", "in today's world", "in recent years", "it is worth noting",
          "it is important to note", "notably", "furthermore", "moreover", "plays a key role"]


def evidence_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in EVID.splitlines():
        m = re.match(r"\|\s*(EV-\d+)\s*\|", line)
        if m:
            rows[m.group(1)] = line
    return rows


def numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", text.replace(",", " "))]


def close(value: float, shown: str) -> bool:
    d = len(shown.split(".")[1]) if "." in shown else 0
    target = float(shown)
    for v in (value, abs(value), value * 100, abs(value) * 100):
        if round(v, d) == target or abs(abs(v) - abs(target)) < 0.5 * 10 ** (-d) + 1e-9:
            return True
    return False


def main() -> int:
    rows = evidence_rows()
    bad = 0
    for n, raw in enumerate(TEX.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("%") or "\\includegraphics" in line or line.startswith("\\label"):
            continue
        body, _, comment = re.sub(r"(?<!\\)%", "\x00", line).partition("\x00")
        if "&" in body or body.startswith("\\"):
            continue                      # equation and table lines are checked by hand
        tags = re.findall(r"EV-\d+", comment)
        # numeric tokens: decimals and percentages in prose (skip math-only equation lines)
        prose = re.sub(r"\$[^$]*\$", " ", body)
        prose = re.sub(r"\\[A-Za-z]+\{[^}]*\}", " ", prose)
        toks = re.findall(r"\d+\.\d+|\d+\\%", prose)
        rest = re.sub(r"\d+\.\d+", " ", prose)
        toks += [t for t in re.findall(r"\b\d{3,}\b", rest) if not t.startswith("20")]
        if toks and not tags:
            print(f"line {n}: numbers {toks} but no EV tag: {line[:90]}")
            bad += 1
            continue
        for t in tags:
            if t not in rows:
                print(f"line {n}: unknown evidence id {t}")
                bad += 1
        pool: list[float] = []
        for t in tags:
            pool += numbers(rows.get(t, ""))
        for tok in toks:
            shown = tok.replace("\\%", "")
            if not any(close(v, shown) for v in pool):
                print(f"line {n}: {shown} not found in {tags}: {line[:80]}")
                bad += 1
    low = TEX.lower()
    for w in BANNED:
        for m in re.finditer(r"(?<![a-z])" + re.escape(w), low):
            line_no = low[: m.start()].count("\n") + 1
            print(f"banned '{w}' at line {line_no}")
            bad += 1
    for m in re.finditer("\u2014|---", TEX):
        print("em dash at line", TEX[: m.start()].count("\n") + 1)
        bad += 1
    for m in re.finditer(r"(?<![.!?])\n\s*Additionally", TEX):
        print("'Additionally' at sentence start, line", TEX[: m.start()].count("\n") + 2)
        bad += 1
    print("problems:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
