"""Copy the approved reference entries (PAPER_PLAN.md section 7) from full_paper/overleaf/refs.bib."""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "full_paper" / "overleaf" / "refs.bib"
OUT = Path(__file__).resolve().parents[1] / "refs.bib"
KEYS = ["tissera2020severe", "weng2024graph", "gulmohamed2026denguegnn", "liu2025seirlstm",
        "phaijoo2018sensitivity", "guo2019astgcn", "bai2021a3tgcn", "shi2019aagcn", "li2018dcrnn",
        "wu2019graphwavenet", "kipf2017gcn", "rodriguez2023einns", "wang2022causalgnn", "cao2023mepognn",
        "deng2020colagnn", "raissi2019pinn", "krishnapriyan2021failure", "yoon2019timegan", "kim2022revin",
        "shao2022stid", "salinas2020deepar", "bracher2021evaluating"]
entries = {m.group(2): e for e in re.split(r"\n(?=@)", SRC.read_text(encoding="utf-8"))
           if (m := re.match(r"@(\w+)\{([^,]+),", e))}
missing = [k for k in KEYS if k not in entries]
assert not missing, missing
OUT.write_text("\n\n".join(entries[k].strip() for k in KEYS) + "\n", encoding="utf-8")
print("wrote", len(KEYS), "entries")
