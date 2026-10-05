"""Regenerate included tables from saved historical results without training."""
from common import TAB
from historical_tables import p3
from publication import main

if __name__ == "__main__":
    p3()
    path = TAB / "ablation_p3.tex"
    text = path.read_text(encoding="utf-8")
    text = "\n".join(line for line in text.splitlines() if not line.startswith("AAGCN, direct, NB + season") and not line.startswith("AAGCN, direct, squared error + season")) + "\n"
    text = text.replace("Indicative ablation on", "Historical development ablation on").replace("3 seeds, 9 runs per row", "3 seeds, 9 neural runs per row").replace("GCN, residual (graph control)", "GCN, residual (baseline GNN)")
    text = text.replace(r"Table~\ref{tab:main}", "the nine-origin table in the main manuscript")
    path.write_text(text, encoding="utf-8")
    # Historical p3 also writes the omitted negative-ablation table; preserve it
    # as an archive. The manuscript does not include or claim those values.
    (TAB / "protocols.tex").write_text(r"""\begin{table}[!ht]
\centering\small
\caption{Stored rebuilt-data development protocols. Each neural arm uses three seeds; persistence is deterministic. Both use 30 validation forecast starts. Test shares describe forecast starts, whose target weeks overlap partition boundaries.}
\label{tab:protocols}
\begin{tabular}{@{}lll@{}}\toprule
Protocol & Origin fractions & Test share \\\midrule
Nine-origin & 0.50--0.90, step 0.05 & 5\% \\
Three-origin & 0.55, 0.70, 0.85 & 15\% \\
\bottomrule\end{tabular}\end{table}
% EV-141
""", encoding="utf-8")
    main()
