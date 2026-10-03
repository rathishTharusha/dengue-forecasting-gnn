"""Stage S8: Seroprevalence Validation.

Compares the implied cumulative infected fraction at the survey midpoint against
the nine-district IgG seroprevalence survey.

The survey figures are read from ``data/external/seroprevalence_nine_districts.csv``
(Jeewandara et al., J Med Virol 2024, sampled 2022-09 to 2023-03), using the
``10-20 (all)`` age row for each district. Rule R4: the survey post-dates most of
the study period, so this stage is validation only and never feeds a forecast.

Outputs: analysis/results/seir_gnn/s8_seroprevalence/s8_seroprevalence_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "analysis" / "lib"))
sys.path.insert(0, str(REPO / "analysis" / "_build"))
sys.path.insert(0, str(REPO / "src"))

import corrected_data as cd  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s5-results", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s5_seir_gnn" / "s5_seir_gnn_results.json"))
    ap.add_argument("--out", default=str(REPO / "analysis" / "results" / "seir_gnn" / "s8_seroprevalence" / "s8_seroprevalence_results.json"))
    ap.add_argument("--survey-week", type=int, default=496,
                    help="week index of the survey midpoint (496 = 2022-12-17)")
    args = ap.parse_args()

    data = cd.load()
    cases = data.cases
    pop = data.population[-1] # Most recent population
    rho = 1.0 / 11.0
    districts = data.names

    # The survey ran 2022-09 to 2023-03; week 496 is its midpoint, 2022-12-17.
    week = int(args.survey_week)
    cum_cases = np.nansum(cases[:week], axis=0)
    implied_infected_frac = np.clip(cum_cases / (rho * pop), 0.0, 1.0)

    # The survey itself, not a stand-in for it: nine districts, all-ages row.
    survey_path = REPO / "data" / "external" / "seroprevalence_nine_districts.csv"
    survey_df = pd.read_csv(survey_path)
    survey_df = survey_df[survey_df["age_group"] == "10-20 (all)"]
    survey_data = {
        row.district: float(row.seroprevalence_pct) / 100.0
        for row in survey_df.itertuples()
    }
    unknown = sorted(set(survey_data) - set(districts))
    if unknown:
        raise SystemExit(f"survey districts not in the case series: {unknown}")

    results_rows = []
    model_vals = []
    survey_vals = []

    for idx, d_name in enumerate(districts):
        if d_name in survey_data:
            s_val = survey_data[d_name]
            m_val = float(implied_infected_frac[idx])
            err = m_val - s_val
            model_vals.append(m_val)
            survey_vals.append(s_val)
            results_rows.append({
                "district": d_name,
                "model_implied_frac": round(m_val, 4),
                "survey_seroprevalence": round(s_val, 4),
                "error": round(err, 4),
                "sign": "+" if err > 0 else "-"
            })

    if len(model_vals) > 1:
        corr, pval = spearmanr(model_vals, survey_vals)
    else:
        corr, pval = 0.0, 1.0

    output = {
        "spearman_rho": round(float(corr), 4),
        "p_value": round(float(pval), 4),
        "n_districts": len(results_rows),
        "survey_week_index": week,
        "survey_source": str(survey_path.relative_to(REPO)).replace("\\", "/"),
        "district_comparison": results_rows,
        "note": ("Descriptive comparison between implied cumulative infection since 2013 "
                 "under rho = 1/11 and the surveyed IgG seroprevalence. The two measure "
                 "different things: the survey covers ages 10-20 and lifetime exposure, "
                 "the model tracks the whole population from 2013 onwards, so only the "
                 "ranking across districts is meaningful.")
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("=== Stage S8 Seroprevalence Comparison ===")
    print(f"Spearman rho: {corr:.4f} (p={pval:.4f})")
    print(pd.DataFrame(results_rows).to_string(index=False))
    print(f"Wrote S8 results to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
