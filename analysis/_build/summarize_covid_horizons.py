"""Export per-horizon metrics from the frozen COVID experiment's predictions."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=ROOT/'analysis/results/covid_covariates')
OUT = parser.parse_args().out
rows = []
for path in sorted(OUT.glob('*.npz')):
    arm, tail = path.stem.rsplit('_o', 1)
    origin, seed = tail.split('_s')
    with np.load(path) as pack:
        pred, truth, pers = (pack[k] for k in ('prediction', 'truth', 'persistence'))
        for h in range(truth.shape[-1]):
            for label, values in ((arm, pred), ('persistence', pers)):
                err = values[..., h] - truth[..., h]
                denom = np.abs(values[..., h]) + np.abs(truth[..., h])
                smape = np.divide(200 * np.abs(err), denom,
                                  out=np.zeros_like(err), where=denom > 0)
                rows.append(dict(arm=label, paired_with=arm, origin=float(origin),
                                 seed=int(seed), horizon=h + 1,
                                 n_windows=len(pack['origin_index']),
                                 RMSE=float(np.sqrt(np.mean(err ** 2))),
                                 MAE=float(np.mean(np.abs(err))),
                                 bias=float(err.mean()), SMAPE=float(smape.mean())))
pd.DataFrame(rows).to_csv(OUT / 'per_horizon.csv', index=False)
print(f'Wrote {len(rows)} per-horizon rows')
