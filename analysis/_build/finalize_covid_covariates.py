"""Finish report exports after all 81 COVID runs; no training or tuning."""
from pathlib import Path
import itertools
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis/results/covid_covariates'

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    OUT=parser.parse_args().out
    deadline = time.monotonic() + 24 * 3600
    while time.monotonic() < deadline:
        try:
            rows = json.loads((OUT / 'runs.json').read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            rows = []
        if len(rows) == 81 and (OUT / 'report.md').exists():
            break
        time.sleep(50)
    else:
        (OUT / 'finalization_status.txt').write_text('Incomplete after 24 hours; inspect runner output. No full-result claim is supported.', encoding='utf-8')
        return 1
    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows)
    assert len(df.drop_duplicates(['arm', 'origin', 'seed'])) == 81
    assert set(df.arm) == {'base', 'policy', 'policy_mobility'}
    base = df[df.arm.eq('base')].set_index(['origin','seed'])
    summary = []
    rng = np.random.default_rng(20261002)
    for arm, g in df.groupby('arm'):
        g = g.set_index(['origin','seed'])
        delta = (g.test_RMSE-base.test_RMSE).groupby(level='origin').mean().to_numpy()
        assert len(delta) == 9
        ties = np.abs(delta) < 1e-4
        adjusted = delta.copy()
        adjusted[ties] = 0
        signs = np.array(list(itertools.product([-1,1], repeat=9)))
        p = float((np.abs((signs*adjusted).mean(1)) >= abs(adjusted.mean())-1e-12).mean())
        boot = rng.choice(delta, size=(10000,9), replace=True).mean(1)
        lo, hi = np.quantile(boot,[.025,.975])
        summary.append(dict(arm=arm, val_RMSE=g.val_RMSE.mean(),test_RMSE=g.test_RMSE.mean(),
                            test_MAE=g.test_MAE.mean(),bias=g.bias.mean(),
                            delta_vs_base=delta.mean(), origins_won=int((adjusted<0).sum()),
                            origins_tied=int(ties.sum()),p_vs_base=p,
                            delta_ci_lower=lo, delta_ci_upper=hi))
    result = pd.DataFrame(summary)
    ordered = result.index[result.arm.ne('base')].tolist()
    ordered.sort(key=lambda k:result.loc[k,'p_vs_base'])
    running = 0.
    for rank,k in enumerate(ordered):
        running=max(running,min(1.,(2-rank)*result.loc[k,'p_vs_base']))
        result.loc[k,'p_holm']=running
    result.to_csv(OUT/'summary.csv',index=False)
    per = df.groupby(['arm','origin'])[['test_RMSE','test_MAE','bias','pers_RMSE',
        'train_policy_windows','train_mobility_windows','test_policy_windows','test_mobility_windows']].mean()
    per.to_csv(OUT/'per_origin.csv')
    text = '# COVID policy and mobility: completed exploratory test\n\n'
    text += '81 matched evaluations, nine origins and three seeds. Scores are means of split metrics, not globally pooled RMSE.\n\n'
    text += '```text\n'+result.to_string(index=False)+'\n```\n\n'
    text += 'Origin-clustered exact sign flips; Holm adjustment for the two versus-base comparisons. Differences below 0.0001 RMSE count as numerical ties. Confidence intervals resample the nine origins, averaging seeds first; these nine-origin intervals have limited precision.\n\n'
    text += '```text\n'+per.to_string()+'\n```\n\n'
    text += 'Archived policy/mobility vintages are unverified. Latest external row is origin minus two weeks. Missing external data have neutral model effects. Training exposure counts do not establish variation or identification. No lockdown effect can be learned from an all-zero training exposure. This is an exploratory follow-up after previous searches, not independent confirmation or evidence of a causal policy effect.\n\n'
    text += 'The source-manifest boundary mismatch concerns an unused file. Checked tracked model inputs match HEAD (input_audit.json). The resume serialization fix changed the scheduler check only, with training definitions verified unchanged (resume_fix.json).\n'
    if 'device' in df:
        assert df.groupby('origin').device.nunique().eq(1).all(), 'Each origin must use one backend for all arms and seeds'
        devices=df.groupby('origin').device.first()
        devices.to_csv(OUT/'devices_by_origin.csv')
        text+='\nHardware: '+devices.to_string()+'\nAll arms and seeds within each origin use the same backend; CPU/CUDA training need not be bit-identical.\n'
    (OUT/'report.md').write_text(text,encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'analysis/_build/summarize_covid_horizons.py'),'--out',str(OUT)],check=True,cwd=ROOT)
    (OUT/'finalization_status.txt').write_text('Complete: 81 unique rows, matched predictions, corrected summary and per-horizon metrics exported.',encoding='utf-8')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
