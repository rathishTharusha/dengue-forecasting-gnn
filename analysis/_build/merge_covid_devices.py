"""Merge fixed CPU origins with a complete, paired CUDA final origin."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
CPU=ROOT/'analysis/results/covid_covariates'
GPU=ROOT/'analysis/results/covid_covariates_gpu'
OUT=ROOT/'analysis/results/covid_covariates_final'

def main():
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:
            gpu=json.loads((GPU/'runs.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            gpu=[]
        if len(gpu)==9:
            break
        time.sleep(30)
    else:
        (OUT/'finalization_status.txt').write_text('Incomplete CUDA origin; inspect GPU stderr.log. No complete comparison available.',encoding='utf-8')
        return 1
    cpu=json.loads((OUT/'cpu_72_rows.json').read_text())
    assert len(cpu)==72
    assert {r['origin'] for r in gpu}=={.9}
    assert {r['arm'] for r in gpu}=={'base','policy','policy_mobility'}
    rows=cpu+gpu
    assert len({(r['arm'],r['origin'],r['seed']) for r in rows})==81
    for r in rows:
        source=GPU if r['device']=='cuda' else CPU
        name=f"{r['arm']}_o{r['origin']}_s{r['seed']}.npz"
        shutil.copyfile(source/name,OUT/name)
    (OUT/'runs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    config=dict(cpu_config=json.loads((CPU/'config.json').read_text()),
                gpu_config=json.loads((GPU/'config.json').read_text()),
                benchmark=json.loads((ROOT/'analysis/results/covid_cuda_benchmark/benchmark.json').read_text()),
                note='GPU selected on timing before its final-origin test scores; all arms/seeds at each origin use one backend. Original CPU results are retained separately. Hardware variation is a limitation; CPU/CUDA training need not match exactly.')
    (OUT/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    shutil.copyfile(CPU/'input_audit.json',OUT/'input_audit.json')
    shutil.copyfile(CPU/'resume_fix.json',OUT/'resume_fix.json')
    (OUT/'report.md').write_text('Pending automatic report export; 81 matched rows assembled.',encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'analysis/_build/finalize_covid_covariates.py'),'--out',str(OUT)],check=True,cwd=ROOT)
    print('Final report ready:',OUT/'report.md',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
