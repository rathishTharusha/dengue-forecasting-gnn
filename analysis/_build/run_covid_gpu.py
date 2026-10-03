"""Validate and benchmark CUDA without touching the CPU experiment."""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import inspect
import hashlib
import json
import sys
import types
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'analysis/_build'))
import run_covid_covariates as c


def make_jobs(out, epochs):
    data=c.v2.cd.load()
    cases,adj,artifact,missing,_=c.v2.rcb.prepare('rebuilt')
    frame=pd.read_csv(ROOT/'data/external/covid_response_weekly.csv',parse_dates=['week_start'])
    external,available=c.external_inputs(frame,data.week_start)
    jobs=[]
    for fold in c.v2.build_folds(cases,missing,'9origin'):
        f=c.v2.features_train_only(data,'cases',fold)
        ex=np.broadcast_to(external[:,None,None,:],(*f.shape[:3],2))
        features=np.concatenate([f,ex],axis=-1).copy()
        for arm in c.ARMS:
            for seed in (0,1,2):
                cfg=dict(c.v2.ARMS['v2']); cfg['epochs']=epochs
                jobs.append(dict(arm=arm,cfg=cfg,arch='STGAT',seed=seed,dataset='rebuilt',
                    artifact=artifact,coupling='implicit',origin=fold.origin,
                    train_idx=np.asarray(fold.train_index),val_idx=np.asarray(fold.val_index),
                    test_idx=np.asarray(fold.test_index),cases=cases,features=features,
                    population=data.population,edge_index=np.stack(np.nonzero(adj)),
                    adj_dense=adj,available=available,out_dir=str(out)))
    return jobs


def install_gpu_runner():
    # Clone the simulator: only newly-created tensor device placement changes.
    sim=types.ModuleType('covid_cuda_simulator')
    text=inspect.getsource(c.seir_sim)
    text=text.replace('dtype=state.dtype)', 'dtype=state.dtype, device=state.device)')
    exec(compile(text,str(ROOT/'analysis/lib/seir_sim.py'),'exec'),sim.__dict__)
    c.seir_sim=sim
    source=inspect.getsource(c.v2.run_job)
    source=source.replace('dtype=torch.float32)', 'dtype=torch.float32, device=device)')
    source=source.replace('dtype=torch.long)', 'dtype=torch.long, device=device)')
    assert '    if cfg.get("const"):' in source
    source=source.replace('    if cfg.get("const"):', '    model = model.to(device)\n    if cfg.get("const"):')
    before='return feat_t[idx], st0, pop, y, s_, o_i, o_e'
    assert before in source
    source=source.replace(before,'return feat_t[idx], *(t.to(device) for t in (st0, pop, y, s_, o_i, o_e))')
    source=source.replace('keep = torch.tensor(~np.asarray(art, dtype=bool))',
                          'keep = torch.tensor(~np.asarray(art, dtype=bool), device=device)')
    namespace=dict(c.v2.__dict__)
    namespace.update(device=torch.device('cuda'),SEIRGNNv2=c.CovidModel)
    exec(compile(source,'<covid_cuda_training>','exec'),namespace)
    return namespace['run_job']


def validate_forward(job):
    torch.manual_seed(0)
    c.ACTIVE_ARM='policy_mobility'
    edge=torch.tensor(job['edge_index'],dtype=torch.long)
    adj=torch.tensor(job['adj_dense'],dtype=torch.float32)
    cpu=c.CovidModel('STGAT',3,edge,adj,mod_clamp=.2)
    cpu.covid_weight.data[:]=torch.tensor([-.2,.1])
    gpu=copy.deepcopy(cpu).cuda()
    gpu.edge_index=gpu.edge_index.cuda()
    cpu.eval(); gpu.eval()
    idx=job['test_idx'][:2]
    _,s,oi,oe,pop,y=c.v2.initial_state(job['cases'],job['population'],idx,True,0.)
    st=cpu.build_state(s,oi,oe)
    x=torch.tensor(job['features'][idx])
    with torch.no_grad():
        a=cpu.to_cases(cpu(x,st),pop)
        b=gpu.to_cases(gpu(x.cuda(),st.cuda()),pop.cuda()).cpu()
    # Compare reported cases, not tiny compartment fractions.
    torch.testing.assert_close(a,b,rtol=1e-4,atol=.05)
    return float((a-b).abs().max())


GPU_FUNCTION = None


def run_gpu_job(job):
    global GPU_FUNCTION
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    if GPU_FUNCTION is None:
        GPU_FUNCTION=install_gpu_runner()
    c.v2.run_job=GPU_FUNCTION
    rec=c.run_job(job)
    rec['device']='cuda'
    torch.cuda.empty_cache()
    return rec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train',action='store_true')
    ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--origins',nargs='+',type=float)
    ap.add_argument('--out',type=Path,default=ROOT/'analysis/results/covid_cuda_benchmark')
    args=ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: run with PYTHONPATH pointing at .gpu_runtime')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    args.out.mkdir(parents=True,exist_ok=True)
    original=c.v2.run_job
    gpu_run=install_gpu_runner()
    jobs=make_jobs(args.out,400 if args.train else 20)
    error=validate_forward(jobs[-9])
    print('GPU:',torch.cuda.get_device_name(0),'forward max difference:',error,flush=True)
    if not args.train:
        job=jobs[-9]
        warm=copy.copy(job); warm['cfg']=dict(job['cfg'],epochs=2)
        c.v2.run_job=gpu_run
        c.run_job(warm)
        c.v2.run_job=original
        cpu=c.run_job(job)
        c.v2.run_job=gpu_run
        gpu=c.run_job(job)
        result=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                    epochs=20,origin=job['origin'],cpu_seconds=cpu['elapsed'],
                    gpu_seconds=gpu['elapsed'],forward_max_case_error=error,
                    peak_gpu_memory_mb=torch.cuda.max_memory_allocated()/1024**2,
                    scientific_result=False)
        (args.out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
        return
    if not args.origins:
        raise SystemExit('--train requires explicit whole origins, all arms and seeds')
    jobs=[j for j in jobs if j['origin'] in args.origins]
    assert len(jobs)==9*len(set(args.origins))
    c.v2.run_job=gpu_run
    records_path=args.out/'runs.json'
    records=json.loads(records_path.read_text()) if records_path.exists() else []
    completed={(r['arm'],r['origin'],r['seed']) for r in records}
    config=dict(device='cuda',gpu=torch.cuda.get_device_name(0),torch=torch.__version__,
                origins=args.origins,arms=c.ARMS,seeds=[0,1,2],epochs=400,
                device_change_only=True,original_cpu_results_preserved=True)
    config['cpu_runner_sha256']=hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()
    config['gpu_runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['covid_source_sha256']=hashlib.sha256((ROOT/'data/external/covid_response_weekly.csv').read_bytes()).hexdigest()
    config_path=args.out/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text())==json.loads(json.dumps(config)), 'GPU resume config differs; use a new output directory'
    else:
        config_path.write_text(json.dumps(config,indent=2),encoding='utf-8')
    pending=[j for j in jobs if (j['arm'],j['origin'],j['seed']) not in completed]
    print(f'{len(pending)} remaining GPU jobs; {args.workers} workers',flush=True)
    torch.cuda.empty_cache()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_gpu_job,j) for j in pending]):
            records.append(future.result())
            records_path.write_text(json.dumps(records,indent=2),encoding='utf-8')
            pd.DataFrame(records).to_csv(args.out/'runs.csv',index=False)
            print('Saved GPU runs:',len(records),flush=True)

if __name__=='__main__':
    main()
