"""Reproduce the completed audit findings; never trains or reads new-period cases."""
import json,subprocess,hashlib
import numpy as np
from common import REPO,RES,LEGACY,KAGGLE

def holm(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p);out=np.empty(len(p));running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(p)-rank)*p[i]));out[i]=running
    return out.tolist()

def main():
    c=np.load(LEGACY)[:,:,5].astype(float);ids=list(range(3,len(c)-3));folds=[]
    for o in [.55,.70,.85]:
        te=ids[int(o*len(ids)):int(min(o+.15,1)*len(ids))]
        err=np.stack([np.repeat(c[i-1,:,None],3,axis=1)-c[i:i+3].T for i in te])
        touch=np.array([395 in range(i-3,i+3) for i in te])
        folds.append({'origin':o,'all_windows':float(np.sqrt(np.mean(err**2))),
                      'excluded_windows':float(np.sqrt(np.mean(err[~touch]**2))),
                      'n_test':len(te),'n_excluded':int(touch.sum())})
    floor={k:float(np.mean([r[k] for r in folds])) for k in ['all_windows','excluded_windows']}
    expected=json.loads((KAGGLE/'results.json').read_text())['array_floor']
    assert np.isclose(floor['all_windows'],expected['all_windows'])
    assert np.isclose(floor['excluded_windows'],expected['without_row_395'])
    old=json.loads((REPO/'notebooks/baseline/sri_lanka_adj_list.json').read_text())
    def gitjson(path):return json.loads(subprocess.check_output(['git','show','3878e70:'+path],cwd=REPO,text=True))
    new=gitjson('notebooks/baseline/sri_lanka_adj_list.json')
    borders=gitjson('data/external/district_borders_gadm41.json')
    edges=lambda a:{(x,y) for x,ys in a.items() for y in ys}
    oe,ne=edges(old),edges(new);truth={frozenset(x) for x in borders['shared_borders']}
    assert {frozenset(x) for x in ne}==truth and all((b,a) in ne for a,b in ne)
    result={'legacy_floor':floor,'folds':folds,'excluded_window_rule':'395 in range(start-3,start+3); mean of three origin RMSEs',
            'relative_reduction':1-floor['excluded_windows']/floor['all_windows'],
            'graph':{'corrected_commit':'3878e70','shared_borders':len(truth),'old_directed_neighbors':len(oe),
                     'corrected_directed_neighbors':len(ne),'loader_self_loops':len(new),
                     'removed':sorted(oe-ne),'added':sorted(ne-oe),'matches_stored_gadm_borders':True},
            'legacy_sha256':hashlib.sha256(LEGACY.read_bytes()).hexdigest()}
    (RES/'review_audit_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
