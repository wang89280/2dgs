"""Reused within-scene feature holdout; not independent geometric ground truth."""
import numpy as np
from scipy.optimize import least_squares
from evaluate import stats
def prediction(m,tracks):
    errors=[];failure=0;rows=[]
    for tid,obs in tracks.items():
        anchors=obs[::2];queries=obs[1::2];a=[]
        try:
            for o in anchors:
                i=o['image'];c=m['cameras'][i];p=np.column_stack((c['r'],-c['r']@c['c']));q=m['unproject'](i,o['uv'])[0];a.extend([q[0]*p[2]-p[0],q[1]*p[2]-p[1]])
            _,_,v=np.linalg.svd(a);x=v[-1,:3]/v[-1,3]
            def residual(x):return np.concatenate([m['project'](o['image'],x)[0]-o['uv'] for o in anchors])
            x=least_squares(residual,x,loss='soft_l1',f_scale=2,max_nfev=60).x
            if any((m['cameras'][o['image']]['r']@(x-m['cameras'][o['image']]['c']))[2]<=0 for o in anchors):raise ValueError('anchor behind camera')
        except (KeyError,ValueError,np.linalg.LinAlgError):
            failure+=len(queries);rows.extend(dict(track=int(tid),camera=o['image'],error_px=None) for o in queries);continue
        for o in queries:
            try:
                i=o['image'];c=m['cameras'][i]
                if (c['r']@(x-c['c']))[2]<=0:raise ValueError('query behind camera')
                e=float(np.linalg.norm(m['project'](i,x)[0]-o['uv']))
                if not np.isfinite(e):raise ValueError('nonfinite query')
                errors.append(e);rows.append(dict(track=int(tid),camera=i,error_px=e))
            except (KeyError,ValueError):failure+=1;rows.append(dict(track=int(tid),camera=o['image'],error_px=None))
    total=len(errors)+failure
    return dict(queries=total,failed=failure,finite_fraction=len(errors)/total,finite_prediction_px=stats(errors),fraction_within_3px=sum(e<=3 for e in errors)/total,capped_mean_20px=(sum(min(20,e) for e in errors)+failure*20)/total,rows=rows)

def summary(p):return {k:v for k,v in p.items() if k!='rows'}

def paired(a,b):
    assert [(r['track'],r['camera']) for r in a['rows']]==[(r['track'],r['camera']) for r in b['rows']], 'Unpaired query identities'
    ga={};gb={}
    for rows,g in [(a['rows'],ga),(b['rows'],gb)]:
        for r in rows:g.setdefault(r['track'],[]).append(20 if r['error_px'] is None else min(20,r['error_px']))
    ids=sorted(ga);assert ids==sorted(gb)
    n=np.array([len(ga[i]) for i in ids]);assert all(len(ga[i])==len(gb[i]) for i in ids)
    d=np.array([sum(ga[i])-sum(gb[i]) for i in ids]);rng=np.random.default_rng(24002);bs=[]
    for _ in range(2000):q=rng.integers(0,len(ids),len(ids));bs.append(d[q].sum()/n[q].sum())
    return dict(difference=float(d.sum()/n.sum()),ci95=np.quantile(bs,[.025,.975]).tolist())
