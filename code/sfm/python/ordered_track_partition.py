"""Anchor-validated ordered track partition; distinct from the existing lifecycle pipeline."""
import numpy as np
from scipy.optimize import least_squares
from geometry import project_replica

def pack(m,obs):
    R=np.array([m['cameras'][o['image']]['r'] for o in obs]);C=np.array([m['cameras'][o['image']]['c'] for o in obs]);uv=np.array([o['uv'] for o in obs])
    cache=m.setdefault('_track_lifecycle_normalized',{})
    keys=[(o['image'],*map(float,o['uv'])) for o in obs]
    missing=[j for j,k in enumerate(keys) if k not in cache]
    if missing:
        if 'calibration' in m:  # This native model shares one calibrated sensor.
            rays=m['unproject'](obs[0]['image'],uv[missing])
            for j,q in zip(missing,rays):cache[keys[j]]=q
        else:
            for j in missing:cache[keys[j]]=m['unproject'](obs[j]['image'],uv[j])[0]
    q=np.array([cache[k] for k in keys])
    def residual(x):
        if 'calibration' in m:
            v=(R@(x-C)[:,:,None])[:,:,0]
            return (project_replica(v[:,:2]/v[:,2,None],m['calibration'])-uv).ravel()
        return np.concatenate([m['project'](o['image'],x)[0]-o['uv'] for o in obs])
    return R,C,q,residual

def fit(m,obs):
    R,C,q,residual=pack(m,obs)
    p=np.concatenate([R,(-(R@C[:,:,None]))],axis=2)
    A=np.stack([q[:,0,None]*p[:,2]-p[:,0],q[:,1,None]*p[:,2]-p[:,1]],axis=1).reshape(-1,4)
    _,_,v=np.linalg.svd(A);x=v[-1,:3]/v[-1,3]
    x=least_squares(residual,x,loss='soft_l1',f_scale=2,max_nfev=60).x
    assert np.isfinite(x).all() and all((m['cameras'][o['image']]['r']@(x-m['cameras'][o['image']]['c']))[2]>0 for o in obs)
    rays=np.array([(x-m['cameras'][o['image']]['c'])/np.linalg.norm(x-m['cameras'][o['image']]['c']) for o in obs]);angle=np.degrees(np.arccos(np.clip(np.min(rays@rays.T),-1,1)))
    assert angle>=2.
    return x

def sse(m,obs,x):
    if not obs:return 0.
    _,_,_,residual=pack(m,obs);e=residual(x);return float(e@e)

def split(m,obs,x):
    n=len(obs)
    if n<8:return None
    base=sse(m,obs,x);penalty=2.25*(3*np.log(2*n)+2*np.log(3));best=None
    for k in sorted({n//3,n//2,2*n//3}):
        if min(k,n-k)<4:continue
        try:
            xa=fit(m,obs[:k]);xb=fit(m,obs[k:]);after=sse(m,obs[:k],xa)+sse(m,obs[k:],xb)
            if after>=.5*base or base-after<=penalty:continue
            cv1=cv2=0.
            for parity in (0,1):
                full=fit(m,obs[parity::2]);ia=list(range(parity,n,2));qa=[i for i in range(n) if i not in ia]
                left=fit(m,[obs[i] for i in ia if i<k]);right=fit(m,[obs[i] for i in ia if i>=k])
                cv1+=sse(m,[obs[i] for i in qa],full);cv2+=sse(m,[obs[i] for i in qa if i<k],left)+sse(m,[obs[i] for i in qa if i>=k],right)
            if cv2>=.75*cv1:continue
            if best is None or after<best['sse_after']:best=dict(cut=k,boundary_camera=(obs[k-1]['image']+obs[k]['image'])/2,left=xa.tolist(),right=xb.tolist(),sse_before=base,sse_after=after,cv_before=cv1,cv_after=cv2)
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
    return best
