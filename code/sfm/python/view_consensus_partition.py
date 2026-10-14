"""Geometric view-neighborhood track hypotheses, validated without query pixels.

This is an independent small multi-landmark experiment inspired by geometric
multi-model fitting, not an implementation of Progressive-X or TC-SfM. Intended
for approximately planar orbital capture. No measurement is moved or discarded.
"""
import numpy as np
from ordered_track_partition import pack, fit, sse

POLICY=dict(min_views=3,max_landmarks=3,proposal_finalists=3,min_angle_deg=2.,
            sigma_px=1.5,max_sse_ratio=.5,max_loo_ratio=.75,
            max_camera_plane_variance_ratio=.15,assignment='nearest angular anchor',
            label_refinement_iterations=3)

def view_angles(m):
    if '_view_consensus_angles' in m:return m['_view_consensus_angles']
    ids=sorted(m['cameras']);C=np.array([m['cameras'][i]['c'] for i in ids]);C-=C.mean(axis=0)
    _,s,v=np.linalg.svd(C,full_matrices=False)
    ratio=float(s[-1]**2/(s@s));assert ratio<=POLICY['max_camera_plane_variance_ratio'],'Non-orbital cameras: retain single-point model'
    q=C@v[:2].T;a=np.arctan2(q[:,1],q[:,0])
    m['_view_consensus_angles']={i:float(x) for i,x in zip(ids,a)}
    m['_view_consensus_plane_ratio']=ratio
    return m['_view_consensus_angles']

def order(m,obs):
    a=view_angles(m);q=sorted(obs,key=lambda o:a[o['image']]);theta=np.array([a[o['image']] for o in q])
    gaps=np.diff(np.r_[theta,theta[0]+2*np.pi]);start=(int(np.argmax(gaps))+1)%len(q)
    return q[start:]+q[:start]

def linear(m,obs):
    R,C,q,residual=pack(m,obs);p=np.concatenate([R,-R@C[:,:,None]],axis=2)
    A=np.stack([q[:,0,None]*p[:,2]-p[:,0],q[:,1,None]*p[:,2]-p[:,1]],axis=1).reshape(-1,4)
    _,_,v=np.linalg.svd(A);x=v[-1,:3]/v[-1,3]
    if not np.isfinite(x).all() or np.any((R@(x-C)[:,:,None])[:,2,0]<=0):raise ValueError('Invalid DLT')
    return x

def loo(m,obs):
    """Every validation point is absent from its triangulation fit."""
    e=0.
    for j,o in enumerate(obs):
        x=fit(m,obs[:j]+obs[j+1:]);e+=sse(m,[o],x)
    return e

def proposal(m,obs,x):
    n=len(obs);v=POLICY['min_views']
    if n<2*v:return None
    before=sse(m,obs,x)
    cuts=list(range(v,n-v+1));penalty=POLICY['sigma_px']**2*(3*np.log(2*n)+2*np.log(max(len(cuts),1)))
    if before<=penalty:return None
    candidates=[]
    for k in cuts:
        try:
            xa=linear(m,obs[:k]);xb=linear(m,obs[k:]);after=sse(m,obs[:k],xa)+sse(m,obs[k:],xb)
            candidates.append((after,k))
        except (ValueError,np.linalg.LinAlgError):continue
    cv_before=None;best=None
    for _,k in sorted(candidates)[:POLICY['proposal_finalists']]:
        try:
            xa=fit(m,obs[:k]);xb=fit(m,obs[k:]);after=sse(m,obs[:k],xa)+sse(m,obs[k:],xb)
            if after>=POLICY['max_sse_ratio']*before or before-after<=penalty:continue
            if cv_before is None:cv_before=loo(m,obs)
            cv_after=loo(m,obs[:k])+loo(m,obs[k:])
            if cv_after>=POLICY['max_loo_ratio']*cv_before:continue
            candidate=dict(cut=k,left=xa.tolist(),right=xb.tolist(),sse_before=before,sse_after=after,
                           cv_before=cv_before,cv_after=cv_after,gain=before-after-penalty)
            if best is None or candidate['gain']>best['gain']:best=candidate
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
    return best

def partition(m,anchors,x=None):
    """Only anchor observations are accepted. Returns static 3D child landmarks."""
    if len(anchors)<2*POLICY['min_views']:return None
    if len({o['image'] for o in anchors})!=len(anchors):return None
    obs=order(m,anchors)
    if x is None:x=fit(m,obs)
    groups=[dict(observations=obs,xyz=np.asarray(x).tolist())];history=[]
    pending={0:proposal(m,obs,x)}
    while len(groups)<POLICY['max_landmarks']:
        usable=[(q['gain'],j,q) for j,q in pending.items() if q is not None]
        if not usable:break
        _,j,q=max(usable,key=lambda t:t[0]);g=groups[j];k=q['cut'];history.append(q)
        groups[j]=dict(observations=g['observations'][:k],xyz=q['left'])
        groups.append(dict(observations=g['observations'][k:],xyz=q['right']))
        if len(groups)<POLICY['max_landmarks']:
            for i in (j,len(groups)-1):pending[i]=proposal(m,groups[i]['observations'],np.array(groups[i]['xyz']))
    if not history:return None
    # Alternate geometric labeling and refitting; circular end segments can
    # describe the same landmark. Sparse labels are merged into supported ones,
    # never discarded along with their observations. Revalidate the final
    # hypothesis on anchor-only leave-one-out predictions.
    original_groups=groups;refined=False
    for _ in range(POLICY['label_refinement_iterations']):
        try:
            _,_,_,residual=pack(m,obs)
            errors=np.stack([np.linalg.norm(residual(np.array(g['xyz'])).reshape(-1,2),axis=1) for g in groups])
            labels=np.argmin(errors,axis=0)
            supported=[j for j in range(len(groups)) if np.sum(labels==j)>=POLICY['min_views']]
            if len(supported)<2:break
            labels=np.argmin(errors[supported],axis=0)
            if any(np.sum(labels==j)<POLICY['min_views'] for j in range(len(supported))):break
            updated=[]
            for j in range(len(supported)):
                members=[o for k,o in enumerate(obs) if labels[k]==j]
                updated.append(dict(observations=members,xyz=fit(m,members).tolist()))
            groups=updated;refined=True
        except (AssertionError,ValueError,np.linalg.LinAlgError):break
    if refined:
        try:
            after=sum(sse(m,g['observations'],np.array(g['xyz'])) for g in groups)
            cv_after=sum(loo(m,g['observations']) for g in groups)
            if after>=POLICY['max_sse_ratio']*sse(m,obs,x) or cv_after>=POLICY['max_loo_ratio']*loo(m,obs):groups=original_groups;refined=False
        except (AssertionError,ValueError,np.linalg.LinAlgError):groups=original_groups;refined=False
    return dict(landmarks=[dict(cameras=[o['image'] for o in g['observations']],xyz=g['xyz']) for g in groups],history=history,
                camera_plane_variance_ratio=m['_view_consensus_plane_ratio'],labels_consolidated=refined)

def choose(m,result,camera):
    """Assignment uses camera geometry only, never query image measurements."""
    a=view_angles(m);theta=a[camera]
    distances=[min(abs(np.arctan2(np.sin(a[i]-theta),np.cos(a[i]-theta))) for i in g['cameras']) for g in result['landmarks']]
    return int(np.argmin(distances))
