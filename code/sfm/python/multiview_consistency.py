"""Training-track cross-view geometry guard for image-derived measurements.

No coordinates are replaced by projected coordinates. Geometry decides only
whether to accept an entire image proposal or retain an entire original track.
Evaluation/heldout tracks are never inputs to this module.
"""
from pathlib import Path
import csv,json,time
import numpy as np
from model_io import OBS,digest,save
from evaluate import load_replica

POLICY=dict(min_views=4,max_displacement_px=6.0,full_triangulation_rmse_px=2.0,
            cross_view_required_ratio=.9,cross_view_error_cap_px=10.0,
            triangulator='fixed-pose normalized-ray DLT',
            folds='camera-ID sorted even and odd observations; each predicts the opposite half',
            acceptance='all checks pass; otherwise revert whole track without dropping any observation',
            camera_parameters_updated=False,heldout_observations_used=False)

def triangulate(rays,projections):
    """Batched DLT: rays[T,V,2], projection[T,V,3,4] world-to-camera."""
    a=rays[...,None]*projections[:,:,2,None,:]-projections[:,:,:2,:]
    _,_,vh=np.linalg.svd(a.reshape(len(a),-1,4),full_matrices=False)
    with np.errstate(divide='ignore',invalid='ignore'):
        return vh[:,-1,:3]/vh[:,-1,3,None]

def errors(model,ids,cameras,xyz,xy,poses):
    repeated=np.broadcast_to(xyz[:,None],(*cameras.shape,3)).reshape(-1,3)
    ci=cameras.ravel();uv=xy.reshape(-1,2)
    error=np.full(len(ci),np.inf)
    for c in np.unique(ci):
        ix=np.flatnonzero(ci==c);points=repeated[ix]
        z=((points-poses[c,:3])@poses[c,3:].reshape(3,3).T)[:,2]
        good=(z>1e-10)&np.isfinite(points).all(1)
        if good.any():
            pred=model['project'](int(ids[c]),points[good])
            error[ix[good]]=np.linalg.norm(pred-uv[ix[good]],axis=1)
    return error.reshape(cameras.shape)

def gate(source,pixels,output):
    source,pixels,output=map(Path,(source,pixels,output))
    if output.exists():raise FileExistsError(output)
    start=time.time();sfm=source/'sfm'
    original=np.fromfile(sfm/'observations.bin',OBS)
    proposed=np.fromfile(pixels/'observations_refined.bin',OBS)
    info=json.loads((pixels/'summary.json').read_text(encoding='utf8'))
    core=info['input_sha256']
    if any(digest(sfm/n)!=h for n,h in core.items()):raise ValueError('Proposal source hash mismatch')
    for field in ('camera','track','aux','depth','size'):
        if original[field].tobytes()!=proposed[field].tobytes():raise ValueError('Observation identity changed')
    model=load_replica(source)
    ids=np.fromfile(sfm/'camera_ids.bin','<u4');poses=np.fromfile(sfm/'poses.bin','<f8').reshape(-1,12)
    r=poses[:,3:].reshape(-1,3,3)
    pm=np.concatenate([r,(-np.einsum('nij,nj->ni',r,poses[:,:3]))[:,:,None]],axis=-1)
    oldxy=np.c_[original['u'],original['v']].astype(float)
    newxy=np.c_[proposed['u'],proposed['v']].astype(float)
    ray_old=np.zeros_like(oldxy);ray_new=np.zeros_like(newxy)
    for c in range(len(ids)):
        ix=np.flatnonzero(original['camera']==c)
        ray_old[ix]=model['unproject'](int(ids[c]),oldxy[ix])
        ray_new[ix]=model['unproject'](int(ids[c]),newxy[ix])
    groups={};tracks={}
    order=np.lexsort((ids[original['camera']],original['track']))
    boundaries=np.r_[0,np.flatnonzero(np.diff(original['track'][order]))+1,len(order)]
    for begin,end in zip(boundaries[:-1],boundaries[1:]):
        rows=order[begin:end];tid=int(original['track'][rows[0]])
        tracks[tid]=rows;groups.setdefault(len(rows),[]).append(rows)
    accepted=np.zeros(len(original),bool);stats=[]
    for count,rows in sorted(groups.items()):
        rows=np.asarray(rows);cam=original['camera'][rows]
        tids=original['track'][rows[:,0]]
        maxshift=np.linalg.norm(newxy[rows]-oldxy[rows],axis=-1).max(1)
        if count<POLICY['min_views']:
            stats.extend((int(t),count,False,float(s),'too_few_views',None,None,None) for t,s in zip(tids,maxshift))
            continue
        full=triangulate(ray_new[rows],pm[cam])
        full_errors=errors(model,ids,cam,full,newxy[rows],poses)
        full_rmse=np.sqrt(np.mean(full_errors**2,axis=1))
        cross={}
        for name,rays,xy in [('raw',ray_old,oldxy),('refined',ray_new,newxy)]:
            cross_errors=np.full(rows.shape,np.inf)
            for fold in (0,1):
                anchor=np.arange(fold,count,2);query=np.arange(1-fold,count,2)
                point=triangulate(rays[rows[:,anchor]],pm[cam[:,anchor]])
                cross_errors[:,query]=errors(model,ids,cam[:,query],point,xy[rows[:,query]],poses)
            cross[name]=np.minimum(cross_errors,POLICY['cross_view_error_cap_px']).mean(1)
        shift_ok=maxshift<=POLICY['max_displacement_px']
        reprojection_ok=np.isfinite(full_rmse)&(full_rmse<=POLICY['full_triangulation_rmse_px'])
        cross_ok=cross['refined']<=POLICY['cross_view_required_ratio']*cross['raw']
        good=shift_ok&reprojection_ok&cross_ok
        accepted[rows[good]]=True
        for i,t in enumerate(tids):
            reasons=','.join(n for n,v in [('shift',shift_ok[i]),('triangulation',reprojection_ok[i]),('cross_view',cross_ok[i])] if not v)
            stats.append((int(t),count,bool(good[i]),float(maxshift[i]),reasons,
                          float(full_rmse[i]) if np.isfinite(full_rmse[i]) else None,
                          float(cross['raw'][i]),float(cross['refined'][i])))
    output.mkdir(parents=True)
    result=original.copy();result[accepted]=proposed[accepted]
    result.tofile(output/'observations_refined.bin')
    np.ones(len(result),dtype='<f4').tofile(output/'observation_sigma.bin')
    with (output/'track_consistency.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['track','views','accepted','max_shift_px','rejection_reasons',
                                  'refined_full_dlt_rmse','raw_cross_view_error','refined_cross_view_error']);w.writerows(stats)
    d=np.hypot(result['u'].astype(float)-original['u'],result['v'].astype(float)-original['v'])
    metadata=dict(complete=True,method='released multiview network with native training-track geometry guard',
                  input_sha256=core,network_proposal_sha256=digest(pixels/'observations_refined.bin'),
                  policy=POLICY,tracks=len(tracks),accepted_tracks=sum(r[2] for r in stats),
                  observations=len(result),accepted_observations=int(accepted.sum()),
                  changed_observations=int((d>1e-5).sum()),displacement_px=dict(mean=float(d.mean()),max=float(d.max())),
                  source_unchanged=all(digest(sfm/n)==h for n,h in core.items()),elapsed_seconds=time.time()-start)
    assert metadata['source_unchanged']
    save(output/'summary.json',metadata)
    return metadata
