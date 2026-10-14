"""Export the frozen train-only pose probe and refit every original track.

No feature coordinates, track membership, calibration, or evaluation pose is changed.
The identical point solver is run with the old poses as a matched-budget diagnostic.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/sfm/python'))
from geometry import replica_parameters, project_replica, write_ply
from model_io import OBS, patch_xml
OUT=ROOT/'outputs/sfm_render_pose_20261014'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')

def residual(x,pose,obs,k,jacobian=False):
    r=pose[obs['camera'],3:].reshape(-1,3,3)
    v=np.einsum('nij,nj->ni',r,x[obs['track']]-pose[obs['camera'],:3])
    z=v[:,2]; xy=v[:,:2]/z[:,None]
    e=project_replica(xy,k)-np.c_[obs['u'],obs['v']]
    if not jacobian:return e,z
    eps=1e-6; j=np.empty((len(obs),2,2))
    for dim in range(2):
        shift=np.zeros(2);shift[dim]=eps
        j[:,:,dim]=(project_replica(xy+shift,k)-project_replica(xy-shift,k))/(2*eps)
    jxy=(r[:,:2,:]-xy[:,:,None]*r[:,2:3,:])/z[:,None,None]
    return e,z,np.einsum('nij,njk->nik',j,jxy)

def metrics(x,pose,obs,k):
    e,z=residual(x,pose,obs,k);q=np.sum(e*e,axis=1)
    return dict(rmse_px=float(np.sqrt(q.mean())),max_px=float(np.sqrt(q.max())),
        mean_px=float(np.sqrt(q).mean()),negative_depth_observations=int((z<=0).sum()),
        observations=len(obs),points=int(len(np.unique(obs['track']))))

def refit(x,pose,obs,k,iterations=20):
    x=x.copy();n=len(x);ids=obs['track'];damp=np.full(n,1e-5);trace=[]
    # A point owns an independent 3x3 normal system; damping and acceptance are per point.
    for step in range(iterations):
        e,z,j=residual(x,pose,obs,k,True)
        h=np.zeros((n,3,3));b=np.zeros((n,3))
        np.add.at(h,ids,np.einsum('nji,njk->nik',j,j))
        np.add.at(b,ids,np.einsum('nji,nj->ni',j,e))
        diag=np.maximum(np.diagonal(h,axis1=1,axis2=2),1e-12)
        system=h+np.eye(3)[None]*(damp[:,None]*diag)[:,None,:]
        delta=np.linalg.solve(system,b[...,None])[...,0]
        proposed=x-delta;ee,zz=residual(proposed,pose,obs,k)
        cost=np.bincount(ids,weights=np.sum(e*e,axis=1),minlength=n)
        newcost=np.bincount(ids,weights=np.sum(ee*ee,axis=1),minlength=n)
        bad=np.bincount(ids,weights=(zz<=0)|~np.isfinite(zz),minlength=n)>0
        accept=(newcost<cost)&~bad&np.isfinite(proposed).all(1)
        x[accept]=proposed[accept];damp[accept]=np.maximum(damp[accept]/3,1e-12)
        damp[~accept]=np.minimum(damp[~accept]*10,1e6)
        trace.append(dict(iteration=step+1,accepted_points=int(accept.sum()),rmse_px=float(np.sqrt(np.minimum(cost,newcost).sum()/len(obs)))))
    return x,trace

def main():
    prep=json.loads((OUT/'geometry_preparation.json').read_text(encoding='utf8'))
    source=Path(prep['native_source']);probe=json.loads((OUT/'probe_summary.json').read_text(encoding='utf8'))
    assert probe['complete'] and len(probe['records'])==70
    for name,h in prep['protected_sha256'].items():assert sha(name)==h,name
    assert sha(OUT/'geometry_anchors.npz')==prep['anchors_sha256']
    anchor=np.load(OUT/'geometry_anchors.npz');a=anchor['gauge_rotation'];scale=float(anchor['gauge_scale']);offset=anchor['gauge_translation']
    ids=np.fromfile(source/'camera_ids.bin','<u4');pose=np.fromfile(source/'poses.bin','<f8').reshape(-1,12)
    xyz=np.fromfile(source/'points.bin','<f8').reshape(-1,3);obs=np.fromfile(source/'observations.bin',OBS)
    active=np.fromfile(source/'active_points.bin','<u4');k=replica_parameters(source/'calibration.bin')
    newpose=pose.copy();lookup={int(i):j for j,i in enumerate(ids)};render_poses={};polar_error=[]
    for rec in probe['records']:
        ci=int(rec['name'])-1;assert ci%8!=0
        rr=np.asarray(rec['r']);u,_,vt=np.linalg.svd(rr);r=u@vt
        assert np.linalg.det(r)>0
        t=np.asarray(rec['t']);c=-r.T@t
        newpose[lookup[ci],:3]=a.T@(c-offset)/scale
        newpose[lookup[ci],3:]=(r@a).ravel()
        render_poses[rec['name']]=dict(r=r.tolist(),t=t.tolist())
        polar_error.append(float(np.max(np.abs(rr-r))))
    assert np.array_equal(newpose[ids%8==0],pose[ids%8==0])
    # Verify the world-point Jacobian before any optimization.
    subset=obs[::1000];_,_,jj=residual(xyz,newpose,subset,k,True)
    errors=[]
    for d in range(3):
        xx=xyz.copy();xx[:,d]+=1e-6;ee,_=residual(xx,newpose,subset,k)
        xx=xyz.copy();xx[:,d]-=1e-6;em,_=residual(xx,newpose,subset,k)
        errors.append(float(np.max(np.abs((ee-em)/2e-6-jj[:,:,d]))))
    assert max(errors)<1e-3,errors
    base,base_trace=refit(xyz,pose,obs,k)
    candidate,trace=refit(xyz,newpose,obs,k)
    result=dict(original=metrics(xyz,pose,obs,k),baseline_same_refit=metrics(base,pose,obs,k),
        candidate_before_point_refit=metrics(xyz,newpose,obs,k),candidate_after_point_refit=metrics(candidate,newpose,obs,k),
        coordinate_system='Original 3072x2048 distorted feature pixels; same 257967 observations and track membership.',
        point_solver='20 fixed-camera least-squares LM steps, independent per-point accept/reject; no filtering.',
        candidate_trace=trace,baseline_trace=base_trace,point_jacobian_max_absolute_error=errors,
        camera_polar_projection_max_change=max(polar_error),
        fixed_intrinsics=True,evaluation_camera_poses_unchanged=True,all_feature_measurements_unchanged=True,
        interpretation='Reprojection fit is not independent 3D accuracy. The pose proposal uses a frozen trained 2DGS proxy and training RGB; reserved tracks were already used by the original SfM/proxy.')
    assert result['candidate_after_point_refit']['negative_depth_observations']==0
    dest=OUT/'candidate/sfm'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    for name in ['camera_ids.bin','calibration.bin','calibration.json','observations.bin','active_points.bin','cameras.xml']:
        shutil.copy2(source/name,dest/name)
    candidate.astype('<f8').tofile(dest/'points.bin');newpose.astype('<f8').tofile(dest/'poses.bin')
    tree=ET.parse(dest/'cameras.xml')
    for camera in tree.getroot().findall('chunk/cameras/camera'):
        ci=int(camera.attrib['label'])-1;p=newpose[lookup[ci]];matrix=np.eye(4)
        matrix[:3,:3]=p[3:].reshape(3,3).T;matrix[:3,3]=p[:3]
        camera.find('transform').text=' '.join(format(v,'.17g') for v in matrix.ravel())
    tree.write(dest/'cameras.xml',encoding='utf8',xml_declaration=True);patch_xml(dest)
    (dest/'sparse').mkdir();write_ply(dest/'sparse/points.ply',candidate[active],np.full((len(active),3),127,np.uint8))
    save(dest/'summary.json',dict(aligned_cameras=len(ids),sparse_points=len(active),observations=len(obs),
        reprojection_rmse_px=result['candidate_after_point_refit']['rmse_px'],reprojection_max_px=result['candidate_after_point_refit']['max_px'],
        fitted_intrinsics=0,track_membership_changed=False,native_code_called=False,absolute_accuracy_measured=False))
    save(OUT/'render_poses.json',render_poses);save(OUT/'geometry_comparison.json',result)
    base.astype('<f8').tofile(OUT/'baseline_refit_points.bin')
    for name,h in prep['protected_sha256'].items():assert sha(name)==h,name
    save(OUT/'export_audit.json',dict(passed=True,original_unchanged=True,
        unchanged_files={n:sha(dest/n) for n in ['camera_ids.bin','calibration.bin','observations.bin','active_points.bin']},
        candidate_sha256={str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()},
        evaluation_pose_bytes_identical=True,train_poses_updated=70))
    print(json.dumps({k:result[k] for k in ['original','baseline_same_refit','candidate_before_point_refit','candidate_after_point_refit']},indent=2))

if __name__=='__main__':main()
