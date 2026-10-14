"""Final sparse model diagnostics; no geometric ground truth is assumed."""
from pathlib import Path
import json
import numpy as np
from geometry import replica_parameters,project_replica,write_ply
from model_io import save
OBS=np.dtype([('camera','<u4'),('track','<u4'),('aux','<u8'),('u','<f4'),('v','<f4'),('depth','<f4'),('size','<f4')])

def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    if not len(x):return dict(count=0)
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),rmse=float(np.sqrt(np.mean(x*x))),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max()))

def load_replica(folder):
    sfm=folder/'sfm';p=replica_parameters(sfm/'calibration.bin')
    ids=np.fromfile(sfm/'camera_ids.bin','<u4');poses=np.fromfile(sfm/'poses.bin','<f8').reshape(-1,12)
    cameras={int(i):dict(c=a[:3],r=a[3:].reshape(3,3)) for i,a in zip(ids,poses)}
    xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3);active=np.fromfile(sfm/'active_points.bin','<u4');obs=np.fromfile(sfm/'observations.bin',OBS)
    groups={int(i):[] for i in active}
    for o in obs:groups[int(o['track'])].append((int(ids[o['camera']]),np.array([o['u'],o['v']],float)))
    points={i:(xyz[i],o) for i,o in groups.items()}
    def project(i,x):
        c=cameras[i];v=(np.atleast_2d(x)-c['c'])@c['r'].T
        return project_replica(v[:,:2]/v[:,2,None],p)
    def unproject(i,uv):
        uv=np.atleast_2d(uv);q=(uv-[p['width']/2+p['cx'],p['height']/2+p['cy']])/p['f']
        for _ in range(12):
            e=project_replica(q,p)-uv
            jx=(project_replica(q+[1e-6,0],p)-project_replica(q-[1e-6,0],p))/2e-6
            jy=(project_replica(q+[0,1e-6],p)-project_replica(q-[0,1e-6],p))/2e-6
            j=np.stack((jx,jy),axis=-1);delta=np.linalg.solve(j,e[...,None])[...,0];q-=delta
            if np.max(np.abs(delta))<1e-12:break
        return q
    return dict(cameras=cameras,points=points,project=project,unproject=unproject,calibration={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()},source_summary=json.loads((sfm/'summary.json').read_text()))

def model_metrics(m,folder):
    errors=[];angles=[];lengths=[];negative=0;bycamera={i:[] for i in m['cameras']};qualified=0
    # Group projected observations to avoid Python calls per observation.
    xy_bycamera={i:[] for i in m['cameras']};world_bycamera={i:[] for i in m['cameras']};point_bycamera={i:[] for i in m['cameras']}
    perpoint={i:[] for i in m['points']}
    for tid,(x,obs) in m['points'].items():
        lengths.append(len(obs));rays=[]
        for i,uv in obs:
            c=m['cameras'][i];v=x-c['c'];z=(c['r']@v)[2];negative+=int(z<=0);rays.append(v/np.linalg.norm(v))
            xy_bycamera[i].append(uv);world_bycamera[i].append(x);point_bycamera[i].append(tid)
        if len(rays)>=2:angles.append(float(np.degrees(np.arccos(np.clip(np.min(np.array(rays)@np.array(rays).T),-1,1)))))
    for i in m['cameras']:
        if not len(world_bycamera[i]):continue
        e=np.linalg.norm(m['project'](i,world_bycamera[i])-xy_bycamera[i],axis=1);errors.extend(e);bycamera[i]=stats(e)
        for tid,v in zip(point_bycamera[i],e):perpoint[tid].append(float(v))
    views=[len({i for i,uv in obs}) for x,obs in m['points'].values()]
    qualified=sum(len({i for i,uv in m['points'][tid][1]})>=3 and max(e)<=4 for tid,e in perpoint.items())
    result=dict(registered_images=len(m['cameras']),points=len(m['points']),observations=len(errors),reprojection_px=stats(errors),track_length=stats(lengths),triangulation_max_angle_degrees=stats(angles),negative_depth_observations=negative,three_view_points_max_error_le_4px=qualified,camera_calibration=m['calibration'],per_camera=bycamera,absolute_accuracy_measured=False)
    if 'source_summary' in m:
        assert np.isclose(result['reprojection_px']['rmse'],m['source_summary']['reprojection_rmse_px'],atol=1e-5,rtol=1e-5),'Replica projection does not reproduce its own C++ metric'
        xyz=np.array([x for x,obs in m['points'].values()]);write_ply(folder/'sparse.ply',xyz,np.full((len(xyz),3),160,dtype=np.uint8))
    result['track_length_definition']='Number of feature observations; can include multiple features in one image. Use unique_view_track_length for view redundancy.'
    result['unique_view_track_length']=stats(views)
    result['unique_point_image_pairs']=sum(views)
    result['extra_same_image_observations']=len(errors)-sum(views)
    save(folder/'metrics.json',result);return result
