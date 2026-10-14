"""Independent multi-start Cauchy triangulation, with fixed cameras and raw pixels."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
from model_io import OBS,save,digest
from evaluate import load_replica,model_metrics
from image_refinement_stage import CORE,measurement_metrics
RESULT=np.dtype([('track','<u4'),('valid','<u4'),('xyz','<f8',(3,))])

def refine(source,output):
    source,output=Path(source),Path(output);sfm=source/'sfm';target=output/'sfm'
    if output.exists():raise FileExistsError(output)
    before={n:digest(sfm/n) for n in CORE};start=time.time();target.mkdir(parents=True)
    for n in [*CORE,'cameras.xml','summary.json','region.xml','calibration.json']:
        if (sfm/n).exists():shutil.copy2(sfm/n,target/n)
    ids=np.fromfile(sfm/'camera_ids.bin','<u4');obs=np.fromfile(sfm/'observations.bin',OBS);active=np.fromfile(sfm/'active_points.bin','<u4')
    anchors=obs.copy();anchors['camera']=ids[obs['camera']];anchors.tofile(output/'all_training_observations.bin')
    exe=ROOT/'bin/triangulate_anchors.exe';points=output/'triangulated_points.bin'
    cmd=[str(exe),str(sfm),str(output/'all_training_observations.bin'),str(points),'cauchy']
    save(output/'command.json',dict(command=cmd,input_sha256=before,executable_sha256=digest(exe)))
    with (output/'native.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4'),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    result=np.fromfile(points,RESULT);assert set(map(int,result['track']))==set(map(int,active)) and len(result)==len(active)
    xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3);old=xyz.copy();valid=result['valid'].astype(bool)
    assert np.isfinite(result['xyz'][valid]).all()
    xyz[result['track'][valid]]=result['xyz'][valid];xyz.tofile(target/'points.bin')
    m=load_replica(output);metrics=measurement_metrics(m,target)
    meta=json.loads((target/'summary.json').read_text(encoding='utf8'));meta.update(point_refinement='multistart_cauchy',reprojection_rmse_px=metrics['rmse'],fixed_intrinsics=True,poses_unchanged=True,track_membership_changed=False)
    save(target/'summary.json',meta);allmetrics=model_metrics(load_replica(output),output);assert allmetrics['negative_depth_observations']==0
    # Match the native sparse export layout so all downstream readers see new XYZ.
    sparse=target/'sparse';sparse.mkdir();dt=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('id','<u4')]);data=np.empty(len(active),dt)
    for j,n in enumerate(('x','y','z')):data[n]=xyz[active,j]
    data['id']=active
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\nproperty uint id\nend_header\n'
    (sparse/'points0.ply').write_bytes(header.encode()+data.tobytes())
    for i in range(len(ids)):
        rows=obs[obs['camera']==i];d=np.empty(len(rows),dtype=[('x','<f4'),('y','<f4'),('size','<f4'),('id','<u4')])
        d['x']=rows['u'];d['y']=rows['v'];d['size']=rows['size'];d['id']=rows['track']
        h=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(d)}\nproperty float x\nproperty float y\nproperty float size\nproperty int id\nend_header\n'
        (sparse/f'p{i}.ply').write_bytes(h.encode()+d.tobytes())
    assert before=={n:digest(sfm/n) for n in CORE}
    assert all(digest(target/n)==before[n] for n in CORE if n!='points.bin')
    report=dict(complete=True,seconds=time.time()-start,refined_points=int(valid.sum()),fallback_to_original_points=int((~valid).sum()),original_pixels_tracks_cameras_and_intrinsics_unchanged=True,source_unchanged=True,source=str(source),point_shift_quantiles=np.quantile(np.linalg.norm(xyz[active]-old[active],axis=1),[0,.5,.9,.99,1]).tolist(),training_reprojection=metrics,note='Point-only refinement does not change the frozen-camera prediction.py metric. Overall SfM or 2DGS gain is not established.')
    save(output/'point_refinement.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(refine(a.source,a.output),ensure_ascii=False,indent=2))
