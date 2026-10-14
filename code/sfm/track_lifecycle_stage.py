"""Optional ordered track partition after SfM; preserves every raw observation.

Only validated partitions change the input. Camera poses and K remain fixed.
Uses exactly the partition function exercised by the anchor/query experiment.
"""
from pathlib import Path
import sys,json,shutil,argparse,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
from ordered_track_partition import fit,split
from model_io import OBS,save,digest
from evaluate import load_replica,model_metrics
CORE=('camera_ids.bin','calibration.bin','poses.bin','points.bin','active_points.bin','observations.bin')

def run(source,output):
    source,output=map(Path,(source,output));done=output/'track_lifecycle.json'
    if done.exists():return json.loads(done.read_text(encoding='utf8'))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);sfm=source/'sfm';target=output/'sfm';target.mkdir();start=time.time()
    protected={n:digest(sfm/n) for n in CORE};m=load_replica(source);rows=np.fromfile(sfm/'observations.bin',OBS);original=rows.copy();ids=np.fromfile(sfm/'camera_ids.bin','<u4');xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3);newpoints=[];next_id=len(xyz);splits={}
    for no,(tid,(x,observations)) in enumerate(m['points'].items()):
        if len(observations)<8:continue
        obs=sorted([dict(image=i,uv=uv.tolist()) for i,uv in observations],key=lambda o:o['image'])
        try:base=fit(m,obs);q=split(m,obs,base)
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
        if q is None:continue
        child=next_id;next_id+=1;xyz[tid]=q['left'];newpoints.append(q['right']);mask=(rows['track']==tid)&(ids[rows['camera']]>q['boundary_camera']);rows['track'][mask]=child;q['child']=child;splits[str(tid)]=q
        if len(splits)%100==0:print(f'validated partitions: {len(splits)}; visited tracks: {no}',flush=True)
    # Copy only authoritative model/metadata files; never carry stale point clouds.
    for n in (*CORE,'summary.json','cameras.xml','calibration.json','region.xml','fixed_xml_export.json'):
        if (sfm/n).exists():shutil.copy2(sfm/n,target/n)
    if newpoints:xyz=np.vstack([xyz,newpoints])
    xyz.astype('<f8').tofile(target/'points.bin');rows.tofile(target/'observations.bin');np.unique(rows['track']).astype('<u4').tofile(target/'active_points.bin')
    # Exact observation preservation, with an explicit child-to-parent identity map.
    parent={v['child']:int(k) for k,v in splits.items()};restored=rows.copy();restored['track']=[parent.get(int(t),int(t)) for t in rows['track']]
    assert original.tobytes()==restored.tobytes()
    assert len(set(zip(map(int,rows['camera']),map(int,rows['track']))))==len(rows)
    model=load_replica(output);model.pop('source_summary');metrics=model_metrics(model,output);assert metrics['negative_depth_observations']==0
    meta=json.loads((sfm/'summary.json').read_text());meta.update(sparse_points=metrics['points'],observations=metrics['observations'],reprojection_rmse_px=metrics['reprojection_px']['rmse'],reprojection_max_px=metrics['reprojection_px']['maximum'],track_lifecycle_partitions=len(splits),track_membership_changed=bool(splits))
    save(target/'summary.json',meta);model_metrics(load_replica(output),output);save(output/'partitions.json',splits)
    assert all(digest(sfm/n)==h for n,h in protected.items())
    assert all(digest(sfm/n)==digest(target/n) for n in ('poses.bin','calibration.bin','camera_ids.bin'))
    result=dict(complete=True,source=str(source.resolve()),seconds=time.time()-start,split_tracks=len(splits),input_points=len(m['points']),output_points=metrics['points'],observations_preserved=len(rows),zero_deleted_observations=True,original_uv_and_metadata_preserved=True,cameras_and_intrinsics_unchanged=True,input_sha256=protected,metrics=metrics,scope='ordered capture, at most two landmarks per parent, no BA, no independent geometry truth')
    save(done,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.input,a.output),ensure_ascii=False,indent=2))
