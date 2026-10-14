"""Optional geometric-view track partition. Fixed K/poses, raw pixels unchanged."""
from pathlib import Path
import argparse,json,shutil,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'python'))
from view_consensus_partition import partition,POLICY
from model_io import OBS,save,digest
from evaluate import load_replica,model_metrics
CORE=('camera_ids.bin','calibration.bin','poses.bin','points.bin','active_points.bin','observations.bin')

def run(source,output):
    source,output=map(Path,(source,output));done=output/'view_consensus.json'
    if done.exists():return json.loads(done.read_text())
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);sfm=source/'sfm';target=output/'sfm';target.mkdir();start=time.time()
    protected={n:digest(sfm/n) for n in CORE};m=load_replica(source);rows=np.fromfile(sfm/'observations.bin',OBS);original=rows.copy();ids=np.fromfile(sfm/'camera_ids.bin','<u4');xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3);extra=[];next_id=len(xyz);partitions={};parents={}
    row_indices={}
    for j,t in enumerate(rows['track']):row_indices.setdefault(int(t),[]).append(j)
    for tid,(x,observations) in m['points'].items():
        if len(observations)<6:continue
        obs=[dict(image=i,uv=uv.tolist()) for i,uv in observations]
        try:q=partition(m,obs)
        except (AssertionError,ValueError,np.linalg.LinAlgError):continue
        if q is None:continue
        assignment={};child_ids=[]
        for j,g in enumerate(q['landmarks']):
            if j==0:child=tid;xyz[tid]=g['xyz']
            else:child=next_id;next_id+=1;extra.append(g['xyz']);parents[child]=tid
            child_ids.append(child)
            for camera in g['cameras']:assignment[camera]=child
        for ri in row_indices[tid]:rows['track'][ri]=assignment[int(ids[rows['camera'][ri]])]
        q['child_ids']=child_ids;partitions[str(tid)]=q
    for n in (*CORE,'summary.json','cameras.xml','calibration.json','region.xml','fixed_xml_export.json'):
        if (sfm/n).exists():shutil.copy2(sfm/n,target/n)
    if extra:xyz=np.vstack([xyz,extra])
    xyz.astype('<f8').tofile(target/'points.bin');rows.tofile(target/'observations.bin');np.unique(rows['track']).astype('<u4').tofile(target/'active_points.bin')
    restored=rows.copy();restored['track']=[parents.get(int(t),int(t)) for t in rows['track']]
    assert original.tobytes()==restored.tobytes()
    assert len(set(zip(map(int,rows['camera']),map(int,rows['track']))))==len(rows)
    model=load_replica(output);model.pop('source_summary');metrics=model_metrics(model,output);assert metrics['negative_depth_observations']==0
    meta=json.loads((sfm/'summary.json').read_text());meta.update(sparse_points=metrics['points'],observations=metrics['observations'],reprojection_rmse_px=metrics['reprojection_px']['rmse'],reprojection_max_px=metrics['reprojection_px']['maximum'],view_consensus_partitions=len(partitions),track_membership_changed=bool(partitions))
    save(target/'summary.json',meta);model_metrics(load_replica(output),output);save(output/'partitions.json',partitions)
    assert all(digest(sfm/n)==h for n,h in protected.items())
    assert all(digest(sfm/n)==digest(target/n) for n in ('poses.bin','calibration.bin','camera_ids.bin'))
    result=dict(complete=True,source=str(source),seconds=time.time()-start,policy=POLICY,split_tracks=len(partitions),added_landmarks=len(extra),observations_preserved=len(rows),zero_deleted_observations=True,original_uv_and_metadata_preserved=True,cameras_and_intrinsics_unchanged=True,input_sha256=protected,metrics=metrics)
    save(done,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.input,a.output)))
