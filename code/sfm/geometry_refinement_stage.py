"""Optional geometric BA; preserves the original feature coordinates and graph."""
from pathlib import Path
import argparse,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
from model_io import OBS,digest,save
from evaluate import load_replica,model_metrics
from image_refinement_stage import CORE,_ordered_rows,_restore_xml_labels

def refine(source,output,mode):
    source,output=Path(source),Path(output)
    if output.exists():raise FileExistsError(output)
    sfm=source/'sfm'; before={n:digest(sfm/n) for n in CORE}
    exe=ROOT/'bin/refine_geometry.exe'
    output.mkdir(parents=True)
    cmd=[str(exe),str(sfm),str(output/'sfm'),mode]
    save(output/'command.json',dict(command=cmd,source=str(source),input_sha256=before,executable_sha256=digest(exe)))
    with (output/'native.log').open('w',encoding='utf8') as f:
        subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'),
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    target=output/'sfm'; ids=np.fromfile(sfm/'camera_ids.bin','<u4'); newids=np.fromfile(target/'camera_ids.bin','<u4')
    ka,oa,_=_ordered_rows(np.fromfile(sfm/'observations.bin',OBS),ids)
    kb,ob,_=_ordered_rows(np.fromfile(target/'observations.bin',OBS),newids)
    assert np.array_equal(ka,kb) and set(ids)==set(newids),'Observation/camera identity changed'
    for n in ('track','u','v','aux','depth'):assert oa[n].tobytes()==ob[n].tobytes(),'Measurement changed: '+n
    assert set(np.fromfile(sfm/'active_points.bin','<u4'))==set(np.fromfile(target/'active_points.bin','<u4'))
    fixed=mode!='calibration'
    if fixed:assert digest(sfm/'calibration.bin')==digest(target/'calibration.bin')
    xml=_restore_xml_labels(sfm,target,ids,newids)
    meta=json.loads((target/'summary.json').read_text())
    meta.update(geometry_refinement=mode,fixed_intrinsics=fixed,fitted_intrinsics=0 if fixed else 7,track_membership_changed=False)
    save(target/'summary.json',meta)
    metrics=model_metrics(load_replica(output),output)
    assert metrics['negative_depth_observations']==0
    assert before=={n:digest(sfm/n) for n in CORE},'Input changed'
    result=dict(complete=True,mode=mode,original_pixels_and_identities_preserved=True,
                input_unchanged=True,fixed_intrinsics=fixed,xml_identity_audit=xml,
                metrics=metrics,ba_calls=3)
    save(output/'geometry_refinement.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['fixed','cauchy','calibration'],required=True);a=p.parse_args();refine(a.source,a.output,a.mode)
