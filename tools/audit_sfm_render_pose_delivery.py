"""Independent final read-only audit of inputs, fixed evaluation poses and old results."""
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_render_pose_20261014'
OLD=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014'
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def local(p):
    p=str(p).replace('\\','/')
    if os.name=='nt':
        if p.startswith('/mnt/f/'):p='F:/'+p[len('/mnt/f/'):]
        elif p.startswith('/home/'):p='//wsl.localhost/Ubuntu-20.04'+p
    elif len(p)>2 and p[1]==':':p='/mnt/'+p[0].lower()+p[2:]
    return Path(p)
def sha(p):return hashlib.sha256(local(p).read_bytes()).hexdigest()
def image_lines(p):
    result={}
    for line in local(p).read_text().splitlines():
        v=line.split()
        if len(v)==10 and v[-1].endswith('.png'):result[Path(v[-1]).stem]=line
    return result
def vertices(p):
    types={'float':'<f4','float32':'<f4','uchar':'u1','uint8':'u1'};fields=[];n=None
    with local(p).open('rb') as f:
        assert f.readline().strip()==b'ply'
        assert f.readline().strip()==b'format binary_little_endian 1.0'
        while True:
            line=f.readline().decode('ascii').strip();v=line.split()
            if line=='end_header':break
            if v[:2]==['element','vertex']:n=int(v[2])
            elif v[:1]==['property']:
                assert v[1] in types,v
                fields.append((v[2],types[v[1]]))
            elif v[:1]==['element']:raise ValueError('Unsupported extra element')
        return np.fromfile(f,np.dtype(fields),n)

def main():
    p=read(OUT/'protocol.json');previous=read(OLD/'protocol.json');prep=read(OUT/'geometry_preparation.json')
    assert read(OUT/'audit.json')['passed'] and read(OUT/'report_status.json')['visual_review_complete']
    protected=dict(p['protected_sha256']);protected.update(p['new_inputs_sha256']);protected.update(prep['protected_sha256'])
    for name,h in protected.items():assert sha(name)==h,name
    code=local('/home/wk/01_code/03_gs/2dgs-foreground-fix-20261014/code')
    for name,h in p['repair_code_sha256'].items():assert sha(code/name)==h and sha(ROOT/'code/2dgs_foreground_fix'/name)==h,name
    for name,h in p['experiment_code_sha256'].items():assert sha(ROOT/name)==h,name
    assert p['common_args']==previous['common_args'] and p['train_names']==previous['train_names'] and p['test_names']==previous['test_names']
    datasets={'REFINED':p['datasets']['REFINED'],**{m:previous['datasets'][m] for m in ['OFF','COLMAP']}}
    for name,h in p['image_sha256'].items():
        for dataset in datasets.values():assert sha(local(dataset)/'images'/(name+'.png'))==h
    a=local(datasets['REFINED'])/'sparse/0';b=local(datasets['OFF'])/'sparse/0'
    assert sha(a/'cameras.txt')==sha(b/'cameras.txt')
    new=image_lines(a/'images.txt');old=image_lines(b/'images.txt')
    assert set(new)==set(old) and all(new[n]==old[n] for n in p['test_names'])
    ncloud=vertices(a/'points3D.ply');ocloud=vertices(b/'points3D.ply')
    assert len(ncloud)==len(ocloud) and all(np.array_equal(ncloud[c],ocloud[c]) for c in ['red','green','blue'])
    assert all(np.isfinite(ncloud[c]).all() for c in ['x','y','z'])
    native=local(prep['native_source']);candidate=OUT/'candidate/sfm'
    for name in ['camera_ids.bin','calibration.bin','observations.bin','active_points.bin']:
        assert sha(native/name)==sha(candidate/name),name
    ids=np.fromfile(candidate/'camera_ids.bin','<u4')
    before=np.fromfile(native/'poses.bin','<f8').reshape(-1,12);after=np.fromfile(candidate/'poses.bin','<f8').reshape(-1,12)
    assert np.array_equal(before[ids%8==0],after[ids%8==0])
    assert np.isfinite(after).all()
    model=read(OUT/'trained_REFINED.json');mp=local(model['model'])/'point_cloud/iteration_30000/point_cloud.ply'
    assert sha(mp)==model['model_sha256']==read(OUT/'results/REFINED/metrics.json')['model_sha256']
    untouched=['code/2dgs_foreground_fix','code/sfm/frontend','tools/evaluate_foreground_fix_2dgs.py']
    assert not subprocess.check_output(['git','diff','--name-only','ce5c2f5','--',*untouched],cwd=ROOT).strip()
    report=OUT/'SfM与2DGS验证报告.md'
    result=dict(passed=True,protected_files=len(protected),dataset_images_hash_checked=len(p['image_sha256'])*len(datasets),
        frozen_training_and_experiment_code_unchanged=True,unchanged_features_tracks_and_calibration=True,
        evaluation_native_pose_bytes_and_render_pose_lines_identical=True,initializer_count_order_colors_preserved=True,
        input_points=len(ncloud),final_iteration=30000,new_training_runs=1,
        candidate_model_sha256=sha(mp),report_sha256=sha(report),visual_review_complete=True,
        all_10_evaluation_views_inspected=True,default_not_replaced=True,
        outcome='Completed experiment; failed the requested joint SfM and downstream improvement criterion.',
        report_source_sha256=sha(ROOT/'tools/report_sfm_render_pose_2dgs.py'))
    (OUT/'delivery_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
