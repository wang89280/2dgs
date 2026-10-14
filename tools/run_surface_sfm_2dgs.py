"""Frozen GATE/HULL 30k runs; same inputs except initial xyz, no renderer changes."""
import argparse,datetime,fcntl,json,os,subprocess,sys,traceback
from pathlib import Path
from run_foreground_fix_2dgs import read,save,sha,CODE,PACKAGE
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
OLD=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014'
WORK=Path('/home/wk/01_code/03_gs/sfm-surface-init-20261014')
METHODS=['HULL','GATE']
def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def verify(p):
    for group in ['protected_sha256','new_inputs_sha256']:
        for n,h in p[group].items():assert sha(n)==h,n
    for n,h in p['repair_code_sha256'].items():assert sha(CODE/n)==h and sha(PACKAGE/n)==h,n
    for n,h in p['experiment_code_sha256'].items():assert sha(ROOT/n)==h,n
    for m in METHODS:
        for n,h in p['image_sha256'].items():assert sha(Path(p['datasets'][m])/'images'/(n+'.png'))==h,n
def prepare():
    target=OUT/'protocol.json'
    if target.exists():p=read(target);verify(p);return p
    old=read(OLD/'protocol.json');prepared=read(OUT/'prepared.json');checks=read(OUT/'surface_checks.json');surface=read(OUT/'surface_protocol.json')
    assert prepared['complete'] and checks['passed'] and read(OUT/'native_export_audit.json')['passed']
    inputs={}
    for obj in prepared['outputs'].values():inputs.update(obj['sha256'])
    inputs.update(checks['GATE']['sha256']);inputs.update(surface['protected_sha256'])
    for n in ['prepared.json','surface_checks.json','surface_protocol.json','surface_result.npz','sfm_landmark_updates.npz','native_geometry_comparison.json','native_export_audit.json']:
        inputs[str(OUT/n)]=sha(OUT/n)
    inputs.update({str(f):sha(f) for f in (OUT/'sfm_candidate').rglob('*') if f.is_file()})
    for n in ['geometry_anchors.npz','geometry_preparation.json']:
        f=ROOT/'outputs/sfm_render_pose_20261014'/n;inputs[str(f)]=sha(f)
    protected=dict(old['protected_sha256']);protected.update(old['new_inputs_sha256'])
    for method in ['OFF','COLMAP']:
        for n in ['cfg_args','point_cloud/iteration_30000/point_cloud.ply']:
            f=Path(old['models'][method])/n;protected[str(f)]=sha(f)
        f=OLD/'results'/method/'metrics.json';protected[str(f)]=sha(f)
    scripts=['code/sfm/surface_triangulation.py','tools/prepare_surface_sfm.py','tools/check_surface_sfm.py','tools/diagnose_sfm_seed_surface.py',
        'tools/export_surface_native.py','tools/run_surface_sfm_2dgs.py','tools/evaluate_surface_sfm_2dgs.py','tools/evaluate_foreground_fix_2dgs.py']
    p=dict(dataset='2-1',methods=METHODS,models={m:str(WORK/'models'/m) for m in METHODS},datasets={m:str(WORK/'datasets'/m) for m in METHODS},
        baseline_models={m:old['models'][m] for m in ['OFF','COLMAP']},baseline_results=str(OLD/'results'),
        train_names=old['train_names'],test_names=old['test_names'],iterations=30000,seed=0,width=1600,
        common_args=old['common_args'],repair_code_sha256=old['repair_code_sha256'],image_sha256=old['image_sha256'],
        protected_sha256=protected,new_inputs_sha256=inputs,experiment_code_sha256={n:sha(ROOT/n) for n in scripts},
        new_training_runs=2,rounds_per_method=1,initial_points=checks['gate_and_hull_points'],changed_points=checks['corrected_exported_points'],
        method='Original SfM information-metric triangulation constrained by training silhouettes; poses/K/features fixed; no trained rendering proxy.',
        ablation='GATE and HULL share exact point IDs, count, order, RGB, cameras and images; only initial xyz differs. FILTER diagnostic is not trained.',
        evaluation_policy='Final30k only, no checkpoint selection, no test-RGB alignment, no retuning on evaluation metrics.',
        limits='Single scene, seed0, reused scene evaluation. Full80-image initial SfM geometry. Hull feasibility is not independent 3D accuracy; do not claim millimeters or 99% Metashape equivalence.')
    save(target,p);verify(p);return p
def command(args,label):
    log=OUT/'logs'/(label+'.log');log.parent.mkdir(exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    with log.open('ab') as f:
        child=subprocess.Popen(list(map(str,args)),cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
        save(log.with_suffix('.command.json'),dict(pid=child.pid,args=list(map(str,args))));rc=child.wait()
    if rc:raise RuntimeError(f'{label} failed ({rc}); inspect {log}')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--prepare-only',action='store_true');a=parser.parse_args()
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);p=prepare()
    if a.prepare_only:status('prepared');return
    for method in METHODS:
        model=Path(p['models'][method]);done=OUT/f'trained_{method}.json'
        if not done.exists():
            args=[sys.executable,CODE/'train.py','-s',p['datasets'][method],'-m',model]+p['common_args']
            if (model/'cfg_args').exists():
                if not a.resume:raise RuntimeError('Partial training; inspect checkpoint then use --resume')
                candidates=sorted(model.glob('chkpnt*.pth'),key=lambda q:int(q.stem[6:]))
                if not candidates:raise RuntimeError('No checkpoint; never restart existing run at0')
                last=candidates[-1]
                if int(last.stem[6:])>=30000:
                    assert (model/'point_cloud/iteration_30000/point_cloud.ply').exists();args=None
                else:args+=['--start_checkpoint',last]
            if args:status('training',method=method,target=30000);command(args,'train_'+method)
            f=model/'point_cloud/iteration_30000/point_cloud.ply';assert f.exists()
            save(done,dict(complete=True,model=str(model),iteration=30000,model_sha256=sha(f)))
        if not(OUT/'results'/method/'metrics.json').exists():
            status('evaluating',method=method);command([sys.executable,ROOT/'tools/evaluate_surface_sfm_2dgs.py','--method',method],'evaluate_'+method)
    verify(p)
    save(OUT/'audit.json',dict(passed=True,protected_hashes_match=True,original_models_preserved=True,
        original_ALIKED_LightGlue_and_measurements_preserved=True,same_2dgs_code_and_args=True,
        cameras_intrinsics_and_images_unchanged=True,point_count_order_color_matched=True,iterations=30000,new_training_runs=2))
    status('metrics_complete',visual_review_pending=True)
if __name__=='__main__':
    try:main()
    except BaseException:status('failed',error=traceback.format_exc());raise
