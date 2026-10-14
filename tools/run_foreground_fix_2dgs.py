"""Frozen one-round 2-1 repair trial. Original code, inputs and models are read-only."""
import argparse, ast, datetime, fcntl, hashlib, json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'outputs/latest_2dgs_2-1_20261014'
OUT=ROOT/'outputs/foreground_fix_2dgs_2-1_20261014'
PACKAGE=ROOT/'code/2dgs_foreground_fix'
WORK=Path('/home/wk/01_code/03_gs/2dgs-foreground-fix-20261014')
CODE=WORK/'code'
METHODS=['ON','OFF','COLMAP']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8');tmp.replace(p)
def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def cfg(model):
    tree=ast.parse((model/'cfg_args').read_text(),mode='eval').body
    return {x.arg:ast.literal_eval(x.value) for x in tree.keywords}

def make_mask(path):
    rgb=np.asarray(Image.open(path).convert('RGB'))
    binary=(rgb.max(2)>2).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if n<2:raise ValueError('No foreground: '+str(path))
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));binary=(labels==largest).astype(np.uint8)*255
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(binary);cv2.drawContours(mask,contours,-1,255,cv2.FILLED)
    fraction=float((mask>0).mean());assert .001<fraction<.8,(path,fraction)
    return rgb,mask,fraction

def prepare():
    previous=read(OLD/'protocol.json');protected=dict(previous['protected_sha256'])
    for method in METHODS:
        model=Path(previous['models'][method])
        for name in ['point_cloud/iteration_30000/point_cloud.ply','cfg_args']:
            f=model/name;protected[str(f)]=sha(f)
    for path,expected in protected.items():assert sha(path)==expected, path
    sources={m:Path(cfg(Path(previous['models'][m]))['source_path']) for m in METHODS}
    train=previous['prepared']['training_names'];test=previous['prepared']['test_names']
    assert len(train)==70 and len(test)==10 and not(set(train)&set(test))
    code_files=[p for p in PACKAGE.rglob('*') if p.is_file() and (p.suffix=='.py' or p.name=='LICENSE.md')]
    frozen={str(p.relative_to(PACKAGE)):sha(p) for p in code_files}
    for f in code_files:
        dst=CODE/f.relative_to(PACKAGE);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():assert sha(dst)==sha(f), 'Runtime code differs: '+str(dst)
        else:shutil.copy2(f,dst)
    masks=WORK/'masks_train';masks.mkdir(parents=True,exist_ok=True);mask_meta=[];panels=[]
    image_hashes={}
    for name in train+test:
        hashes={m:sha(sources[m]/'images'/(name+'.png')) for m in METHODS}
        assert len(set(hashes.values()))==1, 'Method images differ: '+name
        image_hashes[name]=hashes['ON']
    for j,name in enumerate(train):
        rgb,mask,fraction=make_mask(sources['ON']/'images'/(name+'.png'))
        dest=masks/(name+'.png')
        if not dest.exists():Image.fromarray(mask).save(dest)
        else:assert np.array_equal(np.asarray(Image.open(dest)),mask)
        mask_meta.append(dict(name=name,foreground_fraction=fraction,sha256=sha(dest)))
        preview=rgb.copy();boundary=cv2.morphologyEx(mask,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        preview[boundary]=[0,255,0]
        im=Image.fromarray(preview);im.thumbnail((300,210));panels.append((name,im))
    assert sorted(p.stem for p in masks.glob('*.png'))==sorted(train)
    for page in range(2):
        canvas=Image.new('RGB',(7*300,5*240),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for j,(name,im) in enumerate(panels[page*35:(page+1)*35]):
            x=j%7*300;y=j//7*240;canvas.paste(im,(x,y+24));draw.text((x+5,y+5),name,fill='black')
        OUT.mkdir(parents=True,exist_ok=True);canvas.save(OUT/f'training_masks_{page+1:02d}.png')
    models={};datasets={};new_inputs={}
    for method,source in sources.items():
        target=WORK/'datasets'/method;(target/'sparse/0').mkdir(parents=True,exist_ok=True)
        for name in ['cameras.txt','images.txt','points3D.ply']:
            src=source/'sparse/0'/name;dst=target/'sparse/0'/name
            if dst.exists():assert sha(dst)==sha(src)
            else:shutil.copy2(src,dst)
            new_inputs[str(dst)]=sha(dst)
        for name,origin in [('images',source/'images'),('masks_train',masks)]:
            link=target/name
            if not link.exists():link.symlink_to(origin.resolve(),target_is_directory=True)
            assert link.resolve()==origin.resolve()
        datasets[method]=str(target);models[method]=str(WORK/'models'/method)
    for meta in mask_meta:new_inputs[str(masks/(meta['name']+'.png'))]=meta['sha256']
    common=['--eval','--resolution','1','--data_device','cpu','--iterations','30000',
        '--test_iterations','30000','--save_iterations','7000','15000','30000',
        '--autosave_iterations','5000','--port','0','--use_masks','--foreground_stability',
        '--lambda_alpha_bg','0.1','--lambda_alpha_fg','0.01','--lambda_normal','0.05',
        '--normal_from_iter','15000','--normal_ramp_iters','5000','--max_screen_radius_fraction','0.25']
    protocol=dict(dataset='2-1',methods=METHODS,models=models,datasets=datasets,original_models=previous['models'],
        iterations=30000,seed=0,train_names=train,test_names=test,width=1600,common_args=common,
        repair_code_sha256=frozen,protected_sha256=protected,new_inputs_sha256=new_inputs,image_sha256=image_hashes,
        mask_rule='Training RGB only: max RGB > 2/255, largest connected component, fill external contour. 3px uncertain band excluded from alpha/normal losses.',
        masks=mask_meta,normalization='Original source geometry/image bytes preserved for each method.',
        scope=previous['interpretation'],new_training_runs=3,rounds_per_method=1,
        evaluation_policy='Final 30000 only; no checkpoint selection or hyperparameter tuning on test images.',
        modifications=['background/foreground alpha loss','unit-normal reliable-foreground loss with 15000-20000 ramp',
            'radius pruning state fix; screen threshold 25% width; post-15000 safety pruning every1000'],
        limits='One scene, seed0. Joint repair package experiment, not isolation of individual repair contributions.')
    p=OUT/'protocol.json'
    if p.exists():assert read(p)==protocol,'Frozen protocol changed'
    else:save(p,protocol)
    import torch,diff_surfel_rasterization._C
    save(OUT/'runtime.json',dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),rasterizer=diff_surfel_rasterization._C.__file__,
        rasterizer_sha256=sha(diff_surfel_rasterization._C.__file__)))
    return protocol

def command(args,label):
    log=OUT/'logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
    start=time.time()
    with log.open('ab') as f:
        child=subprocess.Popen(list(map(str,args)),cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT)
        info=dict(args=list(map(str,args)),pid=child.pid,started=start,log=str(log))
        save(log.with_suffix('.command.json'),info);rc=child.wait()
    info.update(returncode=rc,seconds=time.time()-start);save(log.with_suffix('.command.json'),info)
    if rc:raise RuntimeError(f'{label} failed ({rc}); see {log}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');a=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=prepare()
    if a.prepare_only:status('prepared');print('PREPARED',flush=True);return
    for method in METHODS:
        model=Path(p['models'][method]);done=OUT/f'trained_{method}.json'
        if not done.exists():
            args=[sys.executable,CODE/'train.py','-s',p['datasets'][method],'-m',model]+p['common_args']
            if (model/'cfg_args').exists():
                if not a.resume:raise RuntimeError('Partial training; inspect and use --resume: '+method)
                candidates=sorted(model.glob('chkpnt*.pth'),key=lambda q:int(q.stem[6:]))
                if not candidates:raise RuntimeError('No checkpoint: '+method)
                last=candidates[-1];it=int(last.stem[6:])
                if it>=30000:
                    assert (model/'point_cloud/iteration_30000/point_cloud.ply').exists()
                    args=None
                else:args+=['--start_checkpoint',last]
            if args:
                status('training',method=method,target=30000,model=str(model));command(args,'train_'+method)
            f=model/'point_cloud/iteration_30000/point_cloud.ply';assert f.exists()
            save(done,dict(complete=True,model=str(model),model_sha256=sha(f),iteration=30000))
        target=OUT/'results'/method
        if not(target/'metrics.json').exists():
            status('evaluating',method=method)
            command([sys.executable,ROOT/'tools/evaluate_foreground_fix_2dgs.py','--method',method],'evaluate_'+method)
    protected=dict(p['protected_sha256']);protected.update(p['new_inputs_sha256'])
    changed=[n for n,h in protected.items() if sha(n)!=h]
    code_changed=[n for n,h in p['repair_code_sha256'].items() if sha(CODE/n)!=h or sha(PACKAGE/n)!=h]
    assert not changed and not code_changed,(changed,code_changed)
    save(OUT/'audit.json',dict(passed=True,protected_files=len(protected),original_models_and_code_unchanged=True,
        repair_code_unchanged=True,training_masks_only=True,methods=METHODS,rounds_per_method=1))
    command([sys.executable,ROOT/'tools/report_foreground_fix_2dgs.py'],'report')
    status('metrics_complete',complete=True,visual_review_pending=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
