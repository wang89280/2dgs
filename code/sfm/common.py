from pathlib import Path
import hashlib,json,os,subprocess,time

ROOT=Path(__file__).resolve().parent

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf8')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def cleanup_intermediates(out):
    """Remove only generated scratch data after the final model is validated."""
    import shutil
    out=Path(out).resolve()
    for scratch in (out/'_work',out/'shared/features'):
        resolved=scratch.resolve()
        if not resolved.is_relative_to(out) or resolved==out or scratch.is_symlink():
            raise RuntimeError('Unsafe scratch cleanup path')
        if scratch.exists():shutil.rmtree(scratch)

def command(out,name,args,threads):
    log=out/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
    start=time.time();args=list(map(str,args))
    save(out/'status.json',dict(stage=name,started=start,command=args))
    print(name,flush=True)
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),PYTHONUNBUFFERED='1')
    with log.open('wb') as stream:
        r=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    save(out/'logs'/f'{name}.json',dict(command=args,seconds=time.time()-start,returncode=r.returncode))
    if r.returncode:raise RuntimeError(f'{name} failed with exit {r.returncode}; see {log}')

def prepare(images,out):
    from PIL import Image
    sources=sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    if not 3<=len(sources)<=80:raise ValueError('This pipeline supports 3 to 80 images per reconstruction.')
    target=out/'shared/images';target.mkdir(parents=True)
    rows=[]
    for idx,p in enumerate(sources):
        with Image.open(p) as im:
            if im.size!=(3072,2048):raise ValueError(f'Current geometry thresholds require 3072x2048 input; got {im.size}: {p}')
            if im.getexif().get(274,1)!=1:raise ValueError(f'Bake EXIF orientation into pixels and provide matching calibration first: {p}')
            if 'A' in im.getbands() or 'transparency' in im.info:
                im=Image.alpha_composite(Image.new('RGBA',im.size,(0,0,0,255)),im.convert('RGBA'))
            # New RGB PNG without EXIF, same pixels; source files are read-only.
            q=target/f'{idx+1:04d}.png';Image.fromarray(__import__('numpy').array(im.convert('RGB'))).save(q)
        rows.append(dict(index=idx,source=str(p.resolve()),name=q.name,source_sha256=sha(p),prepared_sha256=sha(q)))
    save(out/'shared/image_manifest.json',rows)
    return rows
