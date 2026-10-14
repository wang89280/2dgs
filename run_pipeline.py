"""Windows images -> final improved fixed-calibration SfM -> WSL 2D Gaussian Splatting."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
SFM = ROOT / 'code/sfm'
GS = ROOT / 'code/2dgs'

# Edit config.json for default settings; explicit CLI options take precedence.
def load_defaults():
    settings=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
    for key in ['images','output','calibration']:
        if settings.get(key) is not None:
            value=Path(settings[key])
            settings[key]=value if value.is_absolute() else ROOT/value
    return settings

DEFAULTS = load_defaults()

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def wslpath(path, distro):
    # Argument lists are used throughout; no generated shell command is executed.
    r = subprocess.run(['wsl.exe', '-d', distro, '--exec', 'wslpath', '-a', '-u', path.as_posix()],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError('WSL path conversion failed: '+r.stderr.decode('utf-8',errors='replace'))
    return r.stdout.decode('utf-8').strip()

def wsl_command(args, directory, command):
    return ['wsl.exe', '-d', args.distro, '--cd', directory, '--exec', *map(str, command)]

def preflight(args):
    import importlib
    for module in ['numpy', 'cv2', 'onnxruntime', 'pycolmap', 'scipy', 'PIL']:
        importlib.import_module(module)
    import onnxruntime
    providers = onnxruntime.get_available_providers()
    if args.provider == 'cuda' and 'CUDAExecutionProvider' not in providers:
        raise RuntimeError('ONNX CUDA provider is unavailable. Configure it or choose --provider cpu.')
    for name in ['bin/initialize_cameras.exe', 'bin/incremental_sfm.exe',
                 'bin/rebuild_tracks.exe', 'bin/maintain_tracks.exe',
                 'weights/aliked-n32.onnx', 'weights/aliked-lightglue.onnx']:
        if not (SFM/name).is_file(): raise FileNotFoundError(SFM/name)
    if args.point_refinement=='robust' and not (SFM/'bin/triangulate_anchors.exe').is_file():raise FileNotFoundError(SFM/'bin/triangulate_anchors.exe')
    if args.image_refinement != 'off':
        for name in ('image_refinement_stage.py','python/image_track_refinement.py','bin/refine_image_observations.exe'):
            if not (SFM/name).is_file():raise FileNotFoundError(SFM/name)
    if args.image_refinement=='learned':
        import torch,onnx
        if not torch.cuda.is_available():raise RuntimeError('The experimental learned refiner requires this Windows CUDA torch environment.')
        if args.max_keypoints!=4096:raise ValueError('The experimental learned refiner currently replays 4096-keypoint tiled extraction.')
        for name in ('python/feature_patch_source.py','python/learned_refinement_predictor.py','python/learned_track_refinement.py',
                     'third_party/keypt2subpx/SOURCE_MANIFEST.json','third_party/keypt2subpx/pretrained/k2s_aliked_pretrained.pth'):
            if not (SFM/name).is_file():raise FileNotFoundError(SFM/name)
    if args.image_refinement=='multiview':
        import torch, torchvision, einops, omegaconf, loguru, kornia
        if not torch.cuda.is_available():raise RuntimeError('Multiview refinement requires Windows CUDA PyTorch.')
        for name in ('multiview_refinement_stage.py','python/multiview_refinement.py',
                     'python/multiview_consistency.py','python/densesfm_roi.py',
                     'bin/refine_multiview_observations.exe','third_party/densesfm/refinement_weights.pt',
                     'third_party/densesfm/upstream_config.yaml'):
            if not (SFM/name).is_file():raise FileNotFoundError(SFM/name)
    result = dict(windows_python=sys.executable, onnx_providers=providers)
    if not args.sparse_only:
        probe = wslpath(ROOT/'tools/check_2dgs_runtime.py', args.distro)
        gs = wslpath(GS, args.distro)
        cmd = wsl_command(args, gs, [args.wsl_python, '-B', probe, '--code', gs])
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode:
            raise RuntimeError('WSL 2DGS dependency check failed: '+r.stderr.decode('utf-8',errors='replace'))
        result['wsl_runtime'] = json.loads(r.stdout.decode('utf-8'))
    return result

def execute(output, label, cmd, cwd):
    log = output/'logs'/f'{label}.log'
    save(output/'status.json', dict(stage=label, running=True, log=str(log)))
    save(output/'logs'/f'{label}_command.json', dict(argv=cmd, cwd=str(cwd)))
    print(f'[{label}] {log}', flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    with log.open('wb') as f:
        result = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f'{label} failed (exit {result.returncode}); see {log}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images', type=Path)
    p.add_argument('--output', type=Path, help='New result directory; default: package outputs/<image-folder>_<timestamp>')
    p.add_argument('--calibration', type=Path)
    p.add_argument('--provider', choices=['cpu','cuda'])
    p.add_argument('--threads', type=int)
    p.add_argument('--max-keypoints', type=int)
    p.add_argument('--pairing', choices=['sequential','exhaustive'])
    p.add_argument('--window', type=int)
    p.add_argument('--initial-pairs', type=int)
    p.add_argument('--stationary-radius-px', type=float)
    p.add_argument('--width', type=int)
    p.add_argument('--eval', action=argparse.BooleanOptionalAction, help='Every eighth sorted view held out for 2DGS rendering evaluation')
    p.add_argument('--iterations', type=int)
    p.add_argument('--sparse-only', action=argparse.BooleanOptionalAction)
    p.add_argument('--skip-render', action=argparse.BooleanOptionalAction, help='Train and save Gaussians without exporting rendered views')
    p.add_argument('--mesh', action=argparse.BooleanOptionalAction, help='Also extract mesh with the local 2DGS default automatic scale')
    p.add_argument('--distro')
    p.add_argument('--wsl-python')
    p.add_argument('--keep-intermediates', action=argparse.BooleanOptionalAction, help='Keep sparse scratch files for debugging')
    p.add_argument('--track-recovery', action=argparse.BooleanOptionalAction, help='Enable graph track recovery (default true); false is an ablation within the same codebase')
    p.add_argument('--image-refinement', choices=['off','localization','weighted','learned','multiview'], default='off',
                   help='Experimental image localization refinement; off by default until validated')
    p.add_argument('--point-refinement',choices=['off','robust'],default='off',help='Optional robust point triangulation with fixed cameras')
    p.add_argument('--track-partition',choices=['off','view-consensus'],default='off',help='Validated multi-landmark track partition; fixed poses/K and original pixels')
    p.add_argument('--check-env', action='store_true', help='Check dependencies only; do not create outputs or train')
    p.add_argument('--dry-run', action='store_true', help='Print commands only; no reconstruction, training, or output writes')
    p.set_defaults(**DEFAULTS)
    a = p.parse_args()
    if os.name != 'nt': p.error('Run this entry from Windows; it invokes WSL for 2DGS.')
    if a.iterations < 1 or a.threads < 1 or a.window < 1 or a.width < 0:
        p.error('Invalid numeric option')
    if a.max_keypoints < 128 or not 1 <= a.initial_pairs <= 3160 or not 0 <= a.stationary_radius_px < float('inf'):
        p.error('Invalid SfM option')
    if a.mesh and (a.skip_render or a.sparse_only): p.error('--mesh requires the rendering stage')
    if a.image_refinement not in ('off','localization','weighted','learned','multiview'): p.error('Invalid image_refinement in config.json')
    if a.point_refinement not in ('off','robust'):p.error('Invalid point_refinement in config.json')
    if a.track_partition not in ('off','view-consensus'):p.error('Invalid track_partition in config.json')
    if a.check_env:
        print(json.dumps(preflight(a), ensure_ascii=False, indent=2)); return
    if not a.images: p.error('Set config.json images or pass --images')
    if a.output is None:
        a.output = ROOT / 'outputs' / f'{a.images.name}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    images, out, cal = a.images.resolve(), a.output.resolve(), a.calibration.resolve()
    if not images.is_dir() or not cal.is_file(): p.error('Input images/calibration not found')
    if out.exists(): p.error('Output already exists; choose a new directory')
    if out==images or images in out.parents: p.error('Output must be outside the input image directory')
    sparse = out/'sparse'
    dataset, model = sparse/'2dgs', out/'gaussian'
    sfm = [sys.executable, '-B', str(SFM/'run.py'), '--images', str(images), '--output', str(sparse),
           '--calibration', str(cal), '--provider', a.provider, '--threads', str(a.threads),
           '--pairing', a.pairing, '--window', str(a.window), '--width', str(a.width),
           '--max-keypoints', str(a.max_keypoints), '--initial-pairs', str(a.initial_pairs),
           '--stationary-radius-px', str(a.stationary_radius_px), '--image-refinement', a.image_refinement,'--point-refinement',a.point_refinement,'--track-partition',a.track_partition]
    if a.eval: sfm.append('--eval')
    if a.keep_intermediates: sfm.append('--keep-intermediates')
    sfm.append('--track-recovery' if a.track_recovery else '--no-track-recovery')
    stages = [('sparse', sfm, str(SFM))]
    if not a.sparse_only:
        gs, ds, md = [wslpath(x,a.distro) for x in [GS,dataset,model]]
        launcher = wslpath(ROOT/'tools/launch_2dgs.py',a.distro)
        checkpoints = sorted({min(7000,a.iterations), a.iterations})
        train = [a.wsl_python, '-B', launcher, '--code', gs, '--entry', 'train', '--', '-s', ds, '-m', md,
                 '--resolution','1','--data_device','cpu','--iterations',str(a.iterations),
                 '--test_iterations',*map(str,checkpoints),'--save_iterations',*map(str,checkpoints),
                 '--autosave_iterations','10000','--port','0']
        if a.eval: train.append('--eval')
        stages.append(('train', wsl_command(a,gs,train),str(ROOT)))
        if not a.skip_render:
            render = [a.wsl_python,'-B',launcher,'--code',gs,'--entry','render','--','-s',ds,'-m',md,
                      '--iteration',str(a.iterations),'--resolution','1','--data_device','cpu']
            if a.eval: render.append('--eval')
            if not a.mesh: render.append('--skip_mesh')
            stages.append(('render',wsl_command(a,gs,render),str(ROOT)))
    plan = dict(input_images=str(images), output=str(out), stages=[dict(name=n,argv=c,cwd=d) for n,c,d in stages],
                options={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                fixed_intrinsics=True, algorithm='final_improved')
    if a.dry_run:
        print(json.dumps(plan,ensure_ascii=False,indent=2)); return
    # Fail missing runtime dependencies before doing the potentially expensive SfM stage.
    runtime = preflight(a)
    out.mkdir(parents=True, exist_ok=False)
    save(out/'pipeline.json',plan); save(out/'runtime.json',runtime)
    start = time.time()
    try:
        for name,cmd,cwd in stages: execute(out,name,cmd,cwd)
        if not (dataset/'sparse/0/images.bin').is_file(): raise RuntimeError('Missing exported dataset')
        if not a.sparse_only:
            final_ply = model/f'point_cloud/iteration_{a.iterations}/point_cloud.ply'
            if not final_ply.is_file(): raise RuntimeError('Missing final trained Gaussian PLY')
        save(out/'complete.json',dict(complete=True,seconds=time.time()-start,dataset=str(dataset),
             gaussian_model=None if a.sparse_only else str(model),iterations=None if a.sparse_only else a.iterations,
             rendered=not a.sparse_only and not a.skip_render,mesh=a.mesh))
        save(out/'status.json',dict(stage='complete',running=False))
        print('Complete: '+str(out),flush=True)
    except BaseException:
        save(out/'failure.json',dict(error=traceback.format_exc()))
        save(out/'status.json',dict(stage='failed',running=False,see='failure.json')); raise

if __name__=='__main__': main()
