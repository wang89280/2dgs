"""Images -> ALIKED/LightGlue -> final fixed-intrinsic SfM -> complete 2DGS dataset."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
from common import ROOT,save,sha,prepare,command,cleanup_intermediates

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True,help='3-80 images, 3072x2048, same calibrated camera')
    p.add_argument('--output',type=Path,required=True,help='New output directory (must not exist)')
    p.add_argument('--calibration',type=Path,default=ROOT/'config/seed_camera_opencv.yml')
    p.add_argument('--provider',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--max-keypoints',type=int,default=4096)
    p.add_argument('--pairing',choices=['sequential','exhaustive'],default='sequential');p.add_argument('--window',type=int,default=10)
    p.add_argument('--initial-pairs',type=int,default=200);p.add_argument('--stationary-radius-px',type=float,default=4)
    p.add_argument('--width',type=int,default=1600,help='Export image width; 0 keeps original resolution')
    p.add_argument('--keep-intermediates',action='store_true',help='Keep initialization, track reconstruction and feature caches for debugging')
    p.add_argument('--track-recovery',action=argparse.BooleanOptionalAction,default=True,help='Ablation: disable graph track rebuilding/merging/completion; keep BA and ordinary filtering')
    p.add_argument('--image-refinement',choices=['off','localization','weighted','learned','multiview'],default='off',help='Experimental image localization refinement; off by default until validated')
    p.add_argument('--point-refinement',choices=['off','robust'],default='off',help='Optional multi-start Cauchy triangulation; fixed cameras and original observations')
    p.add_argument('--track-partition',choices=['off','view-consensus'],default='off',help='Optional validated multi-landmark track partition for orbital capture')
    p.add_argument('--eval',action='store_true',help='Export train-only point colors for the 2DGS every-8th-view evaluation split')
    a=p.parse_args();out=a.output.resolve();images=a.images.resolve();calibration=a.calibration.resolve()
    if os.name!='nt':p.error('The packaged native SfM binaries require Windows x64; exported dataset is portable to Linux/WSL 2DGS.')
    if not images.is_dir() or not calibration.is_file():p.error('Images or calibration path does not exist')
    if out.exists():p.error('Output exists; choose a new directory. Inputs and existing outputs are never overwritten.')
    if images==out or images in out.parents:p.error('Output must be outside the input image directory')
    if a.threads<1 or a.window<1 or a.max_keypoints<128 or a.width<0 or not 1<=a.initial_pairs<=3160:p.error('Invalid numeric option')
    if a.point_refinement=='robust' and not (ROOT/'bin/triangulate_anchors.exe').is_file():p.error('Missing triangulate_anchors.exe')
    if a.image_refinement!='off':
        for name in ('image_refinement_stage.py','python/image_track_refinement.py','bin/refine_image_observations.exe'):
            if not (ROOT/name).is_file():p.error('Image refinement dependency is missing: '+str(ROOT/name))
    if a.image_refinement=='learned':
        import torch,onnx
        if not torch.cuda.is_available():p.error('The experimental learned refiner requires CUDA torch.')
        if a.max_keypoints!=4096:p.error('The experimental learned refiner currently replays the 4096-keypoint tiled extraction protocol.')
    if a.image_refinement=='multiview':
        import torch, torchvision, einops, omegaconf, loguru, kornia
        if not torch.cuda.is_available():p.error('Multiview refinement requires CUDA PyTorch.')
        for name in ('multiview_refinement_stage.py','python/multiview_refinement.py',
                     'python/multiview_consistency.py','python/densesfm_roi.py',
                     'bin/refine_multiview_observations.exe','third_party/densesfm/refinement_weights.pt'):
            if not (ROOT/name).is_file():p.error('Missing '+str(ROOT/name))
    os.environ.update(OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
    from calibration import convert
    out.mkdir(parents=True);start=time.time()
    try:
        params=convert(calibration,out/'shared/calibration_fixed.bin')
        rows=prepare(images,out)
        save(out/'config.json',dict(images=str(images),calibration=str(calibration),camera_params=params.tolist(),
            options=vars(a),algorithm='final_improved',fixed_intrinsics=True,calibration_assumption='Input camera/crop/focus settings must match calibration'))
        command(out,'frontend',[sys.executable,ROOT/'frontend_stage.py','--output',out,'--provider',a.provider,
            '--threads',a.threads,'--max-keypoints',a.max_keypoints,'--pairing',a.pairing,'--window',a.window,
            '--observation-sigma','4','--stationary-radius-px',a.stationary_radius_px],a.threads)
        command(out,'backend',[sys.executable,ROOT/'backend_stage.py','--output',out,'--threads',a.threads,'--initial-pairs',a.initial_pairs,
            '--track-recovery' if a.track_recovery else '--no-track-recovery','--image-refinement',a.image_refinement,'--point-refinement',a.point_refinement,'--track-partition',a.track_partition],a.threads)
        export=[sys.executable,ROOT/'export_2dgs.py','--sfm',out/'model/sfm','--images',out/'shared/images',
            '--output',out/'2dgs','--width',a.width]
        if a.eval:export.append('--eval')
        command(out,'export_2dgs',export,a.threads)
        if not all(sha(Path(r['source']))==r['source_sha256'] for r in rows):raise RuntimeError('Source images changed during execution')
        # Only scratch directories inside this newly created output may be removed.
        # The final model, database, logs, prepared images and exported dataset remain.
        from validate import validate
        save(out/'validation.json',validate(out/'2dgs'))
        if not a.keep_intermediates:
            cleanup_intermediates(out)
        save(out/'complete.json',dict(complete=True,seconds=time.time()-start,dataset=str(out/'2dgs'),
            input_images=len(rows),model=str(out/'model'),image_refinement=a.image_refinement,point_refinement=a.point_refinement,track_partition=a.track_partition,intermediates_retained=a.keep_intermediates,original_images_unchanged=True,train_2dgs_not_started=True))
        save(out/'status.json',dict(stage='complete',dataset=str(out/'2dgs')))
        print('2DGS dataset: '+str(out/'2dgs'),flush=True)
    except Exception:
        save(out/'failure.json',dict(error=traceback.format_exc()));save(out/'status.json',dict(stage='failed',see='failure.json'));raise

if __name__=='__main__':main()
