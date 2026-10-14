"""Evaluate raw fixed-model renders, with the frozen historical metric definitions."""
import argparse, ast, sys, time
from argparse import Namespace
from pathlib import Path
import numpy as np
from PIL import Image
from run_foreground_fix_2dgs import OUT, OLD, CODE, read, save, sha, cfg
sys.path.insert(0,str(CODE))
import torch
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS

def image(t,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((t.detach().permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)).save(path)

@torch.no_grad()
def evaluate(method,verify_original=False):
    protocol=read(OUT/'protocol.json')
    model=Path(protocol['original_models' if verify_original else 'models'][method])
    output=OUT/'results'/method; args=Namespace(**cfg(model))
    f=model/'point_cloud/iteration_30000/point_cloud.ply';before=sha(f)
    g=GaussianModel(args.sh_degree);scene=Scene(args,g,load_iteration=30000,shuffle=False)
    assert [c.image_name for c in scene.getTestCameras()]==protocol['test_names']
    assert sorted(c.image_name for c in scene.getTrainCameras())==sorted(protocol['train_names'])
    assert all(c.gt_mask is None for c in scene.getTestCameras())
    pipe=Namespace(convert_SHs_python=False,compute_cov3D_python=False,depth_ratio=0.,debug=False)
    bg=torch.zeros(3,device='cuda');rows=[]
    lpips=None if verify_original else LPIPS('vgg').eval().cuda()
    for split,views in [('test',scene.getTestCameras()),('train',[] if verify_original else scene.getTrainCameras())]:
        for cam in views:
            pkg=render(cam,g,pipe,bg);pred=pkg['render'].clamp(0,1);gt=cam.original_image.cuda().clamp(0,1)
            assert bool(torch.isfinite(pred).all()) and bool(torch.isfinite(pkg['rend_alpha']).all())
            err=(pred-gt).square();mask=gt.max(0).values>.05;alpha=pkg['rend_alpha'][0]
            r=dict(split=split,name=cam.image_name,psnr_db=float(-10*torch.log10(err.mean().clamp_min(1e-12))),
                psnr_nonblack_db=float(-10*torch.log10(err[:,mask].mean().clamp_min(1e-12))),
                ssim=float(ssim(pred,gt)),nonblack_fraction=float(mask.float().mean()),
                alpha_bg_mean=float(alpha[~mask].mean()),alpha_fg_mean=float(alpha[mask].mean()),
                foreground_visible_fraction=float((pred.max(0).values[mask]>.05).float().mean()))
            if split=='test' and not verify_original:
                yy,xx=torch.where(mask);x0=max(0,int(xx.min())-16);x1=min(gt.shape[2],int(xx.max())+17)
                y0=max(0,int(yy.min())-16);y1=min(gt.shape[1],int(yy.max())+17)
                r.update(crop_box_xyxy=[x0,y0,x1,y1],ssim_crop=float(ssim(pred[:,y0:y1,x0:x1],gt[:,y0:y1,x0:x1])),
                    lpips_vgg=float(lpips(pred[None]*2-1,gt[None]*2-1)),
                    lpips_vgg_crop=float(lpips(pred[None,:,y0:y1,x0:x1]*2-1,gt[None,:,y0:y1,x0:x1]*2-1)))
                for name,t in [('renders',pred),('gt',gt),('alpha',alpha.expand(3,-1,-1)),('error_x5',(pred-gt).abs()*5)]:
                    image(t,output/'test'/name/(cam.image_name+'.png'))
            rows.append(r)
    assert sha(f)==before
    if verify_original:
        original=read(OLD/'results'/method/'metrics.json');lookup={r['name']:r for r in original['rows'] if r['split']=='test'}
        differences={k:max(abs(r[k]-lookup[r['name']][k]) for r in rows) for k in ['psnr_db','psnr_nonblack_db','ssim']}
        assert max(differences.values())<1e-5,differences
        save(OUT/'renderer_equivalence.json',dict(passed=True,method=method,model_unchanged=True,differences=differences))
        print('Original renderer equivalence PASS',differences,flush=True);return
    means={}
    for split in ['train','test']:
        rr=[r for r in rows if r['split']==split]
        keys=[k for k,v in rr[0].items() if isinstance(v,(int,float))]
        means[split]={k:float(np.mean([r[k] for r in rr])) for k in keys}
    save(output/'metrics.json',dict(method=method,iteration=30000,model=str(model),gaussians=len(g.get_xyz),
        rows=rows,means=means,foreground_definition='GT max RGB >0.05, same as historical evaluator',
        raw_render_no_mask_compositing=True,model_sha256=before,model_unchanged=True,
        max_scale=float(g.get_scaling.max()),parameters_finite=all(bool(torch.isfinite(getattr(g,n)).all()) for n in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest'])))
    print(method,means,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['COLMAP','OFF','ON'],required=True);p.add_argument('--verify-original',action='store_true')
    a=p.parse_args();evaluate(a.method,a.verify_original)
