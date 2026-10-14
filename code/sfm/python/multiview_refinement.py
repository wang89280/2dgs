"""DenseSfM released multi-view image refinement connected to native SfM.

ALIKED/LightGlue identities and track membership stay fixed. Only image
measurements change. Released network, local batching/ROI adapter, native BA;
this is not the full DenseSfM pipeline or a reimplementation of its network.
"""
from pathlib import Path
from collections import defaultdict
import sys, time, json, argparse
import numpy as np
import torch
from PIL import Image
from model_io import OBS, digest, save
from image_track_refinement import _image_paths
from densesfm_roi import RoIAlign

VENDOR=Path(__file__).resolve().parents[1]/'third_party/densesfm'
POLICY=dict(image_max_side=1200,resize_multiple=None,max_views=16,track_batch=16,
            seed=666,rounds=1,uniform_ba=True,coordinate_convention='native minus 0.5 before image sampling',
            anchor='median f/depth scale with shared fixed focal length; unique camera observations',
            long_tracks='anchor plus <=15 other views; move anchor once, recrop and freeze for later segments',
            acceptance='finite, within image bounds, and within released network search window; otherwise retain entire original track',
            displacement_hard_limit_px=64,confidence_filter=False,geometry_target_snapping=False,
            heldout_coordinates_used=False,external_colmap_mapper=False)


class Predictor:
    def __init__(self):
        from omegaconf import OmegaConf
        sys.path.insert(0,str(VENDOR))
        from densesfm_network import MultiviewMatcher
        torch.manual_seed(POLICY['seed']);torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        cfg=OmegaConf.to_container(OmegaConf.load(VENDOR/'upstream_config.yaml'),resolve=False)['model']['multiview_refinement']
        cfg['backbone']['s2dnet']['pretrained']=None
        self.model=MultiviewMatcher(cfg,test=True).eval()
        self.model.load_state_dict(torch.load(VENDOR/'refinement_weights.pt',map_location='cpu',weights_only=True),strict=True)
        self.model=self.model.cuda()

    @torch.inference_mode()
    def predict(self,patches,xy,scales,movable):
        """Unpadded tracks of equal length: patches[T,V,3,35,35], xy[T,V,2]."""
        patches=torch.as_tensor(patches,device='cuda',dtype=torch.float32)
        xy=torch.as_tensor(xy,device='cuda',dtype=torch.float32)
        scales=torch.as_tensor(scales,device='cuda',dtype=torch.float32)
        movable=torch.as_tensor(movable,device='cuda',dtype=torch.bool)[None]
        t,v=patches.shape[:2]
        features=self.model.backbone(patches.flatten(0,1),None,True,None,None)[-1]
        features=features.flatten(2).transpose(1,2).reshape(1,t,v,225,128)
        a,b=features[:,:,0],features[:,:,1:]
        mask=torch.ones((1,t,v-1),device='cuda',dtype=torch.bool)
        data=dict(W=15,scales_origin_to_fine_query=scales[None,:,0],
                  scales_origin_to_fine_reference=scales[None,:,1:].transpose(1,2))
        fa,fb=self.model.fine_transformer(a,b,data,query_mask=mask)
        qa,qb,_,_=self.model.gp_matching(fa,fb,a,b,xy[None,:,0],xy[None,:,1:],data,
                                      track_mask=mask,query_movable_mask=movable)
        return torch.cat((qa[:,:,None],qb.transpose(1,2)),dim=2)[0].cpu().numpy()


class ImagePatches:
    def __init__(self,paths,obs):
        self.images=[];self.scales=[];self.sizes=[];self.hashes={}
        self.crop=RoIAlign(35,35,False)
        self.patches=np.empty((len(obs),3,35,35),dtype=np.float32)
        for camera,path in enumerate(paths):
            self.hashes[str(path)]=digest(path)
            with Image.open(path) as f:
                w,h=f.size;s=POLICY['image_max_side']/max(w,h)
                nw,nh=max(2,round(w*s)),max(2,round(h*s))
                image=np.asarray(f.convert('RGB').resize((nw,nh),Image.Resampling.LANCZOS)).copy()
            self.images.append(image);self.scales.append([w/nw,h/nh]);self.sizes.append([w,h])
            ix=np.flatnonzero(obs['camera']==camera)
            xy=np.c_[obs['u'][ix],obs['v'][ix]].astype(np.float32)-.5
            self.patches[ix]=self.extract(camera,xy)
            if (camera+1)%10==0:print(f'Image crops {camera+1}/{len(paths)}',flush=True)
        self.scales=np.asarray(self.scales,dtype=np.float32)
        self.sizes=np.asarray(self.sizes)

    @torch.inference_mode()
    def extract(self,camera,xy):
        im=torch.from_numpy(self.images[camera]).permute(2,0,1)[None].cuda().float()/255
        xy=np.asarray(xy,np.float32)/self.scales[camera]
        result=[]
        for begin in range(0,len(xy),512):
            pos=torch.as_tensor(xy[begin:begin+512],device='cuda')
            boxes=torch.cat([pos-17,pos+17],dim=-1)
            result.append(self.crop(im,boxes,torch.zeros(len(pos),device='cuda',dtype=torch.long)).cpu().numpy())
        return np.concatenate(result) if result else np.empty((0,3,35,35),np.float32)


def refine_observations(sfm,images,output):
    sfm,images,output=map(Path,(sfm,images,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);start=time.time()
    core=('observations.bin','camera_ids.bin','poses.bin','points.bin','calibration.bin','active_points.bin')
    before={n:digest(sfm/n) for n in core}
    obs=np.fromfile(sfm/'observations.bin',OBS)
    ids=np.fromfile(sfm/'camera_ids.bin','<u4')
    poses=np.fromfile(sfm/'poses.bin','<f8').reshape(-1,12)
    xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3)
    if len(set(zip(obs['camera'],obs['track'])))!=len(obs):raise ValueError('Duplicate track-image observation')
    paths=_image_paths(sfm,images,ids)
    predictor=Predictor();cache=ImagePatches(paths,obs)
    rawxy=np.c_[obs['u'],obs['v']].astype(np.float32)-.5
    refined=rawxy.copy();bytrack=defaultdict(list)
    for i,t in enumerate(obs['track']):bytrack[int(t)].append(i)
    groups=defaultdict(list);anchors=[]
    # Geometric depth is only used to select a representative existing anchor.
    # No reprojection is used as an image target or test observation.
    for tid,rows in bytrack.items():
        rows=np.asarray(rows,dtype=int)
        if len(rows)<2:continue
        ci=obs['camera'][rows]
        z=np.einsum('nij,nj->ni',poses[ci,3:].reshape(-1,3,3),xyz[tid]-poses[ci,:3])[:,2]
        if np.any(z<=0):raise ValueError('Nonpositive input depth')
        order=np.argsort(1/(z+1e-4),kind='stable');anchor=rows[order[len(rows)//2]]
        rest=rows[rows!=anchor]  # deterministic native order
        anchors.append(int(anchor))
        for stage,begin in enumerate(range(0,len(rest),15)):
            segment=np.r_[anchor,rest[begin:begin+15]]
            groups[(stage,len(segment))].append(segment)
    completed=0;segments=sum(map(len,groups.values()))
    for (stage,views),items in sorted(groups.items()):
        for begin in range(0,len(items),POLICY['track_batch']):
            rows=np.asarray(items[begin:begin+POLICY['track_batch']])
            # A long track's repeated anchor is cropped at its updated pixel.
            patches=cache.patches[rows].copy()
            if stage:
                for camera in np.unique(obs['camera'][rows[:,0]]):
                    which=np.flatnonzero(obs['camera'][rows[:,0]]==camera)
                    patches[which,0]=cache.extract(int(camera),refined[rows[which,0]])
            pred=predictor.predict(patches,refined[rows],cache.scales[obs['camera'][rows]],
                                   np.full(len(rows),stage==0))
            if stage:
                if not np.allclose(pred[:,0],refined[rows[:,0]],atol=1e-5):raise ValueError('Repeated anchor moved')
                refined[rows[:,1:]]=pred[:,1:]
            else:refined[rows]=pred
            completed+=len(rows)
            if begin%(POLICY['track_batch']*40)==0:
                print(f'Multi-view segments {completed}/{segments}, stage={stage}, views={views}',flush=True)
                save(output/'progress.json',dict(segments=completed,total=segments,elapsed=time.time()-start))
    shift=refined-rawxy
    bound=7*cache.scales[obs['camera']]+.002
    bound[anchors]=3*cache.scales[obs['camera'][anchors]]+.002
    bad=(~np.isfinite(refined).all(1)) | np.any(np.abs(shift)>bound,axis=1)
    bad|=np.any((refined<0)|(refined>cache.sizes[obs['camera']]-1),axis=1)
    bad|=np.linalg.norm(shift,axis=1)>64
    rejected=np.unique(obs['track'][bad])
    reset=np.isin(obs['track'],rejected);refined[reset]=rawxy[reset]
    after=obs.copy();after['u']=refined[:,0]+.5;after['v']=refined[:,1]+.5
    after.tofile(output/'observations_refined.bin')
    np.ones(len(obs),dtype='<f4').tofile(output/'observation_sigma.bin')
    delta=np.hypot(after['u'].astype(float)-obs['u'],after['v'].astype(float)-obs['v'])
    if before!={n:digest(sfm/n) for n in core}:raise ValueError('Input model changed')
    if any(digest(p)!=cache.hashes[str(p)] for p in paths):raise ValueError('Input image changed')
    result=dict(complete=True,method='DenseSfM released multiview refiner + native fixed-K BA',policy=POLICY,
                input_sha256=before,image_sha256=cache.hashes,
                weights_sha256=digest(VENDOR/'refinement_weights.pt'),
                observations=len(obs),tracks=len(bytrack),segments=segments,
                rejected_tracks=len(rejected),rejected_observations=int(reset.sum()),
                changed_observations=int((delta>1e-5).sum()),
                displacement_px=dict(mean=float(delta.mean()),median=float(np.median(delta)),
                                     p90=float(np.quantile(delta,.9)),max=float(delta.max())),
                strict_weights=True,source_unchanged=True,elapsed_seconds=time.time()-start)
    save(output/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('sfm','images','output'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();print(json.dumps(refine_observations(a.sfm,a.images,a.output),indent=2))
