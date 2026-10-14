"""Strict provenance for cached tiled-ALIKED features and Keypt2Subpx patches.

The packaged frontend itself is replayed, with its ONNX session wrapped to
retain tile provenance.  All cached keypoints/descriptors/scores must agree
exactly, in the original feature-ID order, before any patch is returned.
Original images, feature caches and ONNX weights are read-only.  The score map
is exposed by adding a graph output to an in-memory ONNX model only.

Patch coordinates follow the pinned official GlueFactory ALIKED convention:
official_xy=(tile_native_xy-0.5)*[W/(W-1),H/(H-1)].  Its refiner uses integer
truncation and zero padding, not bilinear interpolation.  A learned offset in
that convention maps back with [(W-1)/W,(H-1)/H].
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORE_OUTPUT = '/Sigmoid_output_0'
PATCH_SIZE = 11
PROTOCOL = 'tiled-aliked-exact-keypt2subpx-patches-v1'
GLUEFACTORY_COMMIT = '4a8283517fb77e7527e278e233df502f8b7a3266'


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


@dataclass
class PatchSession:
    ort_session: object
    model_path: Path
    model_sha256: str
    original_outputs: list[str]
    provider: str
    threads: int


def create_session(model_path=None, provider='cuda', threads=4):
    """One reusable ALIKED session; the weight file is never rewritten."""
    import onnx
    import onnxruntime as ort
    if provider not in ('cpu','cuda'):raise ValueError(provider)
    if provider=='cuda':
        # Load the installed Torch CUDA/cuDNN DLLs before ORT on Windows.
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is required for the requested replay provider')
    path=Path(model_path or ROOT/'weights/aliked-n32.onnx').resolve()
    original_hash=_sha(path)
    model=onnx.load(str(path))
    outputs=[o.name for o in model.graph.output]
    if outputs!=['keypoints','descriptors','scores']:
        raise ValueError('Unexpected original ALIKED outputs: '+repr(outputs))
    if not any(SCORE_OUTPUT in node.output for node in model.graph.node):
        raise ValueError('ALIKED graph does not contain the expected sigmoid score map')
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        SCORE_OUTPUT,onnx.TensorProto.FLOAT,[1,1,'height','width']))
    so=ort.SessionOptions();so.intra_op_num_threads=int(threads);so.inter_op_num_threads=1
    so.log_severity_level=3
    providers=[('CUDAExecutionProvider',{'arena_extend_strategy':'kSameAsRequested',
                 'cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'0',
                 'gpu_mem_limit':str(12*1024**3)}),'CPUExecutionProvider'] if provider=='cuda' else ['CPUExecutionProvider']
    session=ort.InferenceSession(model.SerializeToString(),sess_options=so,providers=providers)
    if provider=='cuda' and 'CUDAExecutionProvider' not in session.get_providers():
        raise RuntimeError('ALIKED CUDA provider did not load; exact replay cannot fall back silently')
    if _sha(path)!=original_hash:raise RuntimeError('Original ONNX file changed')
    return PatchSession(session,path,original_hash,outputs,provider,int(threads))


def _frontend():
    directory=ROOT/'frontend'
    sys.path.insert(0,str(directory))
    try:
        spec=importlib.util.spec_from_file_location('_strict_patch_tiled_frontend',directory/'tiled_aliked.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _starts(length,tile):
    if length<=tile:return [0]
    return sorted(set([*range(0,length-tile+1,tile-384),length-tile]))


def _official_patches(tensor, official_xy):
    """Exact 11x11 integer-corner/zero-pad convention of official utils.py."""
    import torch
    value=torch.as_tensor(np.ascontiguousarray(tensor),dtype=torch.float32)
    if value.ndim!=3 or value.shape[0]!=1:raise ValueError('Expected a single-channel CHW patch source')
    value=torch.nn.functional.pad(value,[5,5,5,5],mode='constant',value=0.)
    # Official forward computes (keypoint - [5,5] + 5).int(). Preserve this
    # arithmetic rather than substituting rounded OpenCV sample coordinates.
    points=torch.as_tensor(np.ascontiguousarray(official_xy),dtype=torch.float32)
    corner=(points-torch.tensor([5.,5.],dtype=torch.float32)+5).int().long()
    corner[:,0]=corner[:,0].clamp(0,value.shape[2]-1-PATCH_SIZE)
    corner[:,1]=corner[:,1].clamp(0,value.shape[1]-1-PATCH_SIZE)
    yy,xx=torch.meshgrid(torch.arange(PATCH_SIZE),torch.arange(PATCH_SIZE),indexing='ij')
    x=corner[:,0,None,None]+xx[None]
    y=corner[:,1,None,None]+yy[None]
    return value[0,y,x].unsqueeze(1).numpy(),corner.numpy()


class _Replay:
    def __init__(self, session, width, height):
        self.session=session;self.width=width;self.height=height
        self.tiles=[(x,y,min(1536,width-x),min(1024,height-y))
                    for y in _starts(height,1024) for x in _starts(width,1536)]
        self.rows=[];self.call_count=0

    def run(self, outputs, inputs):
        if outputs is not None:raise ValueError('The frozen tiled frontend output contract changed')
        tile_index=self.call_count
        if tile_index>=len(self.tiles):raise ValueError('Unexpected additional frontend tile call')
        x0,y0,cw,ch=self.tiles[tile_index]
        rgb=inputs['image']
        if rgb.shape!=(1,3,ch,cw):raise ValueError('Frontend tile traversal or shape changed')
        values=self.session.ort_session.run(self.session.original_outputs+[SCORE_OUTPUT],inputs)
        kp,desc,score=values[:3]
        kp=kp.reshape(-1,2).astype(np.float32)
        desc=desc.reshape(-1,desc.shape[-1]);score=score.ravel()
        if desc.shape[1]!=128:raise ValueError('Expected ALIKED 128-D descriptors')
        if kp.size and kp.max()<=2 and kp.min()>=-2:
            # Use the original normalized output, with the exact pinned
            # GlueFactory float32 operation order; avoid a native round-trip.
            official_all=np.array([cw,ch],np.float32)*(kp+1)/2.0
            kp=(kp+1)*np.array([cw-1,ch-1],np.float32)*.5+.5
        else:
            official_all=(kp-.5)*np.array([cw/(cw-1),ch/(ch-1)],np.float32)
        valid=(kp[:,0]>=(96 if x0>0 else 0))&(kp[:,0]<(cw-96 if x0+cw<self.width else cw))
        valid&=(kp[:,1]>=(96 if y0>0 else 0))&(kp[:,1]<(ch-96 if y0+ch<self.height else ch))
        local=kp[valid].copy();global_xy=kp.copy();global_xy[:,0]+=x0;global_xy[:,1]+=y0
        official=official_all[valid].copy()
        scoremap=np.asarray(values[3],np.float32)
        if scoremap.shape!=(1,1,ch,cw):raise ValueError(f'Score map has unexpected scale: {scoremap.shape}')
        if not np.isfinite(scoremap).all() or scoremap.min()<0 or scoremap.max()>1:
            raise ValueError('Invalid sigmoid probability map')
        import torch
        gray=(torch.from_numpy(rgb[0])*torch.tensor([.299,.587,.114],dtype=torch.float32).view(3,1,1)).sum(0,keepdim=True).numpy()
        raw,corner=_official_patches(gray,official)
        saliency,score_corner=_official_patches(scoremap[0],official)
        if not np.array_equal(corner,score_corner):raise AssertionError('Patch centers diverged')
        n=len(local)
        self.rows.append(dict(keypoints_native=global_xy[valid],tile_keypoints_native=local,
            descriptors_raw=desc[valid],scores=score[valid],raw_patches=raw,score_patches=saliency,
            keypoints_official=official,patch_corner=corner,
            tile_index=np.full(n,tile_index,np.int32),tile_origin=np.tile([x0,y0],(n,1)).astype(np.int32),
            tile_size=np.tile([cw,ch],(n,1)).astype(np.int32),
            offset_scale_native=np.tile(np.array([(cw-1)/cw,(ch-1)/ch],np.float32),(n,1))))
        self.call_count+=1
        return values[:3]


def _selected_indices(keypoints,scores,limit):
    selected=[];buckets={};radius=3.
    for i in np.argsort(-scores,kind='stable'):
        x,y=np.floor(keypoints[i]/radius).astype(int)
        near=[j for a in range(x-1,x+2) for b in range(y-1,y+2) for j in buckets.get((a,b),[])]
        if any(np.sum((keypoints[i]-keypoints[j])**2)<radius**2 for j in near):continue
        selected.append(i);buckets.setdefault((x,y),[]).append(i)
        if len(selected)>=limit:break
    return np.asarray(selected,dtype=np.int64)


def _exact(name,actual,expected):
    if actual.shape!=expected.shape:
        raise RuntimeError(f'Feature replay {name} shape mismatch: {actual.shape} != {expected.shape}')
    if actual.dtype!=expected.dtype or actual.tobytes()!=expected.tobytes():
        difference=float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)),initial=0))
        unequal=int(np.count_nonzero(actual!=expected))
        raise RuntimeError(f'Feature replay is NOT exact for {name}: {unequal} unequal values, max abs diff={difference:.12g}; refusing patch provenance')


def prepare_image(image_path,feature_npz,session=None,output=None,max_keypoints=4096,min_score=.01):
    """Return all features in original cache order, or a hash-verified saved NPZ.

    Keys include raw_patches/score_patches (N,1,11,11), descriptors (N,128),
    keypoints_native (N,2), keypoints_official (tile-local), offset_scale_native
    (N,2), tile_index/origin/size, feature_ids (N,), and metadata (a dictionary).
    ``output`` is optional; an existing file is only reused when every input
    fingerprint and the extraction/patch protocol match. Never rewrites inputs.
    """
    image_path,feature_npz=Path(image_path).resolve(),Path(feature_npz).resolve()
    session=session or create_session()
    started=time.monotonic()
    fingerprints=dict(image_sha256=_sha(image_path),feature_cache_sha256=_sha(feature_npz),
                      model_sha256=session.model_sha256,module_sha256=_sha(Path(__file__)),
                      tiled_frontend_sha256=_sha(ROOT/'frontend/tiled_aliked.py'),
                      common_frontend_sha256=_sha(ROOT/'frontend/deep_frontend_common.py'))
    request=dict(protocol=PROTOCOL,**fingerprints,max_keypoints=int(max_keypoints),min_score=float(min_score),
                 provider=session.provider,threads=session.threads)
    if output is not None:
        output=Path(output)
        if output.exists():
            with np.load(output,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata_json'].item()))
                if meta.get('request')!=request or not meta.get('all_cached_features_bitwise_equal'):
                    raise ValueError('Existing feature-patch output has incompatible provenance')
                result={name:data[name].copy() for name in data.files if name!='metadata_json'}
            result['metadata']=meta
            return result
    with np.load(feature_npz,allow_pickle=False) as cached:
        old={name:cached[name].copy() for name in ('keypoints','descriptors','scores')}
    if old['keypoints'].shape!=(len(old['scores']),2) or old['descriptors'].shape!=(len(old['scores']),128):
        raise ValueError('Malformed source feature cache')
    # Validate the recorded ALIKED model when the cache retains its shared manifest.
    manifest=feature_npz.parent.parent/'models.json'
    if manifest.exists():
        model_record=json.loads(manifest.read_text(encoding='utf8'))
        recorded=model_record.get('weights',{}).get('aliked-n32.onnx')
        if recorded is not None and recorded!=session.model_sha256:
            raise ValueError('Replay ALIKED model differs from original extraction weights')
    frame=cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if frame is None:raise ValueError('Cannot decode prepared image')
    h,w=frame.shape[:2]
    replay=_Replay(session,w,h)
    front=_frontend()
    actual=front.extract(replay,image_path,SimpleNamespace(max_keypoints=max_keypoints,min_score=min_score))
    if replay.call_count!=len(replay.tiles):raise ValueError('Incomplete original frontend tile replay')
    for name in ('keypoints','descriptors','scores'):_exact(name,getattr(actual,name),old[name])
    all_rows={key:np.concatenate([row[key] for row in replay.rows],axis=0) for key in replay.rows[0]}
    selected=_selected_indices(all_rows['keypoints_native'],all_rows['scores'],max_keypoints)
    result={key:value[selected].copy() for key,value in all_rows.items() if key!='descriptors_raw'}
    result['descriptors']=front.front.normalize_descriptors(all_rows['descriptors_raw'][selected])
    # Check the provenance replay against the actual imported frontend as well
    # as against historical cache values; ties retain original stable tile order.
    for name,field in (('keypoints','keypoints_native'),('descriptors','descriptors'),('scores','scores')):
        _exact('tile provenance '+name,result[field],getattr(actual,name))
    result['feature_ids']=np.arange(len(actual.keypoints),dtype=np.int64)
    if fingerprints['image_sha256']!=_sha(image_path) or fingerprints['feature_cache_sha256']!=_sha(feature_npz):
        raise RuntimeError('An original image or feature cache changed during replay')
    if session.model_sha256!=_sha(session.model_path):raise RuntimeError('Original ALIKED ONNX changed during replay')
    metadata=dict(request=request,complete=True,all_cached_features_bitwise_equal=True,
                  tile_provenance_matches_imported_frontend=True,features=len(actual.keypoints),
                  image=str(image_path),feature_cache=str(feature_npz),tile_count=len(replay.tiles),
                  tiles=[dict(index=i,x=x,y=y,width=cw,height=ch) for i,(x,y,cw,ch) in enumerate(replay.tiles)],
                  native_coordinates='COLMAP +0.5 pixel centers; original cache feature ID order',
                  official_patch_coordinates='(tile_native_xy-0.5)*[W/(W-1),H/(H-1)], integer truncation, zero padding',
                  official_coordinate_operation='float32 [W,H]*(original_normalized_keypoint+1)/2; no native-coordinate round trip',
                  gluefactory_commit=GLUEFACTORY_COMMIT,scoremap_output=SCORE_OUTPUT,
                  scoremap_normalized=False,original_onnx_modified=False,
                  elapsed_seconds=time.monotonic()-started)
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as stream:
            np.savez_compressed(stream,**result,metadata_json=np.array(json.dumps(metadata,ensure_ascii=False)))
    result['metadata']=metadata
    return result


def prepare_images(images,features,output,session=None,max_keypoints=4096,min_score=.01):
    """Yield (image name, saved patch file, metadata), processing one image at a time."""
    images,features,output=Path(images),Path(features),Path(output)
    session=session or create_session()
    for cache in sorted(features.glob('*.npz')):
        source=images/(cache.stem+'.png')
        if not source.exists():raise FileNotFoundError(source)
        destination=output/cache.name
        result=prepare_image(source,cache,session,destination,max_keypoints,min_score)
        yield source.name,destination,result['metadata']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True);p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--provider',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--model',type=Path);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-keypoints',type=int,default=4096);p.add_argument('--min-score',type=float,default=.01)
    a=p.parse_args();session=create_session(a.model,a.provider,a.threads)
    result=prepare_image(a.image,a.features,session,a.output,a.max_keypoints,a.min_score)
    print(json.dumps(result['metadata'],indent=2,ensure_ascii=False))


if __name__=='__main__':main()
