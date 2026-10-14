"""Same tiled ALIKED-N32 + LightGlue frontend as the final pipeline."""
import argparse,json,sys,time
from common import ROOT,save,sha
sys.path.insert(0,str(ROOT/'frontend'));sys.path.insert(0,str(ROOT/'python'))
import numpy as np
import cv2
import onnxruntime as ort
import pycolmap
import deep_frontend_common as front
from tiled_aliked import extract
from colmap_pipeline_common import import_images_to_database
from stationary_filter import build_tracks_pipeline

def run(out,args):
    shared=out/'shared';db=shared/'database.db';images=shared/'images'
    cfg=json.loads((out/'config.json').read_text(encoding='utf8'))
    options=argparse.Namespace(camera_model='OPENCV',focal_factor=1.4,max_keypoints=args.max_keypoints,min_score=.01)
    ids=import_images_to_database(db,images,options)
    with pycolmap.Database.open(str(db)) as database:
        for c in database.read_all_cameras():
            c.params=np.array(cfg['camera_params']);c.has_prior_focal_length=True;database.update_camera(c)
    cv2.setNumThreads(args.threads)
    so=ort.SessionOptions();so.intra_op_num_threads=args.threads;so.inter_op_num_threads=1
    providers=[('CUDAExecutionProvider',{'arena_extend_strategy':'kSameAsRequested','cudnn_conv_algo_search':'HEURISTIC','cudnn_conv_use_max_workspace':'0','gpu_mem_limit':str(12*1024**3)}),'CPUExecutionProvider'] if args.provider=='cuda' else ['CPUExecutionProvider']
    sessions=[ort.InferenceSession(str(ROOT/'weights'/n),sess_options=so,providers=providers) for n in ('aliked-n32.onnx','aliked-lightglue.onnx')]
    if args.provider=='cuda' and any('CUDAExecutionProvider' not in s.get_providers() for s in sessions):
        raise RuntimeError('CUDA provider did not load. Install compatible CUDA/cuDNN or explicitly use --provider cpu.')
    save(shared/'models.json',dict(weights={p.name:sha(p) for p in (ROOT/'weights').glob('*.onnx')},
        providers=[s.get_providers() for s in sessions],onnxruntime=ort.__version__,pycolmap=pycolmap.__version__))
    cache=shared/'features';cache.mkdir();features={};times=[];start=time.time();paths=sorted(images.glob('*.png'))
    for i,p in enumerate(paths):
        t=time.time();f=extract(sessions[0],p,options)
        if not len(f.keypoints):raise ValueError(f'No features: {p.name}')
        if not np.isfinite(f.keypoints).all():raise ValueError('Nonfinite features')
        features[p.name]=f;np.savez_compressed(cache/(p.stem+'.npz'),keypoints=f.keypoints,descriptors=f.descriptors,scores=f.scores)
        times.append(dict(image=p.name,features=len(f.keypoints),seconds=time.time()-t));print(f'features {i+1}/{len(paths)}: {len(f.keypoints)}',flush=True)
    extraction=time.time()-start;front.write_keypoints(db,ids,features)
    pairs=front.generate_pairs(sorted(features),args.pairing,args.window,True);matched=[];start=time.time();pairrows=[]
    for i,(a,b) in enumerate(pairs):
        m=front.match_lightglue_2out(sessions[1],features[a],features[b],options)
        if m.ndim!=2 or m.shape[1]!=2:raise ValueError('Invalid matches')
        if len(m) and (m[:,0].max()>=len(features[a].keypoints) or m[:,1].max()>=len(features[b].keypoints)):raise ValueError('Match index out of range')
        pairrows.append(dict(a=a,b=b,raw_matches=len(m)))
        if len(m)>=20:matched.append((a,b,m))
        if i%25==0:print(f'matching {i+1}/{len(pairs)}',flush=True)
    if not matched:raise ValueError('No image pairs with >=20 matches')
    matching=time.time()-start;front.write_matches(db,ids,matched);front.write_pairs_file(shared/'pairs.txt',matched)
    opts=pycolmap.TwoViewGeometryOptions();opts.min_num_inliers=15
    start=time.time();pycolmap.verify_matches(str(db),str(shared/'pairs.txt'),opts);verification=time.time()-start
    tracks=build_tracks_pipeline(db,shared/'observations.csv',args.observation_sigma,args.stationary_radius_px)
    save(shared/'tracks.json',tracks);save(shared/'feature_rows.json',times);save(shared/'pair_rows.json',pairrows)
    save(shared/'frontend.json',dict(features=sum(len(f.keypoints) for f in features.values()),candidate_pairs=len(pairs),
        extraction_seconds=extraction,matching_seconds=matching,verification_seconds=verification,
        coordinate_convention='COLMAP pixel centers (+0.5 relative to OpenCV)',geometry_verification='pycolmap only, no COLMAP mapper',database_sha256=sha(db)))

if __name__=='__main__':
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--provider',choices=['cpu','cuda'],required=True);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-keypoints',type=int,default=4096);p.add_argument('--pairing',choices=['sequential','exhaustive'],default='sequential')
    p.add_argument('--window',type=int,default=10);p.add_argument('--observation-sigma',type=float,default=4)
    p.add_argument('--stationary-radius-px',type=float,default=4)
    a=p.parse_args();run(a.output,a)
