"""Experimental N16-trained Keypt2Subpx transfer to unchanged ALIKED-N32 tracks.

Pair-dependent offsets are fused into one observation location. This transfer
is not the paper's detector configuration and is disabled by default.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from model_io import OBS,digest,save
from image_track_refinement import _image_paths,_candidate_edges,POLICY

def consensus_offsets(edges, offsets0, offsets1, n):
    """Require two distinct neighbor observations and agreeing bounded offsets."""
    votes=[[] for _ in range(n)]
    for (a,b),da,db in zip(edges,offsets0,offsets1):
        if np.isfinite(da).all() and np.linalg.norm(da)<=2: votes[a].append((b,da))
        if np.isfinite(db).all() and np.linalg.norm(db)<=2: votes[b].append((a,db))
    output=np.zeros((n,2),np.float32); support=np.zeros(n,np.uint16); spread=np.full(n,np.nan,np.float32)
    accepted=np.zeros(n,bool)
    for row, vv in enumerate(votes):
        if len(vv)<2:continue
        assert len({neighbor for neighbor,_ in vv})==len(vv)
        v=np.array([x for _,x in vv]); center=np.median(v,axis=0)
        distances=np.linalg.norm(v-center,axis=1); spread[row]=float(distances.max())
        # Fixed consistency check, not derived from geometry or test errors.
        if distances.max()>.5:continue
        center=v.mean(axis=0)
        if not np.isfinite(center).all() or np.linalg.norm(center)>2:continue
        output[row]=center; support[row]=len(vv); accepted[row]=True
    return output,support,spread,accepted

def refine_observations(sfm,images,features,output,patch_cache=None):
    from feature_patch_source import create_session,prepare_image
    from learned_refinement_predictor import Predictor
    sfm,images,features,output=map(Path,(sfm,images,features,output))
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True);started=time.monotonic()
    patch_cache=Path(patch_cache) if patch_cache else output/'patch_cache'
    patch_cache.mkdir(parents=True,exist_ok=True)
    original_hashes={n:digest(sfm/n) for n in ('observations.bin','camera_ids.bin')}
    obs=np.fromfile(sfm/'observations.bin',OBS);ids=np.fromfile(sfm/'camera_ids.bin','<u4')
    paths=_image_paths(sfm,images,ids);row_features=np.empty(len(obs),np.int64)
    session=create_session();patches={};manifests={}
    for camera,path in enumerate(paths):
        feature_path=features/(path.stem+'.npz'); destination=patch_cache/(path.stem+'.npz')
        # The helper audits re-extraction and caches only identical feature IDs.
        print(f'learned patch provenance {camera+1}/{len(paths)}',flush=True)
        data=prepare_image(path,feature_path,session=session,output=destination)
        lookup={tuple(map(float,xy)):i for i,xy in enumerate(data['keypoints_native'])}
        if len(lookup)!=len(data['keypoints_native']):raise ValueError('Duplicate cached keypoint')
        rows=np.flatnonzero(obs['camera']==camera)
        for row in rows:
            key=(float(obs[row]['u']),float(obs[row]['v']))
            if key not in lookup:raise ValueError(f'Model observation not an unchanged original feature: camera={camera}, row={row}')
            row_features[row]=lookup[key]
        # Keep only arrays needed by training observations; all IDs were audited.
        patches[camera]=data
        manifests[str(path)]=dict(image_sha256=digest(path),features_sha256=digest(feature_path),
                                 patch_file=str(destination),patch_sha256=digest(destination))
    del session
    edges=_candidate_edges(obs,ids,POLICY)
    offsets0=np.zeros((len(edges),2),np.float32);offsets1=offsets0.copy()
    pairs={}
    for k,(a,b) in enumerate(edges):pairs.setdefault((int(obs[a]['camera']),int(obs[b]['camera'])),[]).append(k)
    predictor=Predictor(device='cuda')
    for index,((ca,cb),indices) in enumerate(pairs.items()):
        indices=np.asarray(indices);a,b=edges[indices].T;fa,fb=row_features[a],row_features[b]
        aa,bb=patches[ca],patches[cb]
        result=predictor.predict_patches(
            np.c_[obs[a]['u'],obs[a]['v']],np.c_[obs[b]['u'],obs[b]['v']],
            aa['descriptors'][fa],bb['descriptors'][fb],aa['raw_patches'][fa],bb['raw_patches'][fb],
            aa['score_patches'][fa],bb['score_patches'][fb],
            offset_scale0_native=aa['offset_scale_native'][fa],offset_scale1_native=bb['offset_scale_native'][fb])
        offsets0[indices]=result['offset0_native'];offsets1[indices]=result['offset1_native']
        if index%25==0:print(f'learned pair predictions {index+1}/{len(pairs)}',flush=True)
    displacement,support,spread,accepted=consensus_offsets(edges,offsets0,offsets1,len(obs))
    refined=obs.copy();refined['u']+=displacement[:,0];refined['v']+=displacement[:,1]
    actual=np.c_[refined['u']-obs['u'],refined['v']-obs['v']];norm=np.linalg.norm(actual,axis=1)
    for name in ('camera','track','aux','depth','size'):
        assert obs[name].tobytes()==refined[name].tobytes()
    assert np.isfinite(actual).all() and norm.max(initial=0)<=2.001
    modified=norm>1e-6
    refined.tofile(output/'observations_refined.bin')
    np.ones(len(obs),dtype='<f4').tofile(output/'observation_sigma.bin')
    accepted.astype('<f4').tofile(output/'observation_confidence.bin')
    np.savez_compressed(output/'diagnostics.npz',edges=edges,offset0_native=offsets0,offset1_native=offsets1,
                        displacement_px=actual,support=support,spread_px=spread,modified=modified,
                        feature_ids=row_features,camera=obs['camera'],track=obs['track'])
    after={n:digest(sfm/n) for n in original_hashes};assert after==original_hashes
    result=dict(complete=True,method='keypt2subpx_n16_to_n32_with_multiview_consensus',
        uses_projection_or_3d=False,changes_track_membership=False,experimental_domain_transfer=True,
        input_sha256=original_hashes,images_and_feature_cache=manifests,
        input_observations=len(obs),output_observations=len(refined),candidate_edges=len(edges),
        modified_observations=int(modified.sum()),modified_fraction=float(modified.mean()),
        displacement_mean_modified_px=float(norm[modified].mean()) if modified.any() else 0,
        displacement_max_px=float(norm.max(initial=0)),
        policy=dict(maximum_displacement_px=2,minimum_distinct_neighbors=2,maximum_vote_distance_from_median_px=.5,
                    neighbor_camera_id_gap=3,neighbors_each_direction=2,ba_sigma='all ones'),
        predictor_metadata=predictor.metadata,module_sha256=digest(Path(__file__)),
        limitations=['Official refiner was trained with ALIKED-N16; current descriptors and score maps are N32.',
                     'No retraining or calibration of confidence; this is an unvalidated transfer experiment.',
                     'Consensus fusion is an engineering integration, not established research novelty.',
                     'Original within-scene heldout targets remain unchanged and do not provide 3D ground truth.'],
        elapsed_seconds=time.monotonic()-started)
    save(output/'summary.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sfm','images','features','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--patch-cache',type=Path)
    a=p.parse_args();refine_observations(a.sfm,a.images,a.features,a.output,a.patch_cache)
