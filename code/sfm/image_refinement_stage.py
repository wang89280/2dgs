"""Optional image-only localization refinement followed by fixed-intrinsic BA."""
from pathlib import Path
import argparse, json, os, shutil, subprocess, sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'python'))
from image_track_refinement import refine_observations
from model_io import OBS, digest, patch_xml, save
from evaluate import load_replica, model_metrics, stats

CORE = ('camera_ids.bin','calibration.bin','poses.bin','points.bin','active_points.bin','observations.bin')

def _ordered_rows(rows, ids):
    """Stable identity order independent of native camera/observation order."""
    if len(ids) != len(set(map(int,ids))) or (len(rows) and int(rows['camera'].max()) >= len(ids)):
        raise ValueError('Invalid camera IDs in image-refinement audit')
    global_camera=ids[rows['camera']]
    order=np.lexsort((rows['track'],global_camera))
    keys=np.c_[global_camera[order],rows['track'][order]]
    if len(np.unique(keys,axis=0)) != len(rows):
        raise ValueError('Duplicate point-image identity')
    return keys,rows[order],order

def _restore_xml_labels(source, target, oldids, newids):
    """Preserve source image identity when the native writer reorders cameras."""
    before=ET.parse(source/'cameras.xml')
    cameras=before.getroot().findall('.//cameras/camera')
    xmlids={int(c.attrib['id']) for c in cameras}
    if len(cameras)!=len(oldids):raise ValueError('Source XML camera count mismatch')
    global_ids=xmlids==set(map(int,oldids))
    local_ids=xmlids==set(range(len(oldids)))
    if global_ids and local_ids and not np.array_equal(oldids,np.arange(len(oldids))):
        # Contiguous IDs in a permuted binary array make ID sets ambiguous.
        # The source transforms resolve which convention its writer used.
        poses=np.fromfile(source/'poses.bin','<f8').reshape(-1,12)
        positions={int(camera):i for i,camera in enumerate(oldids)}
        def transforms_match(use_global):
            for camera in cameras:
                cid=int(camera.attrib['id']);index=positions[cid] if use_global else cid
                values=np.fromstring(camera.findtext('transform',''),sep=' ')
                if len(values)!=16:return False
                matrix=values.reshape(4,4)
                if not np.allclose(matrix[:3,3],poses[index,:3],atol=1e-8,rtol=1e-8):return False
                if not np.allclose(matrix[:3,:3],poses[index,3:].reshape(3,3).T,atol=1e-8,rtol=1e-8):return False
            return True
        global_ids,local_ids=transforms_match(True),transforms_match(False)
        if global_ids==local_ids:raise ValueError('Ambiguous source XML camera identity after transform audit')
    if global_ids:
        labels={int(c.attrib['id']):c.attrib['label'] for c in cameras}
        identity='original_camera_ids'
    elif local_ids:
        labels={int(oldids[int(c.attrib['id'])]):c.attrib['label'] for c in cameras}
        identity='source_local_indices_mapped_through_camera_ids'
    else:raise ValueError('Source XML IDs do not map to native camera IDs')
    if len(set(labels.values()))!=len(labels):raise ValueError('Duplicate source XML image labels')
    tree=ET.parse(target/'cameras.xml')
    output=tree.getroot().findall('.//cameras/camera')
    if len(output)!=len(newids) or {int(c.attrib['id']) for c in output}!=set(range(len(newids))):
        raise ValueError('Unexpected native output XML camera IDs')
    for camera in output:
        original_id=int(newids[int(camera.attrib['id'])])
        camera.set('id',str(original_id));camera.set('label',labels[original_id])
    tree.write(target/'cameras.xml',encoding='utf-8',xml_declaration=True)
    # This also restores affine calibration fields and the native delimiter format.
    patch_xml(target)
    return dict(source_xml_identity=identity,labels_preserved_by_original_camera_id=True,
                labels={str(k):v for k,v in labels.items()},source_xml_sha256=digest(source/'cameras.xml'))

def measurement_metrics(model, sfm):
    ids = np.fromfile(sfm/'camera_ids.bin','<u4')
    obs = np.fromfile(sfm/'observations.bin',OBS)
    errors=[]
    for i, camera in enumerate(ids):
        rows=obs[obs['camera']==i]
        if len(rows):
            xyz=np.array([model['points'][int(t)][0] for t in rows['track']])
            errors.extend(np.linalg.norm(model['project'](int(camera),xyz)-np.c_[rows['u'],rows['v']],axis=1))
    return stats(errors)

def optimize(source, output, pixels, mode, logs, multiview=False):
    """Reuse a frozen pixel proposal for matched BA controls and candidates."""
    source,output,pixels,logs=map(Path,(source,output,pixels,logs))
    if mode not in ('control','localization','weighted'): raise ValueError(mode)
    if output.exists(): raise FileExistsError(output)
    sfm=source/'sfm'; before={name:digest(sfm/name) for name in CORE}
    obs=np.fromfile(sfm/'observations.bin',OBS)
    output.mkdir(parents=True); evidence=output/'image_evidence'; evidence.mkdir()
    # Retain the original measurement targets even after pipeline scratch cleanup.
    for name in ('observations.bin','camera_ids.bin'):
        shutil.copy2(sfm/name,evidence/('original_'+name))
    if mode=='control':
        measured=evidence/'observations_used.bin'; shutil.copy2(sfm/'observations.bin',measured)
        sigma=evidence/'observation_sigma.bin'; np.ones(len(obs),dtype='<f4').tofile(sigma)
    else:
        for p in pixels.iterdir():
            if p.is_file(): shutil.copy2(p,evidence/p.name)
        measured=evidence/'observations_refined.bin'; sigma=evidence/'observation_sigma.bin'
        proposal=json.loads((evidence/'summary.json').read_text(encoding='utf8'))
        if not proposal.get('complete') or any(proposal.get('input_sha256',{}).get(n)!=before[n]
                                               for n in ('observations.bin','camera_ids.bin')):
            raise ValueError('Pixel proposal was not produced from this exact input model')
    measured_rows=np.fromfile(measured,OBS); sigma_rows=np.fromfile(sigma,'<f4')
    if measured.stat().st_size!=len(obs)*OBS.itemsize or sigma.stat().st_size!=len(obs)*4:
        raise ValueError('Measured observation or sigma extent mismatch')
    for name in ('camera','track','aux','depth','size'):
        if measured_rows[name].tobytes()!=obs[name].tobytes():
            raise ValueError('Pixel proposal changed observation metadata: '+name)
    displacement=np.c_[measured_rows['u'].astype(float)-obs['u'],measured_rows['v'].astype(float)-obs['v']]
    shift_limit=64 if multiview else 2
    if not np.isfinite(displacement).all() or np.linalg.norm(displacement,axis=1).max(initial=0)>shift_limit+.001:
        raise ValueError('Invalid image-derived coordinate displacement')
    if not np.isfinite(sigma_rows).all() or np.any((sigma_rows<.5)|(sigma_rows>2)):
        raise ValueError('Invalid image-derived sigma')
    if any(digest(evidence/('original_'+n))!=before[n] for n in ('observations.bin','camera_ids.bin')):
        raise ValueError('Original coordinate evidence was overwritten')
    exe=ROOT/('bin/refine_multiview_observations.exe' if multiview else 'bin/refine_image_observations.exe')
    cmd=[str(exe),str(sfm),str(output/'sfm'),str(measured),str(sigma),'weighted' if mode=='weighted' else 'uniform']
    logs.mkdir(parents=True,exist_ok=True)
    log=logs/(output.name+'_image_ba.log')
    save(output/'command.json',dict(command=cmd,mode=mode,source=str(source),input_sha256=before,
                                  executable_sha256=digest(exe)))
    env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with log.open('w',encoding='utf8') as f:
        subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,env=env,
                       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    after=np.fromfile(output/'sfm/observations.bin',OBS)
    ids=np.fromfile(sfm/'camera_ids.bin','<u4'); newids=np.fromfile(output/'sfm/camera_ids.bin','<u4')
    expected_keys,expected_rows,expected_order=_ordered_rows(measured_rows,ids)
    saved_keys,saved_rows,_=_ordered_rows(after,newids)
    if len(obs)!=len(after) or not np.array_equal(expected_keys,saved_keys):
        raise ValueError('Native BA changed observation identities')
    for name in ('track','aux','u','v','depth'):
        if expected_rows[name].tobytes()!=saved_rows[name].tobytes():
            raise ValueError('Native BA changed a measurement field: '+name)
    expected_sigma=sigma_rows[expected_order] if mode=='weighted' else np.ones(len(obs),dtype='<f4')
    if expected_sigma.tobytes()!=saved_rows['size'].tobytes():
        raise ValueError('Saved observation size does not match the selected base sigma')
    if set(ids)!=set(newids):raise ValueError('Native BA changed camera identities')
    if set(np.fromfile(sfm/'active_points.bin','<u4'))!=set(np.fromfile(output/'sfm/active_points.bin','<u4')):
        raise ValueError('Native BA changed active point identities')
    if (sfm/'calibration.bin').read_bytes()!=(output/'sfm/calibration.bin').read_bytes():
        raise ValueError('Fixed calibration changed during image refinement')
    if before!={name:digest(sfm/name) for name in CORE}:raise ValueError('Source model changed')
    xml_audit=_restore_xml_labels(sfm,output/'sfm',ids,newids)
    summary=json.loads((output/'sfm/summary.json').read_text(encoding='utf8'))
    summary.update(image_refinement_mode=mode,track_membership_changed=False,
                   image_refinement_applied=mode!='control',fixed_intrinsics=True)
    save(output/'sfm/summary.json',summary)
    model=load_replica(output)
    # Reproduce native RMSE independently and satisfy the normal final-model
    # artifact contract (metrics.json and sparse.ply), including optional mode.
    metrics=model_metrics(model,output)
    if metrics['negative_depth_observations']:
        raise ValueError('Image-refinement output has nonpositive-depth observations')
    raw_metrics=measurement_metrics(model,sfm)
    used_metrics=measurement_metrics(model,output/'sfm')
    result=dict(complete=True,mode=mode,source=str(source),fixed_intrinsics=True,
                cameras=len(ids),points=len(model['points']),observations=len(obs),
                all_observation_identities_retained=True,track_membership_changed=False,
                original_measurement_reprojection_px=raw_metrics,
                refined_measurement_reprojection_px=used_metrics,
                observation_fields_verified_by_identity=['camera','track','aux','u','v','depth','selected_sigma'],
                coordinate_shift_limit_px=shift_limit,actual_max_coordinate_shift_px=float(np.linalg.norm(displacement,axis=1).max(initial=0)),
                original_coordinate_evidence=dict(observations='image_evidence/original_observations.bin',
                                                  camera_ids='image_evidence/original_camera_ids.bin',
                                                  sha256={n:before[n] for n in ('observations.bin','camera_ids.bin')}),
                xml_identity_audit=xml_audit,
                final_artifacts=dict(metrics='metrics.json',point_cloud='sparse.ply'),
                accuracy_claim='Neither training-target RMSE measures independent geometric accuracy.',
                source_sha256=before,source_unchanged=True)
    save(output/'image_refinement.json',result)
    return output

def execute(sourceRoot,imagesPath,outputRoot,mode,logRoot):
    if mode not in ('localization','weighted','learned','multiview'): raise ValueError(mode)
    output=Path(outputRoot)
    # Pixel diagnostics live outside scratch and remain reviewable.
    pixels=output.parent/(output.name+'_image_proposals')
    if mode=='multiview':
        from multiview_refinement import refine_observations as refine_multiview
        refine_multiview(Path(sourceRoot)/'sfm',Path(imagesPath),pixels)
    elif mode=='learned':
        from learned_track_refinement import refine_observations as refine_learned
        refine_learned(Path(sourceRoot)/'sfm',Path(imagesPath),Path(imagesPath).parent/'features',pixels)
    else:
        refine_observations(Path(sourceRoot)/'sfm',Path(imagesPath),pixels)
    result=optimize(sourceRoot,output,pixels,'localization' if mode in ('learned','multiview') else mode,logRoot,multiview=mode=='multiview')
    if mode in ('learned','multiview'):
        for path in (output/'image_refinement.json',output/'sfm/summary.json'):
            metadata=json.loads(path.read_text(encoding='utf8'))
            metadata['image_proposal_method']='densesfm_multiview_released_weights' if mode=='multiview' else 'keypt2subpx_n16_to_n32_with_multiview_consensus'
            metadata['mode' if path.name=='image_refinement.json' else 'image_refinement_mode']=mode
            metadata['image_ba_weighting']='uniform'
            save(path,metadata)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','images','output','logs'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--mode',choices=('localization','weighted','learned','multiview'),default='localization')
    a=p.parse_args();execute(a.source,a.images,a.output,a.mode,a.logs)
