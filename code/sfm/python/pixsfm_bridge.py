"""Lossless raw OPENCV model bridge: no undistortion, image or keypoint changes."""
from pathlib import Path
import json,shutil,xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from model_io import OBS,digest,save
from evaluate import load_replica,model_metrics
from geometry import replica_parameters

def export_raw(source,out):
    source,out=Path(source),Path(out)
    if (out/'bridge.json').exists():return
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);m=load_replica(source);p=m['calibration'];k=p['k']
    assert p['b2']==0 and all(k[i]==0 for i in (2,3,6,7,8,9,10,11))
    params=[p['f']+p['b1'],p['f'],p['width']/2+p['cx'],p['height']/2+p['cy'],k[0],k[1],k[5],k[4]]
    (out/'cameras.txt').write_text(f'1 OPENCV {p["width"]} {p["height"]} '+' '.join(format(x,'.17g') for x in params)+'\n',encoding='utf8')
    sfm=source/'sfm';obs=np.fromfile(sfm/'observations.bin',OBS);ids=np.fromfile(sfm/'camera_ids.bin','<u4');tracks={t:[] for t in m['points']};lines=[]
    for ci,i in enumerate(ids):
        c=m['cameras'][int(i)];q=Rotation.from_matrix(c['r']).as_quat()[[3,0,1,2]];t=-c['r']@c['c'];o=obs[obs['camera']==ci]
        lines.append(str(i+1)+' '+' '.join(format(x,'.17g') for x in [*q,*t])+f' 1 {i+1:04d}.png\n')
        lines.append(' '.join(f'{v["u"]:.17g} {v["v"]:.17g} {v["track"]+1}' for v in o)+'\n')
        for idx,v in enumerate(o):tracks[int(v['track'])].append((int(i)+1,idx))
    (out/'images.txt').write_text(''.join(lines),encoding='utf8')
    lines=[]
    for tid,(x,_) in m['points'].items():lines.append(str(tid+1)+' '+' '.join(format(v,'.17g') for v in x)+' 160 160 160 0 '+' '.join(f'{i} {j}' for i,j in tracks[tid])+'\n')
    (out/'points3D.txt').write_text(''.join(lines),encoding='utf8')
    # Independent pycolmap projection check detects tangent order and pixel offset mistakes.
    import pycolmap
    rec=pycolmap.Reconstruction(str(out));assert rec.num_reg_images()==len(ids) and rec.num_points3D()==len(m['points'])
    worst=0.;checked=0
    for tid,(x,observations) in list(m['points'].items())[::31]:
        for i,uv in observations:
            im=rec.images[i+1];v=im.cam_from_world()*x;u=rec.cameras[1].img_from_cam(v)
            worst=max(worst,float(np.max(np.abs(u-m['project'](i,x)[0]))));checked+=1
    assert worst<1e-7
    save(out/'bridge.json',dict(source=str(source),projection_max_difference_px=worst,projections_checked=checked,raw_measurements_preserved=True,camera_parameters=params,point_id_offset=1,image_id_offset=1))

def import_geometry(source,colmap,output):
    import pycolmap
    source,colmap,output=map(Path,(source,colmap,output));sfm=source/'sfm';target=output/'sfm'
    if (output/'bridge_audit.json').exists():return
    target.mkdir(parents=True,exist_ok=True);rec=pycolmap.Reconstruction(str(colmap));old=load_replica(source);ids=np.fromfile(sfm/'camera_ids.bin','<u4')
    assert set(rec.images)=={int(i)+1 for i in ids} and set(rec.points3D)=={i+1 for i in old['points']}
    p=old['calibration'];k=p['k'];expected=np.array([p['f']+p['b1'],p['f'],p['width']/2+p['cx'],p['height']/2+p['cy'],k[0],k[1],k[5],k[4]])
    assert np.array_equal(rec.cameras[1].params,expected),'Intrinsics changed'
    for n in ('calibration.bin','camera_ids.bin','active_points.bin','observations.bin','cameras.xml','calibration.json','region.xml','summary.json'):
        if (sfm/n).exists():shutil.copy2(sfm/n,target/n)
    poses=[];obs=np.fromfile(sfm/'observations.bin',OBS)
    for ci,i in enumerate(ids):
        im=rec.images[int(i)+1];tr=im.cam_from_world();R=tr.rotation.matrix();c=-R.T@tr.translation;poses.append(np.r_[c,R.ravel()]);original=obs[obs['camera']==ci]
        assert len(im.points2D)==len(original)
        for a,b in zip(original,im.points2D):assert b.point3D_id==int(a['track'])+1 and np.array_equal(b.xy,[float(a['u']),float(a['v'])])
    poses=np.array(poses);poses.astype('<f8').tofile(target/'poses.bin');xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3)
    for tid in old['points']:xyz[tid]=rec.points3D[tid+1].xyz
    xyz.tofile(target/'points.bin')
    tree=ET.parse(target/'cameras.xml')
    for cam in tree.getroot().findall('chunk/cameras/camera'):
        i=int(cam.get('id'));ci=list(map(int,ids)).index(i);R=poses[ci,3:].reshape(3,3);T=np.eye(4);T[:3,:3]=R.T;T[:3,3]=poses[ci,:3];cam.find('transform').text=' '.join(format(x,'.17g') for x in T.ravel())
    tree.write(target/'cameras.xml',encoding='utf8',xml_declaration=True)
    # Recompute metric before strict native summary cross-check.
    m=load_replica(output);m.pop('source_summary');metrics=model_metrics(m,output)
    # Preserve and score failed featuremetric candidates too; never silently
    # remove their invalid geometry or replace it with the baseline.
    valid_geometry=metrics['negative_depth_observations']==0
    meta=json.loads((target/'summary.json').read_text(encoding='utf8'));meta.update(reprojection_rmse_px=metrics['reprojection_px']['rmse'],reprojection_max_px=metrics['reprojection_px']['maximum'],featuremetric_backend=True,fixed_intrinsics=True)
    save(target/'summary.json',meta);model_metrics(load_replica(output),output)
    save(output/'bridge_audit.json',dict(complete=True,valid_geometry=valid_geometry,fixed_intrinsics=True,original_uv_and_track_ids_preserved=True,metrics=metrics))
