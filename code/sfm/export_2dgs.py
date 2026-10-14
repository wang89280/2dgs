"""Final sparse model -> undistorted centered PINHOLE COLMAP binary/text + RGB PLY.

No COLMAP reconstruction/undistorter executable or 2DGS source is required.
Pixel convention: native/ COLMAP observation = OpenCV pixel center + 0.5.
"""
import argparse,json,struct,sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation
from common import ROOT,save,sha
sys.path.insert(0,str(ROOT/'python'))
from geometry import replica_parameters,project_replica,write_ply
OBS=np.dtype([('camera','<u4'),('track','<u4'),('aux','<u8'),('u','<f4'),('v','<f4'),('depth','<f4'),('size','<f4')])

def export(sfm,images,out,width=1600,evaluation=False,allow_partial=False):
    if out.exists():raise FileExistsError(out)
    sources=sorted(p for p in images.iterdir() if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    needed=['calibration.bin','camera_ids.bin','poses.bin','points.bin','active_points.bin','observations.bin']
    frozen={str(sfm/n):sha(sfm/n) for n in needed}
    p=replica_parameters(sfm/'calibration.bin');ids=np.fromfile(sfm/'camera_ids.bin','<u4')
    poses=np.fromfile(sfm/'poses.bin','<f8').reshape(-1,12)
    xyz=np.fromfile(sfm/'points.bin','<f8').reshape(-1,3);active=np.fromfile(sfm/'active_points.bin','<u4')
    obs=np.fromfile(sfm/'observations.bin',OBS)
    if len(ids)<2 or len(ids)!=len(poses) or len(set(ids))!=len(ids) or int(ids.max())>=len(sources):raise ValueError('Invalid camera IDs or input image count')
    if not allow_partial and len(ids)!=len(sources):raise ValueError(f'Only {len(ids)}/{len(sources)} cameras registered; inspect SfM logs or explicitly use --allow-partial')
    if not len(active) or int(active.max())>=len(xyz):raise ValueError('Invalid active points')
    if not np.isfinite(poses).all() or not np.isfinite(xyz[active]).all():raise ValueError('Nonfinite geometry')
    if int(obs['camera'].max())>=len(ids):raise ValueError('Observation camera index invalid')
    original_w,original_h=p['width'],p['height'];w=width or original_w;h=round(original_h*w/original_w)
    if w<2 or h<2:raise ValueError('Export resolution too small')
    # Source K is 0-based OpenCV, NOT the +0.5-offset SfM camera.
    K=np.array([[p['f']+p['b1'],0,original_w/2+p['cx']-.5],[0,p['f'],original_h/2+p['cy']-.5],[0,0,1.]])
    k=p['k'];D=np.array([k[0],k[1],k[5],k[4],k[2]])
    if p['b2']!=0 or np.any(k[[3,6,7,8,9,10,11]]!=0):raise ValueError('Exporter currently supports Brown radial/tangential model without skew/extra prism terms')
    target=np.array([[K[0,0]*w/original_w,0,(w-1)/2],[0,K[1,1]*h/original_h,(h-1)/2],[0,0,1.]])
    mx,my=cv2.initUndistortRectifyMap(K,D,None,target,(w,h),cv2.CV_32FC1)
    # Check this independently against the exact native forward projection.
    grid=np.array([[x,y] for x in np.linspace(0,w-1,9) for y in np.linspace(0,h-1,7)])
    rays=(grid-target[:2,2])/np.diag(target)[:2]
    cv,_=cv2.projectPoints(np.c_[rays,np.ones(len(rays))],np.zeros(3),np.zeros(3),K,D)
    forward_error=float(np.max(np.abs(project_replica(rays,p)-.5-cv[:,0,:])))
    if forward_error>1e-8:raise RuntimeError('Source camera projection mismatch')
    order=sorted(range(len(ids)),key=lambda i:sources[int(ids[i])].name)
    training=set(order if not evaluation else [v for n,v in enumerate(order) if n%8!=0])
    tests=[v for n,v in enumerate(order) if evaluation and n%8==0]
    if len(training)<2:raise ValueError('Need at least two training cameras')
    centers=poses[:,:3];rot=poses[:,3:].reshape(-1,3,3)
    if np.max(np.abs(rot@rot.transpose(0,2,1)-np.eye(3)))>1e-7 or np.min(np.linalg.det(rot))<.999999:raise ValueError('Invalid camera rotation')
    center=np.mean(centers[sorted(training)],axis=0)
    radius=max(np.linalg.norm(centers[i]-center) for i in training)
    if not np.isfinite(radius) or radius<1e-9:raise ValueError('Degenerate camera trajectory')
    orientation=rot[order[0]]
    normalized_xyz=((xyz-center)@orientation.T)/radius
    cn=((centers-center)@orientation.T)/radius;rn=rot@orientation.T
    tn=-np.einsum('nij,nj->ni',rn,cn)
    quats=Rotation.from_matrix(rn).as_quat()[:,[3,0,1,2]]
    uv=np.column_stack((obs['u'],obs['v'])).astype(float)-.5
    ucv=cv2.undistortPointsIter(uv[:,None,:],K,D,np.eye(3),target,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,50,1e-12))[:,0,:]
    undistorted=ucv+.5  # consistent COLMAP camera principal point at (w/2,h/2).
    valid=np.isfinite(ucv).all(axis=1)&(ucv[:,0]>=0)&(ucv[:,0]<=w-1)&(ucv[:,1]>=0)&(ucv[:,1]<=h-1)&np.isin(obs['track'],active)
    colors=np.zeros((len(xyz),3));counts=np.zeros(len(xyz),int)
    sums=np.zeros(len(xyz),int)
    out_images=out/'images';sparse=out/'sparse/0';out_images.mkdir(parents=True);sparse.mkdir(parents=True)
    names={};image_records=[]
    input_manifest=images.parent/'image_manifest.json'
    original_mapping=json.loads(input_manifest.read_text(encoding='utf8')) if input_manifest.exists() else None
    for ordinal,ci in enumerate(order):
        source=sources[int(ids[ci])]
        with Image.open(source) as im:
            if im.size!=(original_w,original_h):raise ValueError(f'Image/calibration size mismatch: {source}')
            if im.mode!='RGB':raise ValueError('Exporter expects prepared RGB inputs (the main entry prepares these automatically)')
            rgb=np.asarray(im)
        name=f'{ordinal+1:04d}.png';names[ci]=name
        converted=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        Image.fromarray(converted).save(out_images/name)
        rows=np.flatnonzero(valid&(obs['camera']==ci))
        if ci in training and len(rows):
            xy=uv[rows];samples=cv2.remap(rgb,xy[:,0].astype('f4')[:,None],xy[:,1].astype('f4')[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)[:,0,:]
            np.add.at(colors,obs['track'][rows],samples);np.add.at(counts,obs['track'][rows],1)
        image_records.append(dict(name=name,source=str(source.resolve()),source_sha256=sha(source),native_camera_id=int(ids[ci]),split='train' if ci in training else 'test'))
        if original_mapping is not None:
            original=original_mapping[int(ids[ci])]
            image_records[-1].update(original_source=original['source'],original_source_sha256=original['source_sha256'])
    retained=active[counts[active]>=2]
    if not len(retained):raise ValueError('No points with >=2 valid training observations after undistortion')
    point_id={int(t):j+1 for j,t in enumerate(retained)}
    final_xyz=normalized_xyz[retained];rgb=np.rint(colors[retained]/counts[retained,None]).clip(0,255).astype('u1')
    image_observations={};point_tracks={int(t):[] for t in retained};errors={int(t):[] for t in retained}
    camera_export={ci:n+1 for n,ci in enumerate(order)}
    max_roundtrip=0.;negative_depth=0
    for ci in order:
        rows=np.flatnonzero(valid&(obs['camera']==ci)&np.isin(obs['track'],retained))
        records=[]
        for j,oi in enumerate(rows):
            tid=int(obs['track'][oi]);x=normalized_xyz[tid];v=rn[ci]@x+tn[ci]
            if v[2]<=0:negative_depth+=1
            predicted=np.array([target[0,0]*v[0]/v[2]+w/2,target[1,1]*v[1]/v[2]+h/2])
            error=float(np.linalg.norm(predicted-undistorted[oi]));errors[tid].append(error)
            records.append((undistorted[oi,0],undistorted[oi,1],point_id[tid]))
            point_tracks[tid].append((camera_export[ci],j))
        image_observations[ci]=records
        a=(xyz[retained[:100]]-centers[ci])@rot[ci].T
        b=final_xyz[:100]@rn[ci].T+tn[ci]
        max_roundtrip=max(max_roundtrip,float(np.max(np.abs(a[:,:2]/a[:,2,None]-b[:,:2]/b[:,2,None]))))
    if negative_depth or max_roundtrip>1e-8:raise ValueError(f'Invalid export projection/depth: {negative_depth}, {max_roundtrip}')
    parameters=[target[0,0],target[1,1],w/2,h/2]
    with (sparse/'cameras.bin').open('wb') as f:
        f.write(struct.pack('<QiiQQ4d',1,1,1,w,h,*parameters))
    (sparse/'cameras.txt').write_text('1 PINHOLE '+f'{w} {h} '+' '.join(format(x,'.17g') for x in parameters)+'\n',encoding='utf8')
    image_lines=[]
    with (sparse/'images.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(order)))
        for ci in order:
            iid=camera_export[ci];name=names[ci];rows=image_observations[ci]
            f.write(struct.pack('<i7di',iid,*quats[ci],*tn[ci],1));f.write(name.encode()+b'\0');f.write(struct.pack('<Q',len(rows)))
            for u,v,pid in rows:f.write(struct.pack('<ddq',u,v,pid))
            image_lines.append(f'{iid} '+' '.join(format(x,'.17g') for x in [*quats[ci],*tn[ci]])+f' 1 {name}\n'+' '.join(f'{u:.17g} {v:.17g} {pid}' for u,v,pid in rows)+'\n')
    (sparse/'images.txt').write_text(''.join(image_lines),encoding='utf8')
    point_lines=[]
    with (sparse/'points3D.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(retained)))
        for j,tid in enumerate(retained):
            tid=int(tid);pid=point_id[tid];track=point_tracks[tid];error=float(np.mean(errors[tid]))
            f.write(struct.pack('<QdddBBBdQ',pid,*final_xyz[j],*rgb[j],error,len(track)))
            for iid,idx in track:f.write(struct.pack('<ii',iid,idx))
            point_lines.append(f'{pid} '+' '.join(format(x,'.17g') for x in final_xyz[j])+' '+' '.join(map(str,rgb[j]))+f' {error:.17g} '+' '.join(f'{i} {k}' for i,k in track)+'\n')
    (sparse/'points3D.txt').write_text(''.join(point_lines),encoding='utf8')
    write_ply(sparse/'points3D.ply',final_xyz,rgb)
    import pycolmap
    rec=pycolmap.Reconstruction(str(sparse))
    if rec.num_reg_images()!=len(ids) or rec.num_points3D()!=len(retained):raise RuntimeError('COLMAP model round-trip count mismatch')
    checked=0
    for pid,point in rec.points3D.items():
        for e in point.track.elements:
            if rec.images[e.image_id].points2D[e.point2D_idx].point3D_id!=pid:raise RuntimeError('Broken point/image track link')
            checked+=1
    if not all(sha(Path(k))==v for k,v in frozen.items()):raise RuntimeError('Source model changed during export')
    summary=dict(complete=True,registered_images=len(ids),input_images=len(sources),points=len(retained),original_points=len(active),
        observations=checked,width=w,height=h,camera_model='PINHOLE',camera_parameters=parameters,
        evaluation=evaluation,train_images=len(training),test_images=len(tests),
        train_names=[names[i] for i in order if i in training],test_names=[names[i] for i in tests],
        pixel_convention='COLMAP pixels = OpenCV pixels + 0.5. Remapping uses OpenCV K. Rasterizer center is (size-1)/2.',
        forward_projection_check_max_px=forward_error,gauge_projection_roundtrip_max=max_roundtrip,negative_depth_observations=negative_depth,
        gauge=dict(center=center.tolist(),radius=float(radius),rotation=orientation.tolist()),
        dropped_outside_export_image_observations=int((~valid).sum()),dropped_points=len(active)-len(retained),
        source_model_sha256=frozen,source_model_unchanged=True,
        scope='Full-data SfM geometry; optional evaluation withholds only 2DGS color/loss observations. No metric ground truth.')
    save(out/'export_manifest.json',summary);save(out/'image_mapping.json',image_records)
    flag=' --eval' if evaluation else ''
    (out/'TRAIN_2DGS.txt').write_text('In your Linux/WSL 2DGS directory, replace DATASET_PATH with this folder:\npython train.py -s DATASET_PATH -m OUTPUT_MODEL --resolution 1'+flag+'\n\nUse resolution 1: images are already undistorted/resized to the exported camera.\n',encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['complete','registered_images','points','width','height','evaluation','observations']},indent=2))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sfm',type=Path,required=True);p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=1600);p.add_argument('--eval',action='store_true');p.add_argument('--allow-partial',action='store_true')
    a=p.parse_args();export(a.sfm.resolve(),a.images.resolve(),a.output.resolve(),a.width,a.eval,a.allow_partial)
