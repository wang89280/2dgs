"""Validate portable COLMAP/2DGS output without changing it."""
import argparse,json
from pathlib import Path
import numpy as np
import pycolmap
from PIL import Image

def validate(dataset):
    sparse=dataset/'sparse/0'
    for name in ['cameras.bin','images.bin','points3D.bin','points3D.ply']:
        if not (sparse/name).is_file():raise ValueError(f'Missing {name}')
    rec=pycolmap.Reconstruction(str(sparse))
    if rec.num_reg_images()<2 or rec.num_points3D()<1:raise ValueError('Empty reconstruction')
    for c in rec.cameras.values():
        if c.model_name!='PINHOLE':raise ValueError('Expected undistorted PINHOLE')
        if not np.allclose(c.params[2:],[c.width/2,c.height/2],atol=1e-7,rtol=0):raise ValueError('Principal point must be centered')
        if not np.isfinite(c.params).all() or min(c.params[:2])<=0:raise ValueError('Invalid intrinsics')
    for im in rec.images.values():
        with Image.open(dataset/'images'/im.name) as pix:
            c=rec.cameras[im.camera_id]
            if pix.size!=(c.width,c.height) or pix.mode!='RGB':raise ValueError('Image format/camera mismatch')
    nobs=0;errors=[];negative=0
    for pid,point in rec.points3D.items():
        if not np.isfinite(point.xyz).all():raise ValueError('Nonfinite XYZ')
        cameras=set()
        for e in point.track.elements:
            im=rec.images[e.image_id];q=im.points2D[e.point2D_idx]
            if q.point3D_id!=pid:raise ValueError('Inconsistent point track')
            if e.image_id in cameras:raise ValueError('Duplicate camera in point track')
            cameras.add(e.image_id);nobs+=1
            v=im.cam_from_world()*point.xyz
            if v[2]<=0:negative+=1
            pixel=rec.cameras[im.camera_id].img_from_cam(v)
            errors.append(float(np.linalg.norm(pixel-q.xy)))
        if len(cameras)<2:raise ValueError('Point with fewer than 2 cameras')
    if negative:raise ValueError(f'{negative} negative-depth observations')
    header=(sparse/'points3D.ply').open('rb').read(4096).split(b'end_header')[0].decode('ascii')
    for n in ['x','y','z','nx','ny','nz','red','green','blue']:
        if not any(line.endswith(' '+n) for line in header.splitlines()):raise ValueError(f'PLY missing {n}')
    return dict(valid=True,images=rec.num_reg_images(),points=rec.num_points3D(),observations=nobs,
        reprojection_rmse_export_pixels=float(np.sqrt(np.mean(np.square(errors)))),negative_depth_observations=negative)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    print(json.dumps(validate(p.parse_args().dataset.resolve()),indent=2))
