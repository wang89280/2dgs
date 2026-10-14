"""OpenCV calibration -> existing fixed-calibration sparse binary representation."""
import struct,sys
import numpy as np
import cv2
from common import ROOT,sha,save
sys.path.insert(0,str(ROOT/'python'))
from geometry import replica_parameters,project_replica

def convert(path,destination):
    fs=cv2.FileStorage(str(path),cv2.FILE_STORAGE_READ)
    if not fs.isOpened():raise ValueError(f'Cannot read calibration: {path}')
    k=fs.getNode('camera_matrix').mat();d=fs.getNode('distortion_coefficients').mat()
    w=int(fs.getNode('image_width').real());h=int(fs.getNode('image_height').real());fs.release()
    if k is None or d is None:raise ValueError('Need camera_matrix and distortion_coefficients in OpenCV YAML')
    d=d.ravel()
    if (w,h)!=(3072,2048) or k.shape!=(3,3) or len(d)!=5 or d[4]!=0:
        raise ValueError('Frozen backend supports 3072x2048, Brown [k1,k2,p1,p2,k3] with k3=0')
    if not np.isfinite(k).all() or not np.isfinite(d).all() or k[0,0]<=0 or k[1,1]<=0 or k[0,1]!=0:
        raise ValueError('Need finite positive focal lengths and zero skew')
    if not np.array_equal(k[2],[0,0,1]) or k[1,0]!=0:raise ValueError('Invalid K')
    raw=bytearray(256)
    def put(o,fmt,*v):struct.pack_into('<'+fmt,raw,o,*v)
    put(0,'I',1);put(8,'QQ',w,h)
    put(24,'5d',k[1,1],k[0,2]+.5-w/2,k[1,2]+.5-h/2,k[0,0]-k[1,1],0);raw[72]=1
    terms=np.zeros(13);terms[[0,1,2,4,5,12]]=[d[0],d[1],d[4],d[3],d[2],1e9]
    put(80,'13d',*terms);mx=np.finfo(float).max;put(184,'4d',mx,mx,-mx,-mx)
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
    params=replica_parameters(destination);xy=np.random.default_rng(20261012).uniform([-.36,-.25],[.36,.25],(1000,2))
    uv,_=cv2.projectPoints(np.c_[xy,np.ones(len(xy))],np.zeros(3),np.zeros(3),k,d)
    error=float(np.max(np.abs(project_replica(xy,params)-uv[:,0,:]-.5)))
    if error>1e-9:raise RuntimeError('Calibration conversion check failed')
    save(destination.parent/'calibration_conversion.json',dict(source=str(path.resolve()),source_sha256=sha(path),
        K_opencv=k.tolist(),D=d.tolist(),image_size=[w,h],native_pixel_offset=.5,projection_max_error_px=error))
    return np.array([k[0,0],k[1,1],k[0,2]+.5,k[1,2]+.5,*d[:4]])
