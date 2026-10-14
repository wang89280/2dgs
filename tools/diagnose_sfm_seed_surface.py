"""Training-mask-only diagnostics of the unchanged SfM initialization."""
from pathlib import Path
import json,sys
import cv2,numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
DATA=Path('/home/wk/01_code/03_gs/2dgs-foreground-fix-20261014/datasets/OFF')
def cameras():
    c=(DATA/'sparse/0/cameras.txt').read_text().split();w,h=map(int,c[2:4]);fx,fy,cx,cy=map(float,c[4:8]);views={}
    for line in (DATA/'sparse/0/images.txt').read_text().splitlines():
        a=line.split()
        if len(a)==10 and a[-1].endswith('.png'):
            q=np.array(a[1:5],float);r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();t=np.array(a[5:8],float)
            views[int(Path(a[-1]).stem)-1]=(r,t)
    return w,h,np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),views
def quant(a):return {str(q):float(np.quantile(a,q)) for q in [0,.1,.5,.9,.95,.99,1]}
def main():
    OUT.mkdir(parents=True,exist_ok=True);w,h,k,views=cameras()
    v=PlyData.read(DATA/'sparse/0/points3D.ply')['vertex'];x=np.c_[v['x'],v['y'],v['z']].astype(float)
    distance,index=cKDTree(x).query(x,k=25);neighbors=x[index[:,1:]];center=neighbors.mean(1);d=neighbors-center[:,None]
    eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',d,d)/24)
    normal=vec[:,:,0];offset=np.abs(np.sum((x-center)*normal,axis=1));rough=offset/np.maximum(distance[:,12],1e-10)
    outside=np.zeros(len(x),int);close=np.zeros(len(x),int);coverage=[]
    for ci,(r,t) in views.items():
        if ci%8==0:continue
        mask=cv2.imread(str(DATA/'masks_train'/f'{ci+1:04d}.png'),0);assert mask is not None
        signed=cv2.distanceTransform((mask>0).astype('uint8'),cv2.DIST_L2,5)-cv2.distanceTransform((mask==0).astype('uint8'),cv2.DIST_L2,5)
        z=x@r.T+t;uv=z[:,:2]/z[:,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
        d=np.concatenate([cv2.remap(signed,part[:,0].astype('f4')[:,None],part[:,1].astype('f4')[:,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1000).ravel() for part in np.array_split(uv,4)])
        outside+=(d< -3)|(z[:,2]<=0);close+=d<3
        # Coverage is geometric projection coverage, not visibility/accuracy truth.
        seed=np.ones((h,w),np.uint8);u=np.rint(uv).astype(int);good=(u[:,0]>=0)&(u[:,0]<w)&(u[:,1]>=0)&(u[:,1]<h)&(z[:,2]>0)
        seed[u[good,1],u[good,0]]=0;dist=cv2.distanceTransform(seed,cv2.DIST_L2,5);f=mask>0
        coverage.append(dict(name=f'{ci+1:04d}',fraction_fg_within4px=float((dist[f]<=4).mean()),fraction_fg_within8px=float((dist[f]<=8).mean())))
    summary=dict(points=len(x),outside_3px_at_least1=int((outside>=1).sum()),outside_3px_at_least3=int((outside>=3).sum()),
        outside_3px_at_least7=int((outside>=7).sum()),outside_counts=quant(outside),neighbor12_distance=quant(distance[:,12]),
        local_plane_offset_relative_to_neighbor12=quant(rough),local_eigen_ratio=quant(eig[:,0]/np.maximum(eig[:,1],1e-15)),
        mean_fg_coverage4px=float(np.mean([r['fraction_fg_within4px'] for r in coverage])),
        mean_fg_coverage8px=float(np.mean([r['fraction_fg_within8px'] for r in coverage])),coverage=coverage,
        only_training_masks=True,test_RGB_read=False,note='Masks constrain the visual hull, not visible depth. Coverage includes occluded projected points and overestimates real visible coverage.')
    np.savez_compressed(OUT/'seed_diagnostics.npz',xyz=x,outside=outside,near_boundary_count=close,neighbor_distance=distance,neighbor_index=index,normal=normal,local_center=center,eigenvalues=eig,roughness=rough)
    (OUT/'seed_diagnostics.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='coverage'},indent=2))
if __name__=='__main__':main()
