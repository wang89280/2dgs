"""Numerical checks, independent hull audit, and a point-count-matched ablation."""
import json,shutil,sys
from pathlib import Path
import cv2,numpy as np
from plyfile import PlyData
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
sys.path.insert(0,str(ROOT/'code/sfm'))
from surface_triangulation import refine,information,POLICY
from diagnose_sfm_seed_surface import DATA,cameras
from prepare_surface_sfm import WORK,sha,save

def main():
    # Finite-difference J^T J check, and a synthetic repair with a feasible unchanged point.
    r=np.repeat(np.eye(3)[None],3,axis=0);t=np.array([[-.02,0,0],[0,0,0],[.02,0,0]])
    k=np.array([[100.,0,31.5],[0,100.,31.5],[0,0,1]])
    x=np.array([[.047,0,1],[0,.02,1.]])
    cam=np.repeat(np.arange(3),2);point=np.tile(np.arange(2),3)
    def project(x):
        z=x[point]+t[cam];return z[:,:2]/z[:,2,None]*100+31.5
    uv=project(x);j=[]
    for d in range(3):
        dx=np.zeros_like(x);dx[:,d]=1e-6;j.append((project(x+dx)-project(x-dx))/2e-6)
    j=np.stack(j,-1);expected=np.zeros((2,3,3));np.add.at(expected,point,np.einsum('nji,njk->nik',j,j))
    actual=information(x,cam,point,r,t,k)[0];difference=float(np.max(np.abs(expected-actual)))
    assert np.allclose(expected,actual,rtol=1e-7,atol=1e-5),difference
    xx=np.indices((64,64))[1].astype('f4');fields=np.repeat(np.stack([35-xx,-np.ones_like(xx),np.zeros_like(xx)])[None],3,axis=0)
    result=refine(x,r,t,k,fields,[0,1,2],cam,point,uv)
    assert result['filter_keep'].tolist()==[False,True] and not result['keep'][0]
    # That weak-baseline case exceeds the world-motion trust region; it must be rejected.
    # A stronger-baseline case has a feasible small correction within the same frozen limits.
    t=np.array([[-.1,0,0],[0,0,0],[.1,0,0]]);x[0,0]=.062;uv=project(x)
    fields[:,0]=44.5-xx
    result=refine(x,r,t,k,fields,[0,1,2],cam,point,uv)
    assert result['filter_keep'].tolist()==[False,True] and result['keep'].all()
    assert np.array_equal(result['xyz'][1],x[1]) and not np.array_equal(result['xyz'][0],x[0])
    assert result['final_distance'][0]>=-3.05
    # Audit every exported point against all 70 masks with independent float64 bilinear sampling.
    prepared=json.loads((OUT/'prepared.json').read_text());data=np.load(OUT/'surface_result.npz');w,h,k,views=cameras()
    cloud=PlyData.read(DATA/'sparse/0/points3D.ply');v=cloud['vertex'].data
    old=np.c_[v['x'],v['y'],v['z']];new=data['xyz'].astype('f4');keep=data['keep'];min_dist=np.full(len(old),np.inf)
    for ci,(r,t) in views.items():
        if ci%8==0:continue
        mask=cv2.imread(str(DATA/'masks_train'/f'{ci+1:04d}.png'),0)
        signed=cv2.distanceTransform((mask>0).astype('uint8'),cv2.DIST_L2,5)-cv2.distanceTransform((mask==0).astype('uint8'),cv2.DIST_L2,5)
        z=new.astype(float)@r.T+t;uv=z[:,:2]/z[:,2,None]*np.diag(k)[:2]+k[:2,2]
        p=np.floor(uv).astype(int);f=uv-p;valid=(p[:,0]>=0)&(p[:,0]<w-1)&(p[:,1]>=0)&(p[:,1]<h-1)&(z[:,2]>0)
        d=np.full(len(old),-1000.);xx,yy=p[valid].T;fx,fy=f[valid].T
        d[valid]=(1-fx)*(1-fy)*signed[yy,xx]+fx*(1-fy)*signed[yy,xx+1]+(1-fx)*fy*signed[yy+1,xx]+fx*fy*signed[yy+1,xx+1]
        min_dist=np.minimum(min_dist,d)
    worst=float(min_dist[keep].min());assert worst>=-3.052,worst
    assert np.array_equal(new[data['filter_keep']],old[data['filter_keep']])
    protocol=json.loads((OUT/'surface_protocol.json').read_text())
    for p,value in protocol['protected_sha256'].items():assert sha(p)==value,p
    for p,value in protocol['code_sha256'].items():assert sha(ROOT/p)==value,p
    # This control retains EXACTLY the HULL point IDs, order and colors, but original positions.
    gate=WORK/'datasets/GATE';sparse=gate/'sparse/0'
    if not gate.exists():
        sparse.mkdir(parents=True)
        for n in ['cameras.txt','images.txt']:shutil.copy2(DATA/'sparse/0'/n,sparse/n)
        for n in ['images','masks_train']:(gate/n).symlink_to((DATA/n).resolve(),target_is_directory=True)
        cloud['vertex'].data=v[keep].copy();cloud.write(sparse/'points3D.ply')
    gv=PlyData.read(sparse/'points3D.ply')['vertex'].data
    hv=PlyData.read(WORK/'datasets/HULL/sparse/0/points3D.ply')['vertex'].data
    assert len(gv)==len(hv)==int(keep.sum())
    assert all(np.array_equal(gv[n],hv[n]) for n in ['red','green','blue'])
    assert all(np.array_equal(gv[n],v[n][keep]) for n in ['x','y','z'])
    audit=dict(passed=True,point_information_matrix_finite_difference_max_error=difference,
        synthetic_correction_passed=True,trust_region_rejection_passed=True,already_feasible_points_bitwise_unchanged=True,
        all_exported_points_training_hull_verified=True,worst_signed_distance_px=worst,
        point_count_matched_control=True,gate_and_hull_points=len(gv),same_point_ids_order_colors=True,
        corrected_exported_points=int(np.any(np.c_[gv['x'],gv['y'],gv['z']]!=np.c_[hv['x'],hv['y'],hv['z']],axis=1).sum()),
        GATE=dict(dataset=str(gate),sha256={str(f):sha(f) for f in sparse.iterdir()}),
        only_difference_GATE_vs_HULL='Initial point xyz coordinates. Same number, track IDs, order, RGB, cameras, images, train split and downstream code.',
        FILTER='Strict unchanged-point silhouette filter retained as a diagnostic; no downstream FILTER training in this two-run experiment.')
    save(OUT/'surface_checks.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
