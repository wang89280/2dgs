"""Build FILTER and HULL SfM initializations from one immutable base reconstruction."""
import json,hashlib,shutil,sys
from pathlib import Path
import cv2,numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
sys.path[:0]=[str(ROOT/'code/sfm'),str(ROOT/'code/sfm/python')]
from surface_triangulation import refine,POLICY
from diagnose_sfm_seed_surface import DATA,cameras
WORK=Path('/home/wk/01_code/03_gs/sfm-surface-init-20261014')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')
def main():
    if (OUT/'prepared.json').exists():raise FileExistsError('Already prepared; use frozen results')
    OUT.mkdir(parents=True,exist_ok=True)
    previous=json.loads((ROOT/'outputs/foreground_fix_2dgs_2-1_20261014/protocol.json').read_text())
    w,h,k,views=cameras();ids=sorted(views);assert ids==list(range(80))
    r=np.array([views[i][0] for i in ids]);t=np.array([views[i][1] for i in ids])
    training=[int(n)-1 for n in previous['train_names']];test=[int(n)-1 for n in previous['test_names']]
    assert not set(training)&set(test)
    fields=[];mask_hashes={}
    for ci in training:
        file=DATA/'masks_train'/f'{ci+1:04d}.png';mask_hashes[str(file)]=sha(file)
        m=cv2.imread(str(file),0);assert m is not None
        d=cv2.distanceTransform((m>0).astype('uint8'),cv2.DIST_L2,5)-cv2.distanceTransform((m==0).astype('uint8'),cv2.DIST_L2,5)
        fields.append(np.stack([d,cv2.Sobel(d,cv2.CV_32F,1,0,ksize=3)/8,cv2.Sobel(d,cv2.CV_32F,0,1,ksize=3)/8]))
    fields=np.stack(fields)
    cloud=PlyData.read(DATA/'sparse/0/points3D.ply');v=cloud['vertex'].data
    xyz=np.c_[v['x'],v['y'],v['z']].astype(float)
    anchor=np.load(ROOT/'outputs/sfm_render_pose_20261014/geometry_anchors.npz')
    distance,index=cKDTree(anchor['xyz']).query(xyz);assert distance.max()<1e-6 and len(set(index))==len(index)
    lookup=np.full(len(anchor['xyz']),-1,int);lookup[index]=np.arange(len(index))
    valid=lookup[anchor['track']]>=0;point=lookup[anchor['track'][valid]];camera=anchor['camera'][valid];uv=anchor['uv'][valid]
    protected={str(DATA/'sparse/0'/n):sha(DATA/'sparse/0'/n) for n in ['cameras.txt','images.txt','points3D.ply']}
    protected.update(mask_hashes)
    scripts=[Path(__file__),ROOT/'code/sfm/surface_triangulation.py',ROOT/'tools/diagnose_sfm_seed_surface.py']
    protocol=dict(policy=POLICY,base_dataset=str(DATA),training_names=previous['train_names'],test_names=previous['test_names'],
        protected_sha256=protected,code_sha256={str(f.relative_to(ROOT)):sha(f) for f in scripts},
        calibration_and_poses_fixed=True,features_and_correspondences_fixed=True,training_masks_only=True,
        no_rendered_proxy=True,no_new_matches_or_neural_features=True,full80_initial_sfm_geometry=True,
        scope='Silhouette-constrained SfM triangulation and initialization selection. No geometric ground truth; selection RMSE is not a matched accuracy comparison.')
    save(OUT/'surface_protocol.json',protocol)
    result=refine(xyz,r,t,k,fields,training,camera,point,uv,lambda z:print(json.dumps(z),flush=True))
    np.savez_compressed(OUT/'surface_result.npz',track_ids=index,**{n:v for n,v in result.items() if n!='trace'})
    save(OUT/'optimization_trace.json',result['trace'])
    stats=dict(input_points=len(xyz),filter_points=int(result['filter_keep'].sum()),hull_points=int(result['keep'].sum()),
        recovered_by_constrained_triangulation=int((result['keep']&~result['filter_keep']).sum()),
        rejected_initialization_points=int((~result['keep']).sum()),
        original_measurement_rmse_1600px=float(np.sqrt(np.mean(result['before_rmse'][point]**2))),
        all_point_proposal_rmse_1600px=float(np.sqrt(np.mean(result['after_rmse'][point]**2))),
        original_point_coordinates_preserved_when_feasible=True,original_native_model_preserved=True,
        note='RMSE above uses the same original observation counts per point; proposed points that fail gates are not exported. It is not a ground-truth metric.')
    outputs={}
    for method,key in [('FILTER','filter_keep'),('HULL','keep')]:
        dataset=WORK/'datasets'/method;sparse=dataset/'sparse/0';assert not dataset.exists();sparse.mkdir(parents=True)
        for name in ['cameras.txt','images.txt']:shutil.copy2(DATA/'sparse/0'/name,sparse/name)
        for name in ['images','masks_train']:(dataset/name).symlink_to((DATA/name).resolve(),target_is_directory=True)
        cc=PlyData.read(DATA/'sparse/0/points3D.ply');vv=cc['vertex'].data
        if method=='HULL':
            for j,n in enumerate(['x','y','z']):vv[n]=result['xyz'][:,j]
        cc['vertex'].data=vv[result[key]].copy();cc.write(sparse/'points3D.ply')
        save(OUT/f'{method}_track_ids.json',index[result[key]].tolist())
        outputs[method]=dict(dataset=str(dataset),points=int(result[key].sum()),sha256={str(f):sha(f) for f in sparse.iterdir()})
    # Portable native coordinate sidecar; immutable source cameras/observations are referenced, not copied or edited.
    native=(result['xyz']-anchor['gauge_translation'])@anchor['gauge_rotation']/float(anchor['gauge_scale'])
    np.savez_compressed(OUT/'sfm_landmark_updates.npz',original_track_ids=index,xyz_native=native,initialization_active=result['keep'])
    for name,value in protected.items():assert sha(name)==value,name
    for name,value in protocol['code_sha256'].items():assert sha(ROOT/name)==value,name
    save(OUT/'prepared.json',dict(complete=True,stats=stats,outputs=outputs,original_files_unchanged=True))
    print(json.dumps(stats,indent=2),flush=True)
if __name__=='__main__':main()
