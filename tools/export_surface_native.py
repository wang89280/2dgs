"""Write the point corrections in native SfM coordinates; all observations remain."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
sys.path.insert(0,str(ROOT/'code/sfm/python'))
from model_io import OBS
from geometry import replica_parameters,write_ply
from export_sfm_render_pose import metrics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2),encoding='utf8')
def main():
    prep=json.loads((ROOT/'outputs/sfm_render_pose_20261014/geometry_preparation.json').read_text(encoding='utf8'))
    source=Path(prep['native_source']);dest=OUT/'sfm_candidate';assert not dest.exists()
    for n,h in prep['protected_sha256'].items():assert sha(n)==h,n
    data=np.load(OUT/'surface_result.npz');update=np.load(OUT/'sfm_landmark_updates.npz')
    x=np.fromfile(source/'points.bin','<f8').reshape(-1,3);new=x.copy()
    changed=data['keep']&~data['filter_keep'];ids=update['original_track_ids'];new[ids[changed]]=update['xyz_native'][changed]
    obs=np.fromfile(source/'observations.bin',OBS);pose=np.fromfile(source/'poses.bin','<f8').reshape(-1,12);k=replica_parameters(source/'calibration.bin')
    comparison=dict(original=metrics(x,pose,obs,k),candidate_all_original_observations=metrics(new,pose,obs,k),
        coordinate_system='Native3072x2048 distorted pixels; identical cameras, K, 257967 feature observations and all40258 active tracks.',
        changed_tracks=int(changed.sum()),scope='No measurement deleted from this SfM comparison. GS initialization gating is a separate active-point subset; hull feasibility is a constraint, not ground-truth accuracy.')
    dest.mkdir()
    for n in ['camera_ids.bin','calibration.bin','calibration.json','poses.bin','active_points.bin','observations.bin','cameras.xml','fixed_xml_export.json']:shutil.copy2(source/n,dest/n)
    new.astype('<f8').tofile(dest/'points.bin');ids[data['keep']].astype('<u4').tofile(dest/'initialization_active_points.bin')
    (dest/'sparse').mkdir();active=np.fromfile(dest/'active_points.bin','<u4');write_ply(dest/'sparse/points.ply',new[active],np.full((len(active),3),127,np.uint8))
    summary=json.loads((source/'summary.json').read_text());summary.update(reprojection_rmse_px=comparison['candidate_all_original_observations']['rmse_px'],
        reprojection_max_px=comparison['candidate_all_original_observations']['max_px'],track_membership_changed=False)
    save(dest/'summary.json',summary);save(OUT/'native_geometry_comparison.json',comparison)
    for n,h in prep['protected_sha256'].items():assert sha(n)==h,n
    save(OUT/'native_export_audit.json',dict(passed=True,source_unchanged=True,poses_calibration_observations_unchanged=True,
        corrected_tracks=int(changed.sum()),initialization_points=int(data['keep'].sum()),
        candidate_sha256={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}))
    print(json.dumps(comparison,indent=2))
if __name__=='__main__':main()
