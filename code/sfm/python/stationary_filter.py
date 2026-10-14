"""Pixel-radius adaptation of the captured stationary-track voting rule for ALIKED.

ALIKED exports no detector scale. The radius is explicit, in original pixels,
and is independent of the observation sigma passed to bundle adjustment.
"""
from pathlib import Path
import csv,json,shutil,sqlite3,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frontend'))
from geometry import build_tracks
PAIR_BASE=2147483647
DEFAULT_RADIUS_PX=4.0

def save(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')

def stationary(xy,radius):
    """Native vote counts with equal proxy scale radius/4, not measured scale."""
    if not np.isfinite(radius) or radius<=0:raise ValueError('Positive finite radius required')
    xy=np.asarray(xy,dtype=np.float64)
    if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('Invalid coordinates')
    n=len(xy)
    if n<2:return False
    required=1 if n<3 else 2 if n<5 else 3
    delta=xy[:,None,:]-xy[None,:,:]
    close=np.sum(delta*delta,axis=2)<=radius*radius
    np.fill_diagonal(close,False)
    return bool(np.max(close.sum(axis=1))>=required)

def read_tracks(path):
    groups={}
    with Path(path).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):groups.setdefault(int(r['track']),[]).append((int(r['image']),int(r['feature']),float(r['u']),float(r['v'])))
    for obs in groups.values():
        if len({o[0] for o in obs})!=len(obs):raise ValueError('Input track contains duplicate image observations')
    return groups

def filter_database(database,groups,radius=DEFAULT_RADIUS_PX,min_verified_matches=15):
    """Mutate only a caller-owned database copy; remove incident raw/verified edges."""
    conn=sqlite3.connect(database)
    images=sorted(conn.execute('SELECT image_id,name FROM images'),key=lambda r:r[1])
    ids={i:iid for i,(iid,_) in enumerate(images)}
    xy={tid:np.array([(o[2],o[3]) for o in obs]) for tid,obs in groups.items()}
    rejected=[tid for tid,p in xy.items() if stationary(p,radius)]
    blocked={iid:set() for iid,_ in images}
    for tid in rejected:
        for image,feature,_,_ in groups[tid]:blocked[ids[image]].add(feature)
    masks={iid:np.zeros(rows,bool) for iid,rows in conn.execute('SELECT image_id,rows FROM keypoints')}
    for iid,features in blocked.items():
        if features:masks[iid][list(features)]=True
    changes={}
    try:
        with conn:
            for table in ('matches','two_view_geometries'):
                before=after=removed_incident=weak_pairs=weak_matches=0
                for pid,n,c,data in list(conn.execute('SELECT pair_id,rows,cols,data FROM '+table)):
                    before+=n
                    if not n:continue
                    if c!=2:raise ValueError('Expected two feature indices per correspondence')
                    i,j=divmod(pid,PAIR_BASE);m=np.frombuffer(data,'<u4').reshape(n,c)
                    keep=~(masks[i][m[:,0]]|masks[j][m[:,1]])
                    removed_incident+=int((~keep).sum());new=m[keep].copy()
                    if table=='two_view_geometries' and 0<len(new)<min_verified_matches:
                        weak_pairs+=1;weak_matches+=len(new);new=np.empty((0,2),'<u4')
                    assert not masks[i][new[:,0]].any() and not masks[j][new[:,1]].any()
                    after+=len(new)
                    conn.execute('UPDATE '+table+' SET rows=?,data=? WHERE pair_id=?',(len(new),new.tobytes(),pid))
                changes[table]=dict(before=before,after=after,removed_incident=removed_incident,removed_weak_pair_matches=weak_matches,removed_weak_pairs=weak_pairs)
    finally:conn.close()
    # Diagnostic sensitivity only: no reconstruction or outcome-based threshold selection.
    sensitivity={str(r):sum(stationary(p,r) for p in xy.values()) for r in (2.,4.,8.,16.)}
    rejected_features=sorted((iid,int(fid)) for iid,features in blocked.items() for fid in features)
    return dict(applied=True,radius_original_pixels=radius,observation_sigma_used_for_stationary_test=False,
        detector_scale_available=False,native_scale_equivalence_claimed=False,
        rule='At least 1 close other view for 2-view tracks; 2 for 3-4; 3 for >=5. Distance <= explicit radius. Remove entire flagged track and every incident match.',
        verified_models_reestimated=False,min_verified_matches=min_verified_matches,
        input_tracks=len(groups),rejected_tracks=len(rejected),rejected_observations=sum(len(groups[t]) for t in rejected),
        sensitivity_rejected_tracks=sensitivity,rejected_track_ids=rejected,rejected_features=rejected_features,
        database_changes=changes)

def build_tracks_pipeline(database,destination,sigma=4.,radius=DEFAULT_RADIUS_PX):
    database=Path(database);destination=Path(destination)
    if not np.isfinite(radius) or radius<0:raise ValueError('Radius must be finite and nonnegative; 0 disables filtering')
    if radius==0:return build_tracks(database,destination,sigma)
    folder=destination.parent
    for name in ('database_unfiltered.db','observations_unfiltered.csv','tracks_unfiltered.json','stationary_filter.json'):
        if (folder/name).exists():raise FileExistsError(folder/name)
    result=build_tracks(database,destination,sigma)
    src=sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True);dst=sqlite3.connect(folder/'database_unfiltered.db')
    try:src.backup(dst)
    finally:dst.close();src.close()
    shutil.copy2(destination,folder/'observations_unfiltered.csv');save(folder/'tracks_unfiltered.json',result)
    groups=read_tracks(destination);audit=filter_database(database,groups,radius)
    result=build_tracks(database,destination,sigma)
    blocked={tuple(v) for v in audit['rejected_features']};mapping={v['replica_id']:v['database_id'] for v in result['images']}
    rebuilt=read_tracks(destination)
    actual={(mapping[o[0]],o[1]) for rows in rebuilt.values() for o in rows}
    assert not actual&blocked,'Stationary feature survived track rebuilding'
    audit['rebuilt_tracks']=result['tracks'];audit['rebuilt_observations']=result['observations'];audit['rejected_feature_intersection_after_rebuild']=len(actual&blocked)
    audit['passes']=1
    audit['rebuilt_tracks_meeting_rule_after_one_pass']=sum(stationary([(o[2],o[3]) for o in rows],radius) for rows in rebuilt.values())
    save(folder/'stationary_filter.json',audit)
    result.update(stationary_filter_applied=True,stationary_radius_px=radius,stationary_filter_report='stationary_filter.json',unfiltered_tracks=audit['input_tracks'],stationary_tracks_removed=audit['rejected_tracks'])
    return result
