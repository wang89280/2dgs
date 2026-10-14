"""Complete verified graph construction and final track processing."""
from pathlib import Path
import argparse,json,sqlite3,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python'))
from model_io import read,save,audit,patch_xml,digest
from evaluate import load_replica,model_metrics

def complete_graph(shared,out):
    if out.exists():raise FileExistsError(out)
    dbpath=shared/'database.db'
    db=sqlite3.connect('file:'+dbpath.resolve().as_posix()+'?mode=ro',uri=True)
    images=sorted(db.execute('SELECT image_id,name FROM images'),key=lambda r:r[1]);index={iid:i for i,(iid,name) in enumerate(images)}
    keys={iid:np.frombuffer(data,'<f4').reshape(n,c)[:,:2].copy() for iid,n,c,data in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    pairs=[];features=set()
    for pid,n,c,data in db.execute('SELECT pair_id,rows,cols,data FROM two_view_geometries WHERE rows>0'):
        assert c==2;i,j=divmod(pid,2147483647)
        for a,b in np.frombuffer(data,'<u4').reshape(n,c):
            u=(i,int(a));v=(j,int(b));features.update((u,v));pairs.append((u,v))
    db.close()
    ordered=sorted(features,key=lambda q:(index[q[0]],q[1]));lookup={key:i for i,key in enumerate(ordered)}
    nodes=[(index[i],f,float(keys[i][f,0]),float(keys[i][f,1])) for i,f in ordered]
    edges=sorted(set(tuple(sorted((lookup[a],lookup[b]))) for a,b in pairs))
    dtype=np.dtype([('camera','<u4'),('feature','<u4'),('u','<f4'),('v','<f4')])
    out.mkdir(parents=True);np.array(nodes,dtype).tofile(out/'nodes.bin');np.array(edges,dtype='<u4').tofile(out/'edges.bin')
    save(out/'manifest.json',dict(nodes=len(nodes),edges=len(edges),source_database=str(dbpath.resolve()),source_database_sha256=digest(dbpath),construction='All endpoints of verified edges, before greedy track partition. No descriptor extraction, matching or camera estimation is performed here.',node_format='uint32 camera, uint32 feature, float32 u, float32 v',edge_format='two uint32 node indexes',images=[dict(camera=i,database_id=iid,name=name) for i,(iid,name) in enumerate(images)]))

def distribution(model):
    n=np.array([len(obs) for x,obs in model['points'].values()])
    views=np.array([len({i for i,uv in obs}) for x,obs in model['points'].values()])
    return dict(points=len(n),observations=int(n.sum()),unique_point_image_pairs=int(views.sum()),mean_observations=float(n.mean()),mean_unique_views=float(views.mean()),median_unique_views=float(np.median(views)),two_unique_views=int(sum(views==2)),five_plus_unique_views=int(sum(views>=5)),twelve_plus_unique_views=int(sum(views>=12)),twelve_plus_observations=int(sum(n>=12)),tracks_with_duplicate_image=int(sum(n>views)),extra_same_image_observations=int(sum(n-views)))

def optimizer_budget(folder):
    rows=[json.loads(line) for line in (folder/'sfm/optimization.jsonl').read_text().splitlines() if line.strip()]
    return dict(ba_calls=sum(bool(r.get('ba_call')) for r in rows),
                irls_steps=sum('irls' in r for r in rows),
                linear_iterations=sum('iteration' in r for r in rows),
                accepted_linear_steps=sum('iteration' in r and r.get('accepted',False) for r in rows),
                linear_failures=sum(bool(r.get('linear_failure')) for r in rows))

def execute(source,graph,out,engine='maintain_tracks',track_recovery=True,ba_rounds=None):
    if engine not in {'rebuild_tracks','maintain_tracks'}:raise ValueError(engine)
    if ba_rounds is not None and (track_recovery or engine!='maintain_tracks' or not 0<=ba_rounds<=3):
        raise ValueError('BA call replay is only supported for disabled maintenance, with 0..3 calls')
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.time()
    argv=[str(ROOT/'bin'/f'{engine}.exe'),str(source/'sfm'),str(out/'sfm'),str(graph)]
    if not track_recovery:argv.append('--no-track-recovery')
    if ba_rounds is not None:argv.extend(['--ba-rounds',str(ba_rounds)])
    save(out/'command.json',dict(argv=argv,track_recovery=track_recovery,ba_rounds=ba_rounds))
    with (out/'run.log').open('w') as log:
        subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True)
    patch_xml(out/'sfm');check=audit(source,out,graph)
    if engine=='maintain_tracks':assert check['removed_original_observations']==0
    if not track_recovery:
        assert check['added_original_feature_observations']==0
        assert check['surviving_observations_with_changed_track_id']==0
    native_summary=read(out/'sfm/summary.json')
    native_summary.update(track_recovery_enabled=track_recovery,track_membership_changed=check['track_membership_changed'])
    save(out/'sfm/summary.json',native_summary)
    assert (source/'sfm/calibration.bin').read_bytes()==(out/'sfm/calibration.bin').read_bytes()
    m=load_replica(out);metrics=model_metrics(m,out);d=distribution(m)
    nodes=np.fromfile(graph/'nodes.bin',np.dtype([('camera','<u4'),('feature','<u4'),('u','<f4'),('v','<f4')]))
    pending=np.fromfile(out/'sfm/pending_nodes.bin','<u4');assert len(set(map(int,pending)))==len(pending) and np.all(pending<len(nodes))
    active={(i,float(uv[0]),float(uv[1])) for x,obs in m['points'].values() for i,uv in obs}
    deferred={(int(n['camera']),float(n['u']),float(n['v'])) for n in nodes[pending]}
    assert not active&deferred and len(active)+len(deferred)==len(nodes)
    save(out/'pending_audit.json',dict(passed=True,graph_observations=len(nodes),active=len(active),pending=len(pending),active_pending_intersection=0,active_plus_pending_equals_complete_graph=True))
    save(out/'optimizer_budget.json',optimizer_budget(out))
    save(out/'track_distribution.json',d);save(out/'runtime.json',dict(seconds=time.time()-start,stage=engine,track_recovery=track_recovery));return m,metrics,d

