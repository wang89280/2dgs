"""Track bridge and exact replica camera projection. No reconstruction side effects."""
import csv
import sqlite3
from pathlib import Path
import numpy as np

def build_tracks(database, destination, sigma=1.0):
    """Conservative conflict-aware union of verified pair correspondences."""
    db=sqlite3.connect(f'file:{Path(database).as_posix()}?mode=ro',uri=True)
    images=sorted(db.execute('SELECT image_id,name FROM images'),key=lambda row:row[1])
    image_index={iid:i for i,(iid,name) in enumerate(images)}
    keypoints={iid:np.frombuffer(data,np.float32).reshape(rows,cols).copy()
               for iid,rows,cols,data in db.execute('SELECT image_id,rows,cols,data FROM keypoints')}
    parent={}; members={}; conflicts=0; accepted=0
    def root(node):
        if node not in parent: parent[node]=node;members[node]={node[0]}
        while node!=parent[node]:
            parent[node]=parent[parent[node]];node=parent[node]
        return node
    # Stronger verified image pairs first; deterministic ties.
    for pair_id,rows,data in db.execute('SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0 ORDER BY rows DESC,pair_id'):
        a,b=divmod(pair_id,2147483647)
        for x,y in np.frombuffer(data,np.uint32).reshape(rows,2):
            if x>=len(keypoints[a]) or y>=len(keypoints[b]):raise ValueError('Invalid feature index')
            n,m=(a,int(x)),(b,int(y));r,s=root(n),root(m)
            if r==s:continue
            if members[r]&members[s]:conflicts+=1;continue
            if len(members[r])<len(members[s]):r,s=s,r
            parent[s]=r;members[r]|=members.pop(s);accepted+=1
    groups={}
    for node in sorted(parent):groups.setdefault(root(node),[]).append(node)
    tracks=sorted((sorted(t) for t in groups.values() if len(t)>=2),key=lambda t:t[0])
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',newline='',encoding='utf-8') as stream:
        out=csv.writer(stream);out.writerow(['track','image','feature','u','v','size'])
        for tid,track in enumerate(tracks):
            for iid,fid in track:
                u,v=keypoints[iid][fid,:2]
                if not np.isfinite([u,v]).all():raise ValueError('Nonfinite keypoint')
                out.writerow([tid,image_index[iid],fid,float(u),float(v),sigma])
    db.close()
    return {'tracks':len(tracks),'observations':sum(map(len,tracks)),
            'conflicting_edges_rejected':conflicts,'union_edges':accepted,
            'sigma_original_pixels':sigma,'stationary_filter_applied':False,
            'images':[{'database_id':iid,'replica_id':i,'name':name} for i,(iid,name) in enumerate(images)]}

def replica_parameters(path):
    raw=Path(path).read_bytes()
    if len(raw)!=256:raise ValueError('Expected 256-byte replica calibration')
    return {'width':int(np.frombuffer(raw,'<u8',1,8)[0]),'height':int(np.frombuffer(raw,'<u8',1,16)[0]),
            'f':float(np.frombuffer(raw,'<f8',1,24)[0]),'cx':float(np.frombuffer(raw,'<f8',1,32)[0]),
            'cy':float(np.frombuffer(raw,'<f8',1,40)[0]),'b1':float(np.frombuffer(raw,'<f8',1,48)[0]),
            'b2':float(np.frombuffer(raw,'<f8',1,56)[0]),'k':np.frombuffer(raw,'<f8',13,80).copy()}

def project_replica(xy,p):
    """Forward expression from calibration_jacobian.hpp, including radius cap."""
    x,y=np.asarray(xy).T;k=p['k'];radius=x*x+y*y
    rc=np.minimum(radius,k[12]);cs=np.ones_like(radius)
    np.divide(k[12],radius,out=cs,where=radius>k[12])
    rd=rc*k[0]+rc**2*k[1]+rc**3*k[2]+rc**4*k[3]
    tc=1+rc*k[6]+rc**2*k[7];sc=1+rc*k[10]+rc**2*k[11]
    xd=x+((3*x*x+y*y)*k[4]+2*x*y*k[5])*tc*cs+rd*x+rc*k[8]*sc*cs
    yd=y+((3*y*y+x*x)*k[5]+2*x*y*k[4])*tc*cs+rd*y+rc*k[9]*sc*cs
    return np.column_stack(((p['f']+p['b1'])*xd+p['b2']*yd+p['width']/2+p['cx'],p['f']*yd+p['height']/2+p['cy']))

def write_ply(path,xyz,rgb):
    v=np.zeros(len(xyz),dtype=[(n,'<f4') for n in ['x','y','z','nx','ny','nz']]+[(n,'u1') for n in ['red','green','blue']])
    for j,n in enumerate(['x','y','z']):v[n]=xyz[:,j]
    for j,n in enumerate(['red','green','blue']):v[n]=rgb[:,j]
    header='ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(v))+'\n'
    header+=''.join('property float '+n+'\n' for n in ['x','y','z','nx','ny','nz'])
    header+=''.join('property uchar '+n+'\n' for n in ['red','green','blue'])+'end_header\n'
    with Path(path).open('wb') as stream:stream.write(header.encode());stream.write(v.tobytes())
