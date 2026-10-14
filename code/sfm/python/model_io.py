"""Sparse model I/O and observation audits for the final pipeline."""
from pathlib import Path
import json,hashlib
import numpy as np
from geometry import replica_parameters
OBS=np.dtype([('camera','<u4'),('track','<u4'),('aux','<u8'),('u','<f4'),('v','<f4'),('depth','<f4'),('size','<f4')])

def read(p):return json.loads(p.read_text(encoding='utf8'))

def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf8')

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def audit(source,dest,graph):
    ids=np.fromfile(dest/'sfm/camera_ids.bin','<u4');beforeids=np.fromfile(source/'sfm/camera_ids.bin','<u4')
    obs=np.fromfile(dest/'sfm/observations.bin',OBS);old=np.fromfile(source/'sfm/observations.bin',OBS)
    def keys(rows,ids):return [(int(ids[o['camera']]),float(o['u']),float(o['v'])) for o in rows]
    oldset=set(keys(old,beforeids));newkeys=keys(obs,ids);newset=set(newkeys)
    assert len(newkeys)==len(newset)
    assert len(set(zip(map(int,obs['track']),map(int,obs['camera']))))==len(obs)
    nodes=np.fromfile(graph/'nodes.bin',np.dtype([('camera','<u4'),('feature','<u4'),('u','<f4'),('v','<f4')]))
    allowed={(int(n['camera']),float(n['u']),float(n['v'])) for n in nodes}
    assert newset<=allowed
    result=dict(passed=True,original_observations=len(oldset),output_observations=len(newset),retained_original_observations=len(newset&oldset),removed_original_observations=len(oldset-newset),added_original_feature_observations=len(newset-oldset),original_retention_fraction=len(newset&oldset)/len(oldset),unique_ownership=True,original_pixel_coordinates=True)
    old_owner=dict(zip(keys(old,beforeids),map(int,old['track'])))
    new_owner=dict(zip(newkeys,map(int,obs['track'])))
    result['surviving_observations_with_changed_track_id']=sum(old_owner[k]!=new_owner[k] for k in newset&oldset)
    result['track_membership_changed']=(oldset!=newset or result['surviving_observations_with_changed_track_id']>0)
    save(dest/'observation_audit.json',result);return result

def patch_xml(sfm):
    import xml.etree.ElementTree as ET
    p=replica_parameters(sfm/'calibration.bin');path=sfm/'cameras.xml';tree=ET.parse(path);cal=tree.getroot().find('chunk/sensors/sensor/calibration')
    # Original no-EXIF exporter omitted affine terms because they were zero in that branch.
    for name in ('b1','b2'):
        e=cal.find(name)
        if e is None:e=ET.SubElement(cal,name)
        e.text=format(p[name],'.17g')
    tree.write(path,encoding='utf-8',xml_declaration=True)
    # Existing dense parser requires exactly '<sensors ' and '<cameras ' delimiters.
    text=path.read_text(encoding='utf-8').replace('<sensors>','<sensors >').replace('<cameras>','<cameras >')
    path.write_text(text,encoding='utf-8');save(sfm/'fixed_xml_export.json',dict(b1=p['b1'],b2=p['b2'],reason='Retain externally fixed fx != fy in downstream dense reconstruction'))
