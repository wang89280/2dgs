"""Training-view only illustrations; no evaluation image is loaded."""
import json
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from plyfile import PlyData
from diagnose_sfm_seed_surface import DATA,cameras
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sfm_surface_init_20261014'
def main():
    data=np.load(OUT/'surface_result.npz');cloud=PlyData.read(DATA/'sparse/0/points3D.ply')['vertex']
    xyz=np.c_[cloud['x'],cloud['y'],cloud['z']].astype(float);new=data['xyz'];_,_,k,views=cameras();records=[]
    def project(x,r,t):
        z=x@r.T+t;return z[:,:2]/z[:,2,None]*np.diag(k)[:2]+k[:2,2]
    for ci,(r,t) in views.items():
        if ci%8==0:continue
        mask=cv2.imread(str(DATA/'masks_train'/f'{ci+1:04d}.png'),0);distance=cv2.distanceTransform((mask==0).astype('uint8'),cv2.DIST_L2,5)
        uv=project(xyz,r,t);sample=np.concatenate([cv2.remap(distance,z[:,0].astype('f4')[:,None],z[:,1].astype('f4')[:,None],cv2.INTER_LINEAR).ravel() for z in np.array_split(uv,4)])
        bad=(sample>3)&data['keep'];records.append((int(bad.sum()),ci,bad))
    records.sort(key=lambda x:(x[0],x[1]));chosen=[records[-1],records[len(records)//2]]
    canvas=Image.new('RGB',(1200,900),'#eeeeee');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('DejaVuSans.ttf',17)
    except OSError:font=ImageFont.load_default()
    manifest=[]
    for row,(count,ci,bad) in enumerate(chosen):
        name=f'{ci+1:04d}';r,t=views[ci];im=Image.open(DATA/'images'/(name+'.png')).convert('RGB')
        mask=np.asarray(Image.open(DATA/'masks_train'/(name+'.png')))>0;yy,xx=np.where(mask)
        box=(max(0,int(xx.min())-35),max(0,int(yy.min())-35),min(im.width,int(xx.max())+36),min(im.height,int(yy.max())+36))
        select=np.flatnonzero(bad)[::max(1,int(bad.sum())//180)][:180]
        for col,(x,label,color) in enumerate([(xyz,'Original projections',(240,55,55)),(new,'Constrained projections',(40,230,120))]):
            crop=im.crop(box);painter=ImageDraw.Draw(crop);uv=project(x[select],r,t)
            for u,v in uv:
                x0=u-box[0];y0=v-box[1];painter.ellipse((x0-1.2,y0-1.2,x0+1.2,y0+1.2),fill=color)
            crop.thumbnail((580,390),Image.Resampling.LANCZOS);canvas.paste(crop,(col*600+(600-crop.width)//2,row*450+50))
            draw.text((col*600+12,row*450+12),f'{name}: {label}',fill='black',font=font)
        manifest.append(dict(name=name,selection='largest violating count' if row==0 else 'median violating count',violating_retained_points=count,displayed_points=len(select),point_ids=select.tolist()))
    canvas.save(OUT/'training_constraint_examples.png')
    (OUT/'constraint_example_selection.json').write_text(json.dumps(dict(training_views_only=True,examples=manifest,
        limitation='All selected projected 3D points are shown, including occluded points; illustrates silhouette feasibility, not visible surface truth or final rendering.'),indent=2))
if __name__=='__main__':main()
