"""Full-resolution ALIKED extraction with overlapping tiles and global deduplication."""
import numpy as np,cv2
import deep_frontend_common as front
def extract(session,path,args):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError(f'Cannot decode image: {path}')
    h,w=image.shape[:2];tw,th=1536,1024;overlap=384;guard=96
    def starts(length,tile):
        if length<=tile:return [0]
        return sorted(set([*range(0,length-tile+1,tile-overlap),length-tile]))
    keys=[];descriptors=[];scores=[]
    for y0 in starts(h,th):
        for x0 in starts(w,tw):
            crop=image[y0:min(y0+th,h),x0:min(x0+tw,w)];ch,cw=crop.shape[:2]
            rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB).astype(np.float32)/255
            kp,desc,score=session.run(None,{'image':np.transpose(rgb[None],(0,3,1,2)),'max_keypoints':np.array(args.max_keypoints,np.int64),'min_score':np.array(args.min_score,np.float32)})
            kp=kp.reshape(-1,2).astype(np.float32);desc=desc.reshape(-1,desc.shape[-1]);score=score.ravel()
            if kp.size and kp.max()<=2 and kp.min()>=-2:kp=(kp+1)*np.array([cw-1,ch-1],np.float32)*.5+.5
            valid=(kp[:,0]>= (guard if x0>0 else 0))&(kp[:,0]<(cw-guard if x0+cw<w else cw))&(kp[:,1]>=(guard if y0>0 else 0))&(kp[:,1]<(ch-guard if y0+ch<h else ch))
            kp[:,0]+=x0;kp[:,1]+=y0;keys.append(kp[valid]);descriptors.append(desc[valid]);scores.append(score[valid])
    kp=np.concatenate(keys);desc=np.concatenate(descriptors);score=np.concatenate(scores);selected=[];buckets={};radius=3.
    for i in np.argsort(-score,kind='stable'):
        x,y=np.floor(kp[i]/radius).astype(int);near=[j for a in range(x-1,x+2) for b in range(y-1,y+2) for j in buckets.get((a,b),[])]
        if any(np.sum((kp[i]-kp[j])**2)<radius**2 for j in near):continue
        selected.append(i);buckets.setdefault((x,y),[]).append(i)
        if len(selected)>=args.max_keypoints:break
    return front.FeatureSet(kp[selected],front.normalize_descriptors(desc[selected]),score[selected],(w,h))
