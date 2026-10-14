"""Silhouette-feasible SfM landmarks with an observation-information metric.

Only point coordinates are proposed here. Cameras, features and correspondences
are immutable. Masks must contain TRAINING views only. This is an initialization
postprocessor, not a new feature matcher or a Gaussian training loss.
"""
import numpy as np
import torch
import torch.nn.functional as F

POLICY=dict(mask_margin_px=3.,constraint_tolerance_px=.05,iterations=24,
    step_projection_rms_px=.75,total_projection_rms_px=3.,max_world_move=.015,
    max_feature_rmse_increase_px=.5,max_feature_rmse_px=2.0,
    hessian_eigen_floor_fraction=1e-6,distance_gradient_floor=1e-6,
    mask_rule='Existing training-only largest filled RGB foreground masks; exact baseline masks.')

def projection(x,r,t,k):
    v=torch.einsum('vij,nj->vni',r,x)+t[:,None,:]
    uv=v[...,:2]/v[...,2:]*k[:2,:2].diagonal()+k[:2,2]
    return uv,v[...,2]

def information(x,camera,point,r,t,k):
    """Per-track J^T J in original undistorted pixel units, no residual reweighting."""
    rr=r[camera];v=np.einsum('nij,nj->ni',rr,x[point])+t[camera]
    j=(rr[:,:2]-v[:,:2,None]/v[:,2,None,None]*rr[:,2:3])/v[:,2,None,None]
    j*=np.diag(k)[:2][None,:,None]
    h=np.zeros((len(x),3,3));np.add.at(h,point,np.einsum('nji,njk->nik',j,j))
    count=np.bincount(point,minlength=len(x));assert (count>=2).all()
    eigen,vectors=np.linalg.eigh(h)
    floor=np.maximum(eigen[:,-1:]*POLICY['hessian_eigen_floor_fraction'],1e-12)
    inverse=(vectors/np.maximum(eigen,floor)[:,None,:])@vectors.transpose(0,2,1)
    return h,inverse,count,eigen

def feature_rmse(x,camera,point,uv,r,t,k):
    v=np.einsum('nij,nj->ni',r[camera],x[point])+t[camera]
    predicted=v[:,:2]/v[:,2,None]*np.diag(k)[:2]+k[:2,2]
    error=np.sum((predicted-uv)**2,axis=1)
    return np.sqrt(np.bincount(point,weights=error,minlength=len(x))/np.bincount(point,minlength=len(x)))

@torch.no_grad()
def refine(x,rotations,translations,k,signed_fields,training_ids,obs_camera,obs_point,obs_uv,progress=None):
    """Project toward the visual hull along least-constrained landmark directions.

    Each step solves the linearized worst silhouette constraint in a Mahalanobis
    metric derived from original observations. All moves are capped in original
    feature projection space, then audited against ALL training silhouettes and
    the original unmodified observations. Infeasible proposals are not exported.
    """
    assert len(training_ids)==len(signed_fields) and len(set(training_ids))==len(training_ids)
    h,c,count,eigen=information(x,obs_camera,obs_point,rotations,translations,k)
    before=feature_rmse(x,obs_camera,obs_point,obs_uv,rotations,translations,k)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    def tensor(a):return torch.as_tensor(np.asarray(a),device=device,dtype=torch.float32)
    x0=tensor(x);current=x0.clone();cov=tensor(c);hm=tensor(h/count[:,None,None])
    r=tensor(rotations[training_ids]);t=tensor(translations[training_ids]);kt=tensor(k)
    fields=tensor(signed_fields);height,width=fields.shape[-2:]
    point_index=torch.arange(len(x),device=device);trace=[]
    def sample(y):
        uv,z=projection(y,r,t,kt)
        grid=uv/uv.new_tensor([width-1,height-1])*2-1
        sampled=F.grid_sample(fields,grid[:,:,None,:],align_corners=True,mode='bilinear',padding_mode='border')[:,:,:,0]
        valid=(grid.abs()<=1).all(-1)&(z>0)
        dist=torch.where(valid,sampled[:,0],-1000.)
        worst,view=dist.min(0)
        return uv,z,sampled,worst,view
    initial=sample(current);initial_distance=initial[3].cpu().numpy()
    for step in range(POLICY['iterations']):
        uv,z,s,dist,view=sample(current);bad=dist< -POLICY['mask_margin_px']
        rr=r[view];zz=z[view,point_index]
        xy=(uv[view,point_index]-kt[:2,2])/kt[:2,:2].diagonal()
        j=(rr[:,:2]-xy[:,:,None]*rr[:,2:3])/zz[:,None,None]
        j*=kt[:2,:2].diagonal()[None,:,None]
        grad=s[view,1:,point_index]
        world_grad=torch.einsum('ni,nij->nj',grad,j)
        direction=torch.einsum('nij,nj->ni',cov,world_grad)
        divisor=(direction*world_grad).sum(-1)
        valid=bad&(divisor>POLICY['distance_gradient_floor'])
        delta=direction*((-dist-POLICY['mask_margin_px']+.02)/divisor.clamp_min(1e-12))[:,None]
        delta[~valid]=0
        projected=torch.einsum('ni,nij,nj->n',delta,hm,delta).clamp_min(0).sqrt()
        delta*=torch.clamp(POLICY['step_projection_rms_px']/projected.clamp_min(1e-12),max=1)[:,None]
        total=current+delta-x0
        total_rms=torch.einsum('ni,nij,nj->n',total,hm,total).clamp_min(0).sqrt()
        factor=torch.minimum(torch.clamp(POLICY['total_projection_rms_px']/total_rms.clamp_min(1e-12),max=1),
            torch.clamp(POLICY['max_world_move']/torch.linalg.vector_norm(total,dim=1).clamp_min(1e-12),max=1))
        current=x0+total*factor[:,None]
        rec=dict(iteration=step+1,violating_points=int(bad.sum()),worst_outside_px=float((-dist).max()))
        trace.append(rec)
        if progress:progress(rec)
    final=sample(current)[3].cpu().numpy();proposed=current.cpu().numpy().astype(float)
    after=feature_rmse(proposed,obs_camera,obs_point,obs_uv,rotations,translations,k)
    feasible=final>=-POLICY['mask_margin_px']-POLICY['constraint_tolerance_px']
    faithful=(after<=before+POLICY['max_feature_rmse_increase_px'])&(after<=POLICY['max_feature_rmse_px'])
    original_good=initial_distance>=-POLICY['mask_margin_px']-POLICY['constraint_tolerance_px']
    # Preserve unchanged seeds exactly; never alter already feasible points.
    proposed[original_good]=x[original_good]
    after[original_good]=before[original_good]
    keep=original_good|(feasible&faithful)
    exported=x.copy();exported[keep]=proposed[keep]
    return dict(xyz=exported,keep=keep,filter_keep=original_good,proposed_xyz=proposed,
        initial_distance=initial_distance,final_distance=final,before_rmse=before,after_rmse=after,
        faithful=faithful,feasible=feasible,information_eigenvalues=eigen,trace=trace)
