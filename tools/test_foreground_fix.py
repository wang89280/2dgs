"""Loss-gradient and optimizer/radius regression tests for the isolated fix."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/2dgs_foreground_fix'))
import torch
from argparse import Namespace
from scene import GaussianModel
from utils.foreground_stability import mask_regions, stability_losses

def model():
    g=GaussianModel(3)
    vals={'xyz':('_xyz',torch.zeros(2,3)), 'f_dc':('_features_dc',torch.zeros(2,1,3)),
        'f_rest':('_features_rest',torch.zeros(2,15,3)), 'opacity':('_opacity',torch.full((2,1),2.)),
        'scaling':('_scaling',torch.full((2,2),-7.)), 'rotation':('_rotation',torch.tensor([[1.,0.,0.,0.]]*2))}
    groups=[]
    for name,(attr,v) in vals.items():
        param=torch.nn.Parameter(v.cuda());setattr(g,attr,param);groups.append({'params':[param],'name':name})
    g.optimizer=torch.optim.Adam(groups,lr=0.);g.percent_dense=.01
    g.max_radii2D=torch.tensor([99.,8.],device='cuda')
    g.xyz_gradient_accum=torch.zeros(2,1,device='cuda');g.denom=torch.ones(2,1,device='cuda')
    return g

def main():
    g=model();g.densify_and_prune(1.,.05,1.,20.)
    assert len(g.get_xyz)==1, 'Screen-radius pruning still resets before use'
    assert len(g.max_radii2D)==1 and g.max_radii2D[0]==0
    g=model();n=g.prune_unstable(1.,20.);assert n==1 and len(g.get_xyz)==1
    # Optimizer tensors remain aligned after a real Adam step and pruning.
    g=model();sum(p.square().sum() for group in g.optimizer.param_groups for p in group['params']).backward()
    g.optimizer.step();g.optimizer.zero_grad(set_to_none=True);g.prune_unstable(1.,20.)
    for group in g.optimizer.param_groups:
        p=group['params'][0];assert p.shape[0]==1
        assert g.optimizer.state[p]['exp_avg'].shape==p.shape
    m=torch.zeros(1,32,32,device='cuda');m[:,8:24,8:24]=1
    fg,bg=mask_regions(m);assert not bool((fg*bg).any())
    a=torch.full_like(m,.8,requires_grad=True)
    direction=torch.zeros(3,32,32,device='cuda');direction[2]=1
    pkg=dict(rend_alpha=a,rend_normal=direction*a,surf_normal=direction*a.detach(),
        surf_depth=torch.ones_like(a),rend_dist=torch.zeros_like(a))
    opt=Namespace(lambda_alpha_bg=.1,lambda_alpha_fg=.01,lambda_normal=.05,
        normal_from_iter=15000,normal_ramp_iters=5000,lambda_dist=0.)
    silhouette,normal,dist,_,_=stability_losses(pkg,fg,bg,7000,opt)
    assert float(normal)==0
    silhouette.backward(retain_graph=True)
    assert bool((a.grad[bg.bool()]>0).all()) and bool((a.grad[fg.bool()]<0).all())
    assert bool((a.grad[(fg+bg)==0]==0).all())
    a.grad.zero_();_,normal,_,_,_=stability_losses(pkg,fg,bg,20000,opt);normal.backward()
    assert float(a.grad.abs().max())<1e-6, 'Normal alignment rewards opacity'
    print('PASS: screen pruning, post-densification pruning, optimizer alignment, alpha gradients, uncertainty band, normal/opacity decoupling',flush=True)

if __name__=='__main__':main()
