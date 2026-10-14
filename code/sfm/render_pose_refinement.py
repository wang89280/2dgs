"""Camera-only differentiable alignment against a frozen surfel proxy.

The rasterizer has no camera-matrix backward. A camera perturbation is represented
by an exactly equivalent rigid transform of the frozen Gaussian geometry, with
SH colors evaluated in the ORIGINAL world coordinate system. No Gaussian, RGB,
feature measurement, calibration or held-out camera is optimized here.
"""
from types import SimpleNamespace
import torch
import torch.nn.functional as F


def skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def so3_exp(v):
    theta = torch.linalg.vector_norm(v, dim=-1)
    k = skew(v)
    eye = torch.eye(3, dtype=v.dtype, device=v.device)
    return eye + torch.sinc(theta / torch.pi)[..., None, None] * k + (
        0.5 * torch.sinc(theta / (2 * torch.pi)).square()[..., None, None] * (k @ k))


def quat_exp(v):
    theta = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    return torch.cat((torch.cos(theta / 2), 0.5 * torch.sinc(theta / (2 * torch.pi)) * v), -1)


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.linalg.cross(av.expand_as(bv), bv, dim=-1)), -1)


def camera_state(camera, pixel_parameters, depth):
    """Left perturbation of world-to-camera, parameters approximately in pixels."""
    w2c = camera.world_view_transform.T.to(pixel_parameters)
    r0, t0 = w2c[:3, :3], w2c[:3, 3]
    fx = camera.image_width / (2 * __import__('math').tan(camera.FoVx/2))
    fy = camera.image_height / (2 * __import__('math').tan(camera.FoVy/2))
    f = (fx*fy)**0.5
    rot = pixel_parameters[:3] / pixel_parameters.new_tensor([fy, fx, f])
    shift = pixel_parameters[3:] * depth / pixel_parameters.new_tensor([fx, fy, f])
    d = so3_exp(rot)
    identity = torch.eye(3, dtype=d.dtype, device=d.device)
    # Incremental form makes the zero perturbation exact in float32, avoiding
    # a spurious R0.T @ R0 transform and cancellation of large translations.
    dm = d-identity
    r, t = r0+dm @ r0, t0+dm @ t0+shift
    a, b = identity+r0.T @ dm @ r0, r0.T @ (dm @ t0+shift)
    center = camera.camera_center.to(pixel_parameters)-r0.T @ d.T @ shift
    return dict(r=r, t=t, center=center, a=a, b=b, q=quat_exp(r0.T @ rot), rot=rot, shift=shift)


def project(camera, state, xyz):
    import math
    v = xyz @ state['r'].T + state['t']
    f = xyz.new_tensor([camera.image_width/(2*math.tan(camera.FoVx/2)),
                        camera.image_height/(2*math.tan(camera.FoVy/2))])
    center = xyz.new_tensor([(camera.image_width-1)/2, (camera.image_height-1)/2])
    return v[:, :2] / v[:, 2:] * f + center


def preconditioner(camera, xyz, depth):
    """Whiten local image displacements; floor weak modes to limit ambiguity."""
    zero = xyz.new_zeros(6, requires_grad=True)
    jac = torch.autograd.functional.jacobian(
        lambda p: project(camera, camera_state(camera, p, depth), xyz), zero, vectorize=True)
    j = jac.reshape(-1, 6)
    g = j.T @ j / len(xyz)
    values, vectors = torch.linalg.eigh(g)
    floor = values.max().clamp_min(1e-6) * 0.01
    q = (vectors * values.clamp_min(floor).rsqrt()[None, :]) @ vectors.T
    return q.detach(), values.detach()


def render_delta(camera, gaussians, pipe, background, pixel_parameters, depth):
    from gaussian_renderer import render
    from utils.sh_utils import eval_sh
    s = camera_state(camera, pixel_parameters, depth)
    original_xyz = gaussians.get_xyz.detach()
    xyz = original_xyz @ s['a'].T + s['b']
    rotation = quat_mul(s['q'], gaussians.get_rotation.detach())
    proxy = SimpleNamespace(get_xyz=xyz, get_rotation=rotation,
        get_scaling=gaussians.get_scaling.detach(), get_opacity=gaussians.get_opacity.detach(),
        active_sh_degree=gaussians.active_sh_degree, max_sh_degree=gaussians.max_sh_degree)
    directions = F.normalize(original_xyz-s['center'], dim=-1)
    features = gaussians.get_features.detach().transpose(1, 2)
    colors = torch.clamp_min(eval_sh(gaussians.active_sh_degree, features, directions)+0.5, 0)
    return render(camera, proxy, pipe, background, override_color=colors), s


def detached_camera(camera, state):
    import copy
    c = copy.copy(camera)
    w = torch.eye(4, device=c.world_view_transform.device)
    w[:3, :3] = state['r'].detach(); w[:3, 3] = state['t'].detach()
    c.world_view_transform = w.T.contiguous()
    c.full_proj_transform = c.world_view_transform @ c.projection_matrix
    c.camera_center = state['center'].detach()
    return c
