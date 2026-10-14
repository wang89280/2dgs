"""Training-only silhouette and reliable foreground normal constraints."""
import torch
import torch.nn.functional as F


def mask_regions(mask, radius=3):
    """Exclude an uncertain band around the silhouette from alpha/normal loss."""
    binary = (mask > 0.5).float().detach()
    k = 2 * radius + 1
    expanded = F.max_pool2d(binary[None], k, stride=1, padding=radius)[0]
    eroded = -F.max_pool2d(-binary[None], k, stride=1, padding=radius)[0]
    return eroded, 1.0 - expanded


def masked_mean(value, mask):
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def stability_losses(pkg, foreground, background, iteration, opt):
    alpha = pkg['rend_alpha']
    # A black but opaque background cannot satisfy this supervision.
    bg_loss = masked_mean(alpha, background)
    fg_loss = masked_mean(1.0 - alpha, foreground)
    silhouette = opt.lambda_alpha_bg * bg_loss + opt.lambda_alpha_fg * fg_loss
    weight = opt.lambda_normal * min(1.0, max(0.0,
        (iteration - opt.normal_from_iter) / max(1, opt.normal_ramp_iters)))
    rn, sn = pkg['rend_normal'], pkg['surf_normal']
    with torch.no_grad():
        reliable = foreground * (alpha.detach() > 0.5)
        reliable = reliable * (rn.detach().norm(dim=0, keepdim=True) > 1e-6)
        reliable = reliable * (sn.detach().norm(dim=0, keepdim=True) > 1e-6)
        reliable = reliable * torch.isfinite(pkg['surf_depth']) * (pkg['surf_depth'].detach() > 0)
    # Normalize both vectors: increasing opacity is not a way to align normals.
    cosine = (F.normalize(rn, dim=0, eps=1e-6) * F.normalize(sn, dim=0, eps=1e-6)).sum(0, keepdim=True)
    normal = weight * masked_mean(1.0 - cosine.clamp(-1, 1), reliable)
    distortion = opt.lambda_dist * masked_mean(pkg['rend_dist'], reliable)
    return silhouette, normal, distortion, bg_loss, fg_loss
