import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import copy

class CosKernel(nn.Module):  # similar to softmax kernel
    def __init__(self, T, learn_temperature=False):
        super().__init__()
        self.learn_temperature = learn_temperature
        if self.learn_temperature:
            self.T = nn.Parameter(torch.tensor(T))
        else:
            self.T = T

    def __call__(self, x, y, eps=1e-6):
        c = torch.einsum("bnd,bmd->bnm", x, y) / (
            x.norm(dim=-1)[..., None] * y.norm(dim=-1)[:, None] + eps
        )
        if self.learn_temperature:
            T = self.T.abs() + 0.01
        else:
            T = torch.tensor(self.T, device=c.device)
        K = ((c - 1.0) / T).exp()
        return K


class GP(nn.Module):
    def __init__(
        self,
        kernel,
        T=1,
        learn_temperature=False,
        only_attention=False,
        gp_dim=64,
        basis="fourier",
        covar_size=5,
        only_nearest_neighbour=False,
        sigma_noise=0.1,
        no_cov=False,
        predict_features = False,
    ):
        super().__init__()
        self.K = kernel(T=T, learn_temperature=learn_temperature)
        self.sigma_noise = sigma_noise
        self.covar_size = covar_size
        self.pos_conv = torch.nn.Conv2d(2, gp_dim, 1, 1)
        self.only_attention = only_attention
        self.only_nearest_neighbour = only_nearest_neighbour
        self.basis = basis
        self.no_cov = no_cov
        self.dim = gp_dim
        self.predict_features = predict_features

    def get_local_cov(self, cov):
        K = self.covar_size
        b, h, w, h, w = cov.shape
        hw = h * w
        cov = F.pad(cov, 4 * (K // 2,))  # pad v_q
        delta = torch.stack(
            torch.meshgrid(
                torch.arange(-(K // 2), K // 2 + 1), torch.arange(-(K // 2), K // 2 + 1)
            ),
            dim=-1,
        )
        positions = torch.stack(
            torch.meshgrid(
                torch.arange(K // 2, h + K // 2), torch.arange(K // 2, w + K // 2)
            ),
            dim=-1,
        )
        neighbours = positions[:, :, None, None, :] + delta[None, :, :]
        points = torch.arange(hw)[:, None].expand(hw, K**2)
        local_cov = cov.reshape(b, hw, h + K - 1, w + K - 1)[
            :,
            points.flatten(),
            neighbours[..., 0].flatten(),
            neighbours[..., 1].flatten(),
        ].reshape(b, h, w, K**2)
        return local_cov

    def reshape(self, x):
        return rearrange(x, "b d h w -> b (h w) d")

    def project_to_basis(self, x):
        if self.basis == "fourier":
            return torch.cos(8 * math.pi * self.pos_conv(x))
        elif self.basis == "linear":
            return self.pos_conv(x)
        else:
            raise ValueError(
                "No other bases other than fourier and linear currently supported in public release"
            )

    def get_pos_enc(self, y):
        b, c, h, w = y.shape
        coarse_coords = torch.meshgrid(
            (
                torch.linspace(-1, 1, h, device=y.device),
                torch.linspace(-1, 1, w, device=y.device),
            )
        )

        coarse_coords = torch.stack((coarse_coords[1], coarse_coords[0]), dim=-1)[None]
        coarse_coords = rearrange(coarse_coords, "b h w d -> b d h w")
        coarse_embedded_coords = self.project_to_basis(coarse_coords)
        coarse_embedded_coords = repeat(coarse_embedded_coords, 'b d h w -> (b q) d h w', q=b)
        
        return coarse_embedded_coords

    def forward(self, x, y, n_track=None, n_query=None, eval=True, **kwargs):
        b1, c, h1, w1 = x.shape
        b2, c, h2, w2 = y.shape
        
        if n_track is not None:
            x_bsize = b1//n_track
            
        f = self.get_pos_enc(y)
        if self.predict_features:
            f = f + y[:,:self.dim] # Stupid way to predict features
        b, d, h2, w2 = f.shape
        #assert x.shape == y.shape
        x, y, f = self.reshape(x), self.reshape(y), self.reshape(f)
        # K_xx = self.K(x, x) #.view(x_bsize, n_track, 1, )
        
        K_yy = self.K(y, y)
        
        x = rearrange(x,  '(b1 t o) (h1 w1) c -> b1 t o c h1 w1', b1=x_bsize, t=n_track, o=1, h1=h1, w1=w1)
        x = repeat(x, 'b t o c h w -> b t (o q) c h w', q=n_query)
        x = rearrange(x, 'b t q c h w -> (b t q) (h w) c')
        
        K_xy = self.K(x, y)
        K_yx = K_xy.permute(0, 2, 1)
        sigma_noise = self.sigma_noise * torch.eye(h2 * w2, device=x.device)[None, :, :]
        
        # Due to https://github.com/pytorch/pytorch/issues/16963 annoying warnings, remove batch if N large
        # if len(K_yy[0]) > 2000:
        #     K_yy_inv = torch.cat([torch.linalg.inv(K_yy[k:k+1] + sigma_noise[k:k+1]) for k in range(b)])
        # else:
        K_yy_inv = torch.linalg.inv(K_yy + sigma_noise)

        mu_x = K_xy.matmul(K_yy_inv.matmul(f))
        mu_x = rearrange(mu_x, "b (h w) d -> b d h w", h=h1, w=w1)
        
        gp_feats = mu_x
        return gp_feats

class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, input_tensor):
        # x1, x2 = x  # high, low (old, new)
        # x = torch.cat([x1, x2], dim=1)
        x = self.global_pooling(input_tensor)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        res = x * input_tensor
        return res


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        
        if relu:
            return self.relu(x + res)
        else:
            return x + res



class GP_Decoder(nn.Module):
    def __init__(
        self,
        internal_dim,
        upsample_mode="bilinear",
        align_corners=False,
        run_deep=True,
        classify=True,
        classify_dim=7,
    ):
        super().__init__()
        self.internal_dim = internal_dim
        self.feat_input_modules =  nn.Conv2d(internal_dim, internal_dim, 1, 1)
        self.rrb_d = RRB(internal_dim*2, internal_dim)
        self.cab = CAB(internal_dim, internal_dim)
        self.rrb_u = RRB(internal_dim, internal_dim)
        if classify:
            self.terminal_module = nn.Conv2d(internal_dim, classify_dim**2+1, 1, 1, 0)
        else:
            self.terminal_module = nn.Conv2d(internal_dim, 2+1, 1, 1, 0)
            
        self.run_deep = run_deep
        self.classify=classify

    def scales(self):
        return self._scales.copy()

    def forward(self, embeddings, feats):
        feats = self.feat_input_modules(feats)
        embeddings = torch.cat([feats, embeddings], dim=1)
        embeddings = self.rrb_d(embeddings)
        
        if self.run_deep:
            embeddings = self.cab(embeddings)
            embeddings = self.rrb_u(embeddings)
            
        preds = self.terminal_module(embeddings)
        pred_coord = preds[:, :-1]
        pred_certainty = preds[:, -1:]
        return pred_coord, pred_certainty

if __name__ == "__main__":
    gp_dim = 256
         
    kernel_temperature = 0.2
    learn_temperature = False
    no_cov = True
    kernel = CosKernel
    only_attention = False
    basis = "fourier"
    gp32 = GP(
        kernel,
        T=kernel_temperature,
        learn_temperature=learn_temperature,
        only_attention=only_attention,
        gp_dim=gp_dim,
        basis=basis,
        no_cov=no_cov,
    )
    
    f1 = torch.randn([1, 10,30,40])
    f2 = torch.randn([5, 10,60,80])
    
    res = gp32(f1, f2, n_track=1, n_query=5)
    print(res.shape)