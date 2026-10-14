import torch
import torch.nn as nn
import torch.nn.functional as F

from .roma_layers import Block
from .roma_layers import MemEffAttention
import math
# from .dinov2 import vit_large


def cls_to_flow_refine(cls, return_certainty=False):
    B,C,H,W = cls.shape
    device = cls.device
    res = round(math.sqrt(C))
    G = torch.meshgrid(*[torch.linspace(-1, 1, steps = res, device = device) for _ in range(2)])
    G = torch.stack([G[1],G[0]],dim=-1).reshape(C,2)
    cls = cls.softmax(dim=1)
    mode = cls.max(dim=1).indices
    index = torch.stack((mode-1, mode, mode+1, mode - res, mode + res), dim = 1).clamp(0,C - 1).long()
    neighbours = torch.gather(cls, dim = 1, index = index)[...,None]
    flow = neighbours[:,0] * G[index[:,0]] + neighbours[:,1] * G[index[:,1]] + neighbours[:,2] * G[index[:,2]] + neighbours[:,3] * G[index[:,3]] + neighbours[:,4] * G[index[:,4]]
    tot_prob = neighbours.sum(dim=1)
    flow = flow / tot_prob
    return flow


class TransformerDecoder(nn.Module):
    def __init__(self, blocks, hidden_dim, out_dim, is_classifier = False, *args, 
                 amp = False, pos_enc = True, learned_embeddings = False, embedding_dim = None, amp_dtype = torch.float16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = blocks
        self.to_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self._scales = [16]
        self.is_classifier = is_classifier
        self.amp = amp
        self.amp_dtype = amp_dtype
        self.pos_enc = pos_enc
        self.learned_embeddings = learned_embeddings
        if self.learned_embeddings:
            self.learned_pos_embeddings = nn.Parameter(nn.init.kaiming_normal_(torch.empty((1, hidden_dim, embedding_dim, embedding_dim))))

    def scales(self):
        return self._scales.copy()

    def forward(self, gp_posterior, features, old_stuff, new_scale):
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp):
            B,C,H,W = gp_posterior.shape
            x = torch.cat((gp_posterior, features), dim = 1)
            B,C,H,W = x.shape
            grid = get_grid(B, H, W, x.device).reshape(B,H*W,2)
            if self.learned_embeddings:
                pos_enc = F.interpolate(self.learned_pos_embeddings, size = (H,W), mode = 'bilinear', align_corners = False).permute(0,2,3,1).reshape(1,H*W,C)
            else:
                pos_enc = 0
            tokens = x.reshape(B,C,H*W).permute(0,2,1) + pos_enc
            z = self.blocks(tokens)
            out = self.to_out(z)
            out = out.permute(0,2,1).reshape(B, self.out_dim, H, W)
            warp, certainty = out[:, :-1], out[:, -1:]
            return warp, certainty
        
def get_grid(b, h, w, device):
    grid = torch.meshgrid(
        *[
            torch.linspace(-1, 1, n, device=device)
            for n in (b, h, w)
        ]
    )
    grid = torch.stack((grid[2], grid[1]), dim=-1).reshape(b, h, w, 2)
    return grid


if __name__ == "__main__":
    gp_dim = 128
    feat_dim = 128
    decoder_dim = gp_dim + feat_dim
    cls_to_coord_res = 7 # select_size * N?
    
    coordinate_decoder = TransformerDecoder(
        blocks=nn.Sequential(*[Block(decoder_dim, 8, attn_class=MemEffAttention) for _ in range(5)]), 
        hidden_dim=decoder_dim, 
        out_dim=cls_to_coord_res**2 + 1,
        is_classifier=True,
        amp = True,
        pos_enc = False,)

    a = coordinate_decoder(torch.randn([1,128,10,10]),torch.randn([1,128,10,10]),None, None)
    flow = cls_to_flow_refine(
                        a[0],
                    ).permute(0,3,1,2)
    print(a[0].shape)
    print(flow.shape)
    print(flow)
    """
    gm_warp_or_cls, certainty, old_stuff = self.embedding_decoder(
                    gp_posterior, f1_s, old_stuff, new_scale
                )
                
    flow = cls_to_flow_refine(
                        gm_warp_or_cls,
                    ).permute(0,3,1,2)
    """