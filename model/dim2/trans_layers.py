import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb


__all__ = [
    'Mlp',
    'Attention',
    'TransformerBlock',
]

class Mlp(nn.Module):
    def __init__(self, in_dim, hid_dim=None, out_dim=None, act=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hid_dim = hid_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.act = act()
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)



class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x: B, L, C.   Batch, sequence length, dim
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')

        attned = self.to_out(attned)

        return attned


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.layers = nn.ModuleList([])

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, attn_drop, proj_drop)),
                PreNorm(dim, Mlp(dim, mlp_dim, dim, drop=proj_drop))
                ]))
    def forward(self, x):
        
        for attn, ffn in self.layers:
            x = attn(x) + x
            x = ffn(x) + x

        return x

        
