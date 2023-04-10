import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pdb


__all__ = [
    'Mlp',
    'Attention',
    'TransformerBlock',
    'LayerNorm',
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

    def rearrange1(self, x, heads):
        # rearrange is not supported by pytorch2.0 torch.compile
        # 'b l (heads dim_head) -> b heads l dim_head'
        b, l, n = x.shape
        dim_head = int(n / heads)
        x = x.view(b, l, heads, dim_head).contiguous()
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def rearrange2(self, x):
         # 'b heads l dim_head -> b l (dim_head heads)')
         b, heads, l, dim_head = x.shape
         x = x.permute(0, 2, 1, 3).contiguous()
         x = x.view(b, l, -1).contiguous()

         return x


    def forward(self, x):
        # x: B, L, C.   Batch, sequence length, dim
        # 'b l (heads dim_head) -> b heads l dim_head',
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        
        #q, k, v = map(lambda t: rearrange(t, 'b l (heads dim_head) -> b heads l dim_head', heads=self.heads), [q, k, v])
        q, k, v = map(lambda t: self.rearrange1(t, heads=self.heads), [q, k, v])


        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = F.softmax(attn, dim=-1)

        attned = torch.einsum('bhij,bhjd->bhid', attn, v)
        #attned = rearrange(attned, 'b heads l dim_head -> b l (dim_head heads)')
        attned = self.rearrange2(attned)

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

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-5, data_format="channels_first"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):

        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[None, :, None, None, None] * x + self.bias[None, :, None, None, None]
            return x

       
