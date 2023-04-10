import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct, DepthwiseSeparableConv, MBConv, FusedMBConv
from .trans_layers import TransformerBlock, LayerNorm

from einops import rearrange
import pdb


class BidirectionAttention(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0.,
                    map_size=[8,8,8], proj_type='depthwose', kernel_size=[3,3,3], no_map_out=False):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.map_size = map_size
        
        assert proj_type in ['linear', 'depthwise']

        if proj_type == 'linear':
            self.feat_qv = nn.Conv3d(feat_dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
            self.feat_out = nn.Conv3d(self.inner_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.feat_qv = DepthwiseSeparableConv(feat_dim, self.inner_dim*2, kernel_size=kernel_size)
            self.feat_out = DepthwiseSeparableConv(self.inner_dim, out_dim, kernel_size=kernel_size)

        
        self.map_qv = nn.Conv3d(map_dim, self.inner_dim*2, kernel_size=1, stride=1, padding=0, bias=False)
        if no_map_out:
            self.map_out = nn.Identity()
        else:
            self.map_out = nn.Conv3d(self.inner_dim, map_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def rearrange1(self, x, heads):
        # substitue for rearrange as it is not supported by pytorch2.0 torch.compile
        # b (dim_head heads) d h w -> b heads (d h w) dim_head
        b, l, d, h, w = x.shape
        dim_head = int(l / heads)
        x = x.view(b, dim_head, heads, -1).contiguous() # b dim_head heads (dhw)
        x = x.permute(0, 2, 3, 1).contiguous() # b heads (dhw) dim_head

        return x
    def rearrange2(self, x, d, h, w):
        # substitue for rearrange as it is not supported by pytorch2.0 torch.compile
        # 'b heads (d h w) dim_head -> b (dim_head heads) d h w'
        b, heads, l, dim_head = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() # b, dim_head, heads, l
        x = x.view(b, (heads*dim_head), d, h, w).contiguous()

        return x


        
    def forward(self, feat, semantic_map):

        B, C, D, H, W = feat.shape

        feat_q, feat_v = self.feat_qv(feat).chunk(2, dim=1) # B, inner_dim, D, H, W
        map_q, map_v = self.map_qv(semantic_map).chunk(2, dim=1) # B, inner_dim, ms, ms, ms

        #feat_q, feat_v = map(lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=D, h=H, w=W, b=B), [feat_q, feat_v])
        feat_q, feat_v = map(lambda t: self.rearrange1(t, self.heads), [feat_q, feat_v])

        #map_q, map_v = map(lambda t: rearrange(t, 'b (dim_head heads) d h w -> b heads (d h w) dim_head', dim_head=self.dim_head, heads=self.heads, d=self.map_size[0], h=self.map_size[1], w = self.map_size[2], b=B), [map_q, map_v])
        map_q, map_v = map(lambda t: self.rearrange1(t, self.heads), [map_q, map_v])


        attn = torch.einsum('bhid,bhjd->bhij', feat_q, map_q)
        attn *= self.scale
        
        feat_map_attn = F.softmax(attn, dim=-1)  # semantic map is very concise that don't need dropout, 
                                                 # add dropout might cause unstable during training
        map_feat_attn = self.attn_drop(F.softmax(attn, dim=-2))

        feat_out = torch.einsum('bhij,bhjd->bhid', feat_map_attn, map_v)
        #feat_out = rearrange(feat_out, 'b heads (d h w) dim_head -> b (dim_head heads) d h w', d=D, h=H, w=W)
        feat_out = self.rearrange2(feat_out, d=D, h=H, w=W)


        map_out = torch.einsum('bhji,bhjd->bhid', map_feat_attn, feat_v)
        #map_out = rearrange(map_out, 'b heads (d h w) dim_head -> b (dim_head heads) d h w', d=self.map_size[0], h=self.map_size[1], w=self.map_size[2])
        map_out = self.rearrange2(map_out, d=self.map_size[0], h=self.map_size[1], w=self.map_size[2])


        feat_out = self.proj_drop(self.feat_out(feat_out))
        map_out = self.map_out(map_out)

        return feat_out, map_out




class BidirectionAttentionBlock(nn.Module):
    def __init__(self, feat_dim, map_dim, out_dim, heads, dim_head, norm=nn.BatchNorm3d, act=nn.ReLU,
                expansion=4, attn_drop=0., proj_drop=0., map_size=[8, 8, 8], 
                proj_type='depthwise', kernel_size=[3,3,3], no_map_out=False):
        super().__init__()

        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]
        assert proj_type in ['linear', 'depthwise']

        self.norm1 = norm(feat_dim) if norm else nn.Identity() # norm layer for feature map
        self.norm2 = norm(map_dim) if norm else nn.Identity() # norm layer for semantic map
        
        self.attn = BidirectionAttention(feat_dim, map_dim, out_dim, heads, dim_head, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, kernel_size=kernel_size, no_map_out=no_map_out)

        self.shortcut = nn.Sequential()
        if feat_dim != out_dim:
            self.shortcut = ConvNormAct(feat_dim, out_dim, 1, padding=0, norm=norm, act=act, preact=True)

        if proj_type == 'linear':
            self.feedforward = FusedMBConv(out_dim, out_dim, expansion=expansion, kernel_size=1, act=act, norm=norm)
        else:
            self.feedforward = MBConv(out_dim, out_dim, expansion=expansion, kernel_size=kernel_size, act=act, norm=norm)

    def forward(self, x, semantic_map):
        
        feat = self.norm1(x)
        mapp = self.norm2(semantic_map)

        out, mapp = self.attn(feat, mapp)

        out += self.shortcut(x)
        out = self.feedforward(out)

        mapp += semantic_map

        return out, mapp

class PatchMerging(nn.Module):
    """
    Modified patch merging layer that works as down-sampling
    """
    def __init__(self, dim, out_dim, norm=nn.BatchNorm3d, proj_type='linear', down_scale=[2,2,2], kernel_size=[3,3,3]):
        super().__init__()
        self.dim = dim
        assert proj_type in ['linear', 'depthwise']

        self.down_scale = down_scale

        merged_dim = 2 ** down_scale.count(2) * dim

        if proj_type == 'linear':
            self.reduction = nn.Conv3d(merged_dim, out_dim, kernel_size=1, bias=False)
        else:
            self.reduction = DepthwiseSeparableConv(merged_dim, out_dim, kernel_size=kernel_size)

        self.norm = norm(merged_dim)

    def forward(self, x):
        """
        x: B, C, D, H, W
        """
        merged_x = []
        for i in range(self.down_scale[0]):
            for j in range(self.down_scale[1]):
                for k in range(self.down_scale[2]):
                    tmp_x = x[:, :, i::self.down_scale[0], j::self.down_scale[1], k::self.down_scale[2]]
                    merged_x.append(tmp_x)
        
        x = torch.cat(merged_x, 1)
        x = self.norm(x)
        x = self.reduction(x)

        return x

class BasicLayer(nn.Module):
    """
    A basic transformer layer for one stage
    No downsample or upsample operation in this layer, they are wrapped in the down_block of up_block
    """

    def __init__(self, feat_dim, map_dim, out_dim, num_blocks, heads=4, dim_head=64, expansion=4, attn_drop=0., proj_drop=0., map_size=[8,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, act=nn.GELU, kernel_size=[3,3,3], no_map_out=False):
        super().__init__()

        dim1 = feat_dim
        dim2 = out_dim

        self.blocks = nn.ModuleList([])
        for i in range(num_blocks):
            no_map_out_args = False if i != (num_blocks-1) else no_map_out
            self.blocks.append(BidirectionAttentionBlock(dim1, map_dim, dim2, heads, dim_head, expansion=expansion, attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, norm=norm, act=act, kernel_size=kernel_size, no_map_out=no_map_out_args))
            dim1 = out_dim

    def forward(self, x, semantic_map):
        for block in self.blocks:
            x, semantic_map = block(x, semantic_map)

        
        return x, semantic_map


class SemanticMapGeneration(nn.Module):
    def __init__(self, feat_dim, map_dim, map_size):
        super().__init__()

        self.map_size = map_size
        self.map_dim = map_dim

        self.map_code_num = map_size[0] * map_size[1] * map_size[2]

        self.base_proj = nn.Conv3d(feat_dim, map_dim, kernel_size=3, padding=1, bias=False)

        self.semantic_proj = nn.Conv3d(feat_dim, self.map_code_num, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape

        feat = self.base_proj(x) #B, map_dim, d, h, w
        weight_map = self.semantic_proj(x) # B, map_code_num, d, h, w

        weight_map = weight_map.view(B, self.map_code_num, -1)
        weight_map = F.softmax(weight_map, dim=2) #B, map_code_num, dhw)
        feat = feat.view(B, self.map_dim, -1) # B, map_dim, dhw
        
        semantic_map = torch.einsum('bij,bkj->bik', feat, weight_map)

        return semantic_map.view(B, self.map_dim, self.map_size[0], self.map_size[1], self.map_size[2])


class SemanticMapFusion(nn.Module):
    def __init__(self, in_dim_list, dim, heads, depth=1, norm=nn.BatchNorm3d, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim

        # project all maps to the same channel num
        self.in_proj = nn.ModuleList([])
        for i in range(len(in_dim_list)):
            self.in_proj.append(nn.Conv3d(in_dim_list[i], dim, kernel_size=1, bias=False))

        self.fusion = TransformerBlock(dim, depth, heads, dim//heads, dim, attn_drop=attn_drop, proj_drop=proj_drop)

        # project all maps back to their origin channel num
        self.out_proj = nn.ModuleList([])
        for i in range(len(in_dim_list)):
            self.out_proj.append(nn.Conv3d(dim, in_dim_list[i], kernel_size=1, bias=False))

    def forward(self, map_list):
        B, _, D, H, W = map_list[0].shape
        proj_maps = [self.in_proj[i](map_list[i]).view(B, self.dim, -1).permute(0, 2, 1) for i in range(len(map_list))]
        #B, L, C where L=DHW

        proj_maps = torch.cat(proj_maps, dim=1)
        attned_maps = self.fusion(proj_maps)

        attned_maps = attned_maps.chunk(len(map_list), dim=1)

        maps_out = [self.out_proj[i](attned_maps[i].permute(0, 2, 1).view(B, self.dim, D, H, W)) for i in range(len(map_list))]

        return maps_out
################################################################

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d, act=nn.GELU):
        super().__init__()

        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)

        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm, act=act)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out



class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, down_scale=[2,2,2], kernel_size=[3,3,3],
                conv_block=BasicBlock, heads=4, dim_head=64, expansion=1, attn_drop=0., 
                proj_drop=0., map_size=[8,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, 
                act=nn.GELU, map_generate=False, map_dim=None):
        super().__init__()
        

        map_dim = out_ch if map_dim is None else map_dim
        self.map_generate = map_generate
        if map_generate:
            self.map_gen = SemanticMapGeneration(out_ch, map_dim, map_size)

        self.patch_merging = PatchMerging(in_ch, out_ch, norm=norm, proj_type=proj_type, down_scale=down_scale, kernel_size=kernel_size)

        block_list = []
        for i in range(conv_num):
            block_list.append(conv_block(out_ch, out_ch, norm=norm, act=act, kernel_size=kernel_size))
        self.conv_blocks = nn.Sequential(*block_list)

        self.trans_blocks = BasicLayer(out_ch, map_dim, out_ch, num_blocks=trans_num, heads=heads, \
                dim_head=dim_head, norm=norm, act=act, expansion=expansion, attn_drop=attn_drop, \
                proj_drop=proj_drop, map_size=map_size, proj_type=proj_type, kernel_size=kernel_size)

    def forward(self, x):
        x = self.patch_merging(x)

        out = self.conv_blocks(x)

        if self.map_generate:
            semantic_map = self.map_gen(out)
        else:
            semantic_map = None

        out, semantic_map = self.trans_blocks(out, semantic_map)
            

        return out, semantic_map

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, conv_num, trans_num, up_scale=[2,2,2], kernel_size=[3,3,3], 
                conv_block=BasicBlock, heads=4, dim_head=64, expansion=4, attn_drop=0., proj_drop=0.,
                map_size=[4,8,8], proj_type='depthwise', norm=nn.BatchNorm3d, act=nn.GELU, 
                map_dim=None, map_shortcut=False, no_map_out=False):
        super().__init__()


        self.map_shortcut = map_shortcut
        map_dim = out_ch if map_dim is None else map_dim
        if map_shortcut:
            self.map_reduction = nn.Conv3d(in_ch+out_ch, map_dim, kernel_size=1, bias=False)
        else:
            self.map_reduction = nn.Identity() #nn.Conv3d(in_ch, map_dim, kernel_size=1, bias=False)

        self.trans_blocks = BasicLayer(in_ch+out_ch, map_dim, out_ch, num_blocks=trans_num, \

                    heads=heads, dim_head=dim_head, norm=norm, act=act, expansion=expansion, \
                    attn_drop=attn_drop, proj_drop=proj_drop, map_size=map_size,\
                    proj_type=proj_type, kernel_size=kernel_size, no_map_out=no_map_out)

        if trans_num == 0:
            dim1 = in_ch+out_ch
        else:
            dim1 = out_ch

        conv_list = []
        for i in range(conv_num):
            conv_list.append(conv_block(dim1, out_ch, kernel_size=kernel_size, norm=norm, act=act))
            dim1 = out_ch
        self.conv_blocks = nn.Sequential(*conv_list)

    def forward(self, x1, x2, map1, map2=None):
        # x1: low-res feature, x2: high-res feature shortcut from encoder
        # map1: semantic map from previous low-res layer
        # map2: semantic map from encoder shortcut, might be none if we don't have the map from encoder
        
        x1 = F.interpolate(x1, size=x2.shape[-3:], mode='trilinear', align_corners=True)
        feat = torch.cat([x1, x2], dim=1)
        
        if self.map_shortcut and map2 is not None:
            semantic_map = torch.cat([map1, map2], dim=1)
            semantic_map = self.map_reduction(semantic_map)
        else:
            semantic_map = map1
        

        out, semantic_map = self.trans_blocks(feat, semantic_map)
        out = self.conv_blocks(out)

        return out, semantic_map
       

