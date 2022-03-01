import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct
import pdb 

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], block=BasicBlock, norm=nn.BatchNorm3d):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        pad_size = [i//2 for i in kernel_size]
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=pad_size, bias=False)
        self.conv2 = block(out_ch, out_ch, kernel_size=kernel_size, norm=norm)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], down_scale=[2,2,2], pool=True, norm=nn.BatchNorm3d):
        super().__init__() 
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(down_scale, int):
            down_scale = [down_scale] * 3

        block_list = []

        if pool:
            block_list.append(nn.MaxPool3d(down_scale))
            block_list.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))
        else:
            block_list.append(block(in_ch, out_ch, stride=down_scale, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x):
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, kernel_size=[3,3,3], up_scale=[2,2,2], norm=nn.BatchNorm3d):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 3
        if isinstance(up_scale, int):
            up_scale = [up_scale] * 3

        self.up_scale = up_scale


        block_list = []

        block_list.append(block(in_ch+out_ch, out_ch, kernel_size=kernel_size, norm=norm))
        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=True)

        out = torch.cat([x2, x1], dim=1)

        out = self.conv(out)

        return out



