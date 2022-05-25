import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, MBConv, FusedMBConv, ConvNeXtBlock

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, block=BasicBlock):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)

        self.conv2 = block(out_ch, out_ch)

    def forward(self, x): 
        out = self.conv1(x)
        out = self.conv2(out)

        return out 


class down_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, pool=True):
        super().__init__()

        block_list = []


        if pool:
            block_list.append(nn.MaxPool2d(2))
            block_list.append(block(in_ch, out_ch))
        else:
            block_list.append(block(in_ch, out_ch, stride=2))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch, stride=1))

        self.conv = nn.Sequential(*block_list)
    def forward(self, x): 
        return self.conv(x)

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock):
        super().__init__()

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2*out_ch, out_ch))


        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


