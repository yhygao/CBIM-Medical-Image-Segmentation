import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, SingleConv

class AttentionBlock(nn.Module):
    def __init__(self, g_ch, l_ch, int_ch):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, int_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int_ch)
            )
        self.W_x = nn.Sequential(
            nn.Conv2d(l_ch, int_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(int_ch)
            )
        self.psi = nn.Sequential(
            nn.Conv2d(int_ch, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x): 
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1) 
        psi = self.psi(psi)

        return x * psi 


class attention_up_block(nn.Module):
    def __init__(self, in_ch, out_ch, num_block, block=BasicBlock, norm=nn.BatchNorm2d):
        super().__init__()
        
        self.attn = AttentionBlock(in_ch, out_ch, out_ch//2)

        block_list = []
        block_list.append(block(in_ch+out_ch, out_ch))

        for i in range(num_block-1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)

        x2 = self.attn(x1, x2)

        out = torch.cat([x2, x1], dim=1)

        out = self.conv(out)

        return out

