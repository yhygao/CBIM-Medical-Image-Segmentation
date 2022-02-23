import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block
from .utils import get_block, get_norm
from .attention_unet_utils import attention_up_block

class AttentionUNet(nn.Module):
    def __init__(self, in_ch, num_classes, base_ch=32, block='SingleConv', pool=True):
        super().__init__()

        num_block = 2 
        block = get_block(block)

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool)
        self.down4 = down_block(8*base_ch, 16*base_ch, num_block=num_block, block=block, pool=pool)

        self.up1 = attention_up_block(16*base_ch, 8*base_ch, num_block=num_block, block=block)
        self.up2 = attention_up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block)
        self.up3 = attention_up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block)
        self.up4 = attention_up_block(2*base_ch, base_ch, num_block=num_block, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x): 

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4)
        out = self.up2(out, x3)
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)

        return out




