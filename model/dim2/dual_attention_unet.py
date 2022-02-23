import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_block, get_norm
from .unet_utils import inconv, down_block, up_block
from .dual_attention_utils import DAHead

class DAUNet(nn.Module):

    def __init__(self, in_ch, num_classes, base_ch=32, block='BasicBlock', pool=True):
        super().__init__()
        
        block = get_block(block)
        nb = 2 # num_block

        self.inc = inconv(in_ch, base_ch, block=block)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=nb, block=block,  pool=pool)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=nb, block=block, pool=pool)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=nb, block=block, pool=pool)
        self.down4 = down_block(8*base_ch, 16*base_ch, num_block=nb, block=block, pool=pool)

        self.DAModule = DAHead(16*base_ch, num_classes)

        self.up1 = up_block(16*base_ch, 8*base_ch, num_block=nb, block=block)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=nb, block=block)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=nb, block=block)
        self.up4 = up_block(2*base_ch, base_ch, num_block=nb, block=block)

        self.outc = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x): 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        feat_fuse, sasc_pred, sa_pred, sc_pred = self.DAModule(x5)

        out = self.up1(feat_fuse, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2) 
        out = self.up4(out, x1) 
        out = self.outc(out)

        return out 
