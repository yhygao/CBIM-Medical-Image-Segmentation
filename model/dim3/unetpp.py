import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, ConvNormAct
from .utils import get_block, get_norm


class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch, base_ch, scale, kernel_size, num_classes=1, block='SingleConv', norm='bn'):
        super().__init__()

        num_block = 2
        block = get_block(block)
        norm = get_norm(norm)

        n_ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*10]
    
        self.pool0 = nn.MaxPool3d(scale[0])
        self.up0 = nn.Upsample(scale_factor=tuple(scale[0]), mode='trilinear', align_corners=True)
        self.pool1 = nn.MaxPool3d(scale[1])
        self.up1 = nn.Upsample(scale_factor=tuple(scale[1]), mode='trilinear', align_corners=True)
        self.pool2 = nn.MaxPool3d(scale[2])
        self.up2 = nn.Upsample(scale_factor=tuple(scale[2]), mode='trilinear', align_corners=True)
        self.pool3 = nn.MaxPool3d(scale[3])
        self.up3 = nn.Upsample(scale_factor=tuple(scale[3]), mode='trilinear', align_corners=True)


        self.conv0_0 = self.make_layer(in_ch, n_ch[0], num_block, block, kernel_size=kernel_size[0], norm=norm)
        self.conv1_0 = self.make_layer(n_ch[0], n_ch[1], num_block, block, kernel_size=kernel_size[1], norm=norm)
        self.conv2_0 = self.make_layer(n_ch[1], n_ch[2], num_block, block, kernel_size=kernel_size[2], norm=norm)
        self.conv3_0 = self.make_layer(n_ch[2], n_ch[3], num_block, block, kernel_size=kernel_size[3], norm=norm)
        self.conv4_0 = self.make_layer(n_ch[3], n_ch[4], num_block, block, kernel_size=kernel_size[4], norm=norm)

        self.conv0_1 = self.make_layer(n_ch[0]+n_ch[1], n_ch[0], num_block, block, kernel_size=kernel_size[0], norm=norm)
        self.conv1_1 = self.make_layer(n_ch[1]+n_ch[2], n_ch[1], num_block, block, kernel_size=kernel_size[1], norm=norm)
        self.conv2_1 = self.make_layer(n_ch[2]+n_ch[3], n_ch[2], num_block, block, kernel_size=kernel_size[2], norm=norm)
        self.conv3_1 = self.make_layer(n_ch[3]+n_ch[4], n_ch[3], num_block, block, kernel_size=kernel_size[3], norm=norm)

        self.conv0_2 = self.make_layer(n_ch[0]*2+n_ch[1], n_ch[0], num_block, block, kernel_size=kernel_size[0], norm=norm)
        self.conv1_2 = self.make_layer(n_ch[1]*2+n_ch[2], n_ch[1], num_block, block, kernel_size=kernel_size[1], norm=norm)
        self.conv2_2 = self.make_layer(n_ch[2]*2+n_ch[3], n_ch[2], num_block, block, kernel_size=kernel_size[2], norm=norm)

        self.conv0_3 = self.make_layer(n_ch[0]*3+n_ch[1], n_ch[0], num_block, block, kernel_size=kernel_size[0], norm=norm)
        self.conv1_3 = self.make_layer(n_ch[1]*3+n_ch[2], n_ch[1], num_block, block, kernel_size=kernel_size[1], norm=norm)


        self.conv0_4 = self.make_layer(n_ch[0]*4+n_ch[1], n_ch[0], num_block, block, kernel_size=kernel_size[0], norm=norm)


        self.output = nn.Conv3d(n_ch[0], num_classes, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool1(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up1(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up0(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool2(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up2(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up0(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool3(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up3(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up2(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up1(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up0(x1_3)], 1))

        output = self.output(x0_4)

        return output


    def make_layer(self, in_ch, out_ch, num_block, block, kernel_size, norm):
        blocks = []
        blocks.append(block(in_ch, out_ch, kernel_size=kernel_size, norm=norm))

        for i in range(num_block-1):
            blocks.append(block(out_ch, out_ch, kernel_size=kernel_size, norm=norm))

        return nn.Sequential(*blocks)





