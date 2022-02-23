import numpy as np
import torch
import torch.nn as nn
import math


class DAHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DAHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv_a = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
        )
        self.conv_c = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False)
        )

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)


        self.conv_a_1 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1)
        )


        self.conv_c_1 = nn.Sequential(
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, in_channels, 1)
        )

        self.conv_a_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, n_classes, 1)
        )

        self.conv_c_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, n_classes, 1)
        )

        self.fuse_out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels, n_classes, 1)
        )


    def forward(self, x):
        sa_feat = self.conv_a(x)
        sa_feat = self.sa(sa_feat)
        sa_feat = self.conv_a_1(sa_feat)


        sc_feat = self.conv_c(x)
        sc_feat = self.sc(sc_feat)
        sc_feat = self.conv_c_1(sc_feat)

        feat_fusion = sa_feat + sc_feat

        sa_out = self.conv_a_out(sa_feat)
        sc_out = self.conv_c_out(sc_feat)
        sasc_out = self.fuse_out(feat_fusion)


        return feat_fusion, sasc_out, sa_out, sc_out


class PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim, reduction=8):
        super(PAM_Module, self).__init__()

        self.chanel_in = in_dim
        self.reduction = reduction

        self.query_conv = nn.Conv2d(in_dim, in_dim//self.reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//self.reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x: input feature maps (B * C * H * W)
        returns:
            out: attention value + input feature

        """

        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs:
                x: input feature maps (B * C * H * W)
            returns:
                out: attention value + input feature
        """
        m_batchsize, C, height, width = x.shape
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        #energy_new = torch.max(energy, -1, keepdim=True)[0]
        #energy_new = energy_new.expand_as(energy)
        #energy_new -= energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out









if __name__ == '__main__':

    import pdb
    cam = CAM_Module(in_dim=24)
    pdb.set_trace()
    arr = torch.randn((8, 24, 120, 120))
    out = cam(arr)
