# Modified from https://github.com/mattmacy/vnet.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):

    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


# normalization between sub-volumes is necessary
# for good performance

class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
    def forward(self, input):
        self._check_input_dim(input)
        
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    
    return nn.Sequential(*layers)



class InputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.relu1 = ELUCons(elu, outChans)
        self.inChans = inChans
        self.outChans = outChans

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # split input in to 16 channels
        num = int(self.outChans / self.inChans)
        x16 = x.repeat(1, num, 1, 1, 1)
        #x16 = torch.cat((x, x, x, x, x, x, x, x,
        #                 x, x, x, x, x, x, x, x), 0)

        out = self.relu1(torch.add(out, x16))

        return out

class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, scale=2, dropout=False):
        super(DownTransition, self).__init__()

        outChans = 2*inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=scale, stride=scale)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)

        if dropout:
            self.do1 = nn.Dropout3d()

        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))

        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, scale=2, dropout=False):
        super(UpTransition, self).__init__()

        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=scale, stride=scale)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)

        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):

        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))

        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans, elu, nll):
        super(OutputTransition, self).__init__()

        self.conv1 = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(outChans)
        self.conv2 = nn.Conv3d(outChans, outChans, kernel_size=1)
        self.relu1 = ELUCons(elu, outChans)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)

        return out


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, inChans, outChans, scale, baseChans=16, elu=True, nll=False):
        super(VNet, self).__init__()

        self.in_tr = InputTransition(inChans, baseChans, elu)
        self.down_tr32 = DownTransition(baseChans, 1, elu, scale=scale[0])
        self.down_tr64 = DownTransition(baseChans*2, 2, elu, scale=scale[1])
        self.down_tr128 = DownTransition(baseChans*4, 3, elu, dropout=True, scale=scale[2])
        self.down_tr256 = DownTransition(baseChans*8, 2, elu, dropout=True, scale=scale[3])

        self.up_tr256 = UpTransition(baseChans*16, baseChans*16, 2, elu, dropout=True, scale=scale[3])
        self.up_tr128 = UpTransition(baseChans*16, baseChans*8, 2, elu, dropout=True, scale=scale[2])
        self.up_tr64 = UpTransition(baseChans*8, baseChans*4, 1, elu, scale=scale[1])
        self.up_tr32 = UpTransition(baseChans*4, baseChans*2, 1, elu, scale=scale[0])
        self.out_tr = OutputTransition(baseChans*2, outChans, elu, nll)


    def forward(self, x):

        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)

        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)

        out = self.out_tr(out)

        return out
