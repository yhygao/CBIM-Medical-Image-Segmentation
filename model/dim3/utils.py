import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv_layers import BasicBlock, Bottleneck, SingleConv
from .trans_layers import LayerNorm

def get_block(name):
    block_map = {
        'SingleConv': SingleConv,
        'BasicBlock': BasicBlock,
        'Bottleneck': Bottleneck,
    }
    return block_map[name]

def get_norm(name):
    norm_map = {'bn': nn.BatchNorm3d,
                'in': nn.InstanceNorm3d,
                'ln': LayerNorm
                }

    return norm_map[name]

def get_act(name):
    act_map = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU,
        'gelu': nn.GELU,
        'swish': nn.SiLU
    }
    return act_map[name]
