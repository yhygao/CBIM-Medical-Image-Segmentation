import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb


def inference_whole_image(net, img, args=None):
    '''
    img: torch tensor, B, C, H, W
    return: prob (after softmax), B, classes, H, W

    Use this function to inference if whole image can be put into GPU without memory issue
    Better to be consistent with the training window size
    '''
    net.eval()
    
    with torch.no_grad():
        pred = net(img)

        if isinstance(pred, tuple) or isinstance(pred, list):
            pred = pred[0]

    return F.softmax(pred, dim=1)


def inference_sliding_window(net, img, args):
    ''' 
    img: torch tensor, B, C, H, W
    return: prob (after softmax), B, classes, H, W

    The overlap of two windows will be half the window size
    Use this function to inference if out-of-memory occurs when whole image inferecing
    Better to be consistent with the training window size
    '''
    net.eval()

    B, C, H, W = img.shape
    
    win_h, win_w = args.window_size

    half_win_h = win_h // 2
    half_win_w = win_w // 2
    
    pred_output = torch.zeros((B, args.classes, H, W)).to(img.device)

    counter = torch.zeros((B, 1, H, W)).to(img.device)
    one_count = torch.ones((B, 1, win_h, win_w)).to(img.device)

    with torch.no_grad():
        for i in range(H // half_win_h):
            for j in range(W // half_win_w):
                
                h_start_idx, h_end_idx = split_idx(half_win_h, H, i)
                w_start_idx, w_end_idx = split_idx(half_win_w, W, j)
                
                input_tensor = img[:, :, h_start_idx:h_end_idx, w_start_idx:w_end_idx]

                pred = net(input_tensor)

                if isinstance(pred, tuple) or isinstance(pred, list):
                    pred = pred[0]

                pred = F.softmax(pred, dim=1)

                pred_output[:, :, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred
                counter[:, :, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

    pred_output /= counter

    return pred_output





