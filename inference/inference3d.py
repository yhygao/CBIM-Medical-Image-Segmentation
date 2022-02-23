import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import split_idx
import pdb


def inference_whole_image(net, img, args=None):
    '''
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W

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
    img: torch tensor, B, C, D, H, W
    return: prob (after softmax), B, classes, D, H, W

    The overlap of two windows will be half the window size

    Use this function to inference if out-of-memory occurs when whole image inferencing
    Better to be consistent with the training window size
    '''
    net.eval()

    B, C, D, H, W = img.shape

    win_d, win_h, win_w = args.window_size

    flag = False
    if D < win_d or H < win_h or W < win_w:
        flag = True
        diff_D = max(0, win_d-D)
        diff_H = max(0, win_h-H)
        diff_W = max(0, win_w-W)

        img = F.pad(img, (0, diff_W, 0, diff_H, 0, diff_D))
        
        origin_D, origin_H, origin_W = D, H, W
        B, C, D, H, W = img.shape


    half_win_d = win_d // 2
    half_win_h = win_h // 2
    half_win_w = win_w // 2

    pred_output = torch.zeros((B, args.classes, D, H, W)).to(img.device)

    counter = torch.zeros((B, 1, D, H, W)).to(img.device)
    one_count = torch.ones((B, 1, win_d, win_h, win_w)).to(img.device)

    with torch.no_grad():
        for i in range(D // half_win_d):
            for j in range(H // half_win_h):
                for k in range(W // half_win_w):
                    
                    d_start_idx, d_end_idx = split_idx(half_win_d, D, i)
                    h_start_idx, h_end_idx = split_idx(half_win_h, H, j)
                    w_start_idx, w_end_idx = split_idx(half_win_w, W, k)

                    input_tensor = img[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx]
                    
                    pred = net(input_tensor)

                    if isinstance(pred, tuple) or isinstance(pred, list):
                        pred = pred[0]

                    pred = F.softmax(pred, dim=1)

                    pred_output[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += pred

                    counter[:, :, d_start_idx:d_end_idx, h_start_idx:h_end_idx, w_start_idx:w_end_idx] += one_count

    pred_output /= counter
    if flag:
        pred_output = pred_output[:, :, :origin_D, :origin_H, :origin_W]

    return pred_output

                    





