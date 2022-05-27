import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice
import numpy as np
import pdb



def validation(net, dataloader, args):
    
    net.eval()

    dice_list = []
    ASD_list = []
    HD_list = []
    for i in range(args.classes-1): # background is not including in validation
        dice_list.append([])
        ASD_list.append([])
        HD_list.append([])

    inference = get_inference(args)

    with torch.no_grad():
        for i, (images, labels, spacing) in enumerate(dataloader):
            # spacing here is used for distance metrics calculation
            
            inputs, labels = images.float().cuda(), labels.long().cuda()
            
            if args.dimension == '2d':
                inputs = inputs.permute(1, 0, 2, 3)
            
            pred = inference(net, inputs, args)

            _, label_pred = torch.max(pred, dim=1)
            
            if args.dimension == '2d':
                labels = labels.squeeze(0)
            else:
                label_pred = label_pred.squeeze(0)
                labels = labels.squeeze(0).squeeze(0)
                

            #tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            tmp_ASD_list = np.zeros(args.classes-1)
            tmp_HD_list = np.zeros(args.classes-1)

            tmp_ASD_list =  np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            tmp_HD_list = np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)
            
            # exclude background
            dice = dice.cpu().numpy()[1:]

            labels = labels.cpu().numpy()

            for cls in range(0, args.classes-1):
                if cls+1 in np.unique(labels): 
                    # in case some classes are missing in the GT
                    # only classes appear in the GT are used for evaluation
                    ASD_list[cls].append(tmp_ASD_list[cls])
                    HD_list[cls].append(tmp_HD_list[cls])
                    dice_list[cls].append(dice[cls])
            print(i, 'Dice:', dice)

    out_dice = []
    out_ASD = []
    out_HD = []
    for cls in range(0, args.classes-1):
        out_dice.append(np.array(dice_list[cls]).mean())
        out_ASD.append(np.array(ASD_list[cls]).mean())
        out_HD.append(np.array(HD_list[cls]).mean())

    return np.array(out_dice), np.array(out_ASD), np.array(out_HD)
