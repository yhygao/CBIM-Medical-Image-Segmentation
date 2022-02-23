import torch
import torch.nn as nn
import torch.nn.functional as F
from inference.utils import get_inference
from metric.utils import calculate_distance, calculate_dice
import numpy as np
import pdb



def validation(net, dataloader, args):
    
    net.eval()

    dice_list = np.zeros(args.classes-1) # background is not including in validation
    ASD_list = np.zeros(args.classes-1)
    HD_list = np.zeros(args.classes-1)
    
    inference = get_inference(args)

    counter = 0
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
                
            
            print(i)

            tmp_ASD_list, tmp_HD_list = calculate_distance(label_pred, labels, spacing[0], args.classes)
            ASD_list += np.clip(np.nan_to_num(tmp_ASD_list, nan=500), 0, 500)
            HD_list += np.clip(np.nan_to_num(tmp_HD_list, nan=500), 0, 500)

            dice, _, _ = calculate_dice(label_pred.view(-1, 1), labels.view(-1, 1), args.classes)

            dice_list += dice.cpu().numpy()[1:]

            counter += 1

    dice_list /= counter
    ASD_list /= counter
    HD_list /= counter

    return dice_list, ASD_list, HD_list
