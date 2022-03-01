import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from training.utils import update_ema_variables
from training.losses import DiceLoss
from training.validation import validation
from training.utils import exp_lr_scheduler_with_warmup, log_evaluation_result, get_optimizer
import yaml
import argparse
import time
import math
import os
import sys
import pdb
import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)




def train_net(net, args, ema_net=None, fold_idx=0):
    
    data_path = args.data_root
    
    trainset = get_dataset(args, mode='train', fold_idx=fold_idx)
    trainLoader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = get_dataset(args, mode='test', fold_idx=fold_idx)
    testLoader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    
    writer = SummaryWriter(args.log_path + args.unique_name + '_%d'%fold_idx)

    optimizer = get_optimizer(args, net)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda())
    criterion_dl = DiceLoss()


    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000

    
    iter_count = 0
    for epoch in range(args.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, args.epochs))
        epoch_loss = 0

        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        #exp_scheduler = 1e-4
        
        print('current lr:', exp_scheduler)
        
        tic = time.time()
        iter_num_per_epoch = 0
        for i, (img, label) in enumerate(trainLoader, 0):
            
            '''
            for idx in range(img.shape[0]):
                plt.subplot(1,2,1)
                plt.imshow(img[idx, 0, 40, :, :].numpy())
                plt.subplot(1,2,2)
                plt.imshow(label[idx, 0, 40, :, :].numpy())

                plt.show()
            '''
            img = img.cuda()
            label = label.cuda()

            net.train()
            
            training_tic = time.time()
            optimizer.zero_grad()
            
            
            result = net(img)
            
            loss = 0
            
            if isinstance(result, tuple) or isinstance(result, list):
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)


            loss.backward()
            optimizer.step()
            iter_count += 1
            if args.ema:
                update_ema_variables(net, ema_net, args.ema_alpha, iter_count)

            epoch_loss += loss.item()
            batch_time = time.time() - tic
            training_time = time.time() - training_tic
            tic = time.time()
            print('%d batch loss: %.5f, batch_time:%.5f, training_time: %.5f'%(i, loss.item(), batch_time, training_time))
            
            if args.dimension == '3d':
                iter_num_per_epoch += 1
                if iter_num_per_epoch > args.iter_per_epoch:
                    break

        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i+1)))
        torch.cuda.empty_cache()

        writer.add_scalar('Train/Loss', epoch_loss/(i+1), epoch+1)
        writer.add_scalar('LR', exp_scheduler, epoch+1)

        
        if not os.path.isdir('%s%s'%(args.cp_path, args.dataset)):
            os.mkdir('%s%s'%(args.cp_path, args.dataset))

        if not os.path.isdir('%s%s/%s/'%(args.cp_path, args.dataset, args.unique_name)):
            os.mkdir('%s%s/%s/'%(args.cp_path, args.dataset, args.unique_name))
        
        if args.ema:
            net_for_eval = ema_net
        else:
            net_for_eval = net

        
        if (epoch+1) % args.val_frequency == 0:

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args)
            log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
            if dice_list_test.mean() >= best_Dice.mean():
                best_Dice = dice_list_test
                best_HD = HD_list_test
                best_ASD = ASD_list_test

                torch.save(net_for_eval.state_dict(), '%s%s/%s/%d_best.pth'%(args.cp_path, args.dataset, args.unique_name, fold_idx))

            print('save done')
            print('dice: %.5f/best dice: %.5f'%(dice_list_test.mean(), best_Dice.mean()))
    
    return best_Dice, best_HD, best_ASD




def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Conv-Trans Segmentation')
    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./checkpoint/', help='checkpoint path')
    parser.add_argument('--log_path', type=str, default='./log/', help='log path')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    config_path = 'config/%s/%s_%s.yaml'%(args.dataset, args.model, args.dimension)
    if not os.path.exists(config_path):
        raise ValueError("The specified configuration doesn't exist: %s"%config_path)

    print('Loading configurations from %s'%config_path)

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for key, value in config.items():
        setattr(args, key, value)

    return args
    


def init_network(args):
    net = get_model(args, pretrain=args.pretrain)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        for p in ema_net.parameters():
            p.requires_grad_(False)
    else:
        ema_net = None
    return net, ema_net 


if __name__ == '__main__':
    
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_path = args.log_path + '%s/'%args.dataset
   
    Dice_list = []
    HD_list = []
    ASD_list = []

    for i in range(args.k_fold):
    #for i in range(4, 5):
        net, ema_net = init_network(args)

        print(net)
        
        param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(param_num)
        
        net.cuda()
        if args.ema:
            ema_net.cuda()
        
        best_Dice, best_HD, best_ASD = train_net(net, args, ema_net, fold_idx=i)

        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)

    
    if not os.path.exists('exp/exp_%s'%args.dataset):
        os.mkdir('exp/exp_%s'%args.dataset)

    with open('exp/exp_%s/%s.txt'%(args.dataset, args.unique_name), 'w') as f:

        f.write('Dice   HD  ASD\n')
        for i in range(args.k_fold):
            f.write(str(Dice_list[i]) + str(HD_list[i]) + str(ASD_list[i]) + '\n')

        
        total_Dice = np.vstack(Dice_list)
        total_HD = np.vstack(HD_list)
        total_ASD = np.vstack(ASD_list)
    
        
        f.write('avg Dice:' + str(np.mean(total_Dice, axis=0)) + ' std Dice:' + str(np.std(total_Dice, axis=0)) + ' mean:' + str(total_Dice.mean()) + ' std:' + str(np.mean(total_Dice, axis=1).std()) +  '\n')
        f.write('avg HD:' + str(np.mean(total_HD, axis=0)) + ' std HD:' + str(np.std(total_HD, axis=0)) + ' mean:' + str(total_HD.mean()) + ' std:' + str(np.mean(total_HD, axis=1).std()) + '\n')
        f.write('avg ASD:' + str(np.mean(total_ASD, axis=0)) + ' std ASD:' + str(np.std(total_ASD, axis=0)) + ' mean:' + str(total_ASD.mean()) + ' std:' + str(np.mean(total_ASD, axis=1).std()) + '\n')


        
    print('done')

    sys.exit(0)
