import builtins
import logging
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from model.utils import get_model
from training.dataset.utils import get_dataset
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler


from training.utils import update_ema_variables
from training.losses import DiceLoss
from training.validation import validation_ddp as validation
from training.utils import (
    exp_lr_scheduler_with_warmup, 
    log_evaluation_result, 
    get_optimizer, 
    filter_validation_results
)
import yaml
import argparse
import time
import math
import sys
import pdb
import warnings
import matplotlib.pyplot as plt
import copy

from utils import (
    configure_logger,
    save_configure,
    is_master,
    AverageMeter,
    ProgressMeter,
)
warnings.filterwarnings("ignore", category=UserWarning)




def train_net(net, trainset, testset, args, ema_net=None, fold_idx=0):
    
    ########################################################################################
    # Dataset Creation
    

    trainLoader = data.DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    
    
    test_sampler = DistributedSampler(testset) if args.distributed else None
    testLoader = data.DataLoader(
        testset,
        batch_size=1,  # has to be 1 sample per gpu, as the input size of 3D input is different
        shuffle=(test_sampler is None), 
        sampler=test_sampler,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    logging.info(f"Created Dataset and DataLoader")

    ########################################################################################
    # Initialize tensorboard, optimizer, amp scaler and etc.
    writer = SummaryWriter(f"{args.log_path}{args.unique_name}/fold_{fold_idx}") if is_master(args) else None

    optimizer = get_optimizer(args, net)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(args.weight).cuda())
    criterion_dl = DiceLoss()
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    ########################################################################################
    # Start training
    best_Dice = np.zeros(args.classes)
    best_HD = np.ones(args.classes) * 1000
    best_ASD = np.ones(args.classes) * 1000
    
    for epoch in range(args.epochs):

        logging.info(f"Starting epoch {epoch+1}/{args.epochs}")
        exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=args.base_lr, epoch=epoch, warmup_epoch=5, max_epoch=args.epochs)
        logging.info(f"Current lr: {exp_scheduler:.4e}")
       
        train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args)
        
        ##################################################################################
        # Evaluation, save checkpoint and log training info
        net_for_eval = ema_net if args.ema else net
        if (epoch+1) % args.val_freq == 0:

            dice_list_test, ASD_list_test, HD_list_test = validation(net_for_eval, testLoader, args)
            if is_master(args):
                dice_list_test, ASD_list_test, HD_list_test = filter_validation_results(dice_list_test, ASD_list_test, HD_list_test, args) # filter results for some dataset, e.g. amos_mr
                log_evaluation_result(writer, dice_list_test, ASD_list_test, HD_list_test, 'test', epoch, args)
            
                if dice_list_test.mean() >= best_Dice.mean():
                    best_Dice = dice_list_test
                    best_HD = HD_list_test
                    best_ASD = ASD_list_test

                    torch.save(net_for_eval.module.state_dict(), f"{args.cp_path}{args.dataset}/{args.unique_name}/fold_{fold_idx}_best.pth")

                logging.info("Evaluation Done")
                logging.info(f"Dice: {dice_list_test.mean():.4f}/Best Dice: {best_Dice.mean():.4f}")

                writer.add_scalar('LR', exp_scheduler, epoch+1)
    
    return best_Dice, best_HD, best_ASD


def train_epoch(trainLoader, net, ema_net, optimizer, epoch, writer, criterion, criterion_dl, scaler, args):
    batch_time = AverageMeter("Time", ":6.2f")
    epoch_loss = AverageMeter("Loss", ":.2f")
    progress = ProgressMeter(
        len(trainLoader) if args.dimension=='2d' else args.iter_per_epoch, 
        [batch_time, epoch_loss], 
        prefix="Epoch: [{}]".format(epoch+1),
    )
    
    net.train()

    tic = time.time()
    iter_num_per_epoch = 0
    for i, (img, label) in enumerate(trainLoader):
        
        '''
        # uncomment this for visualize the input images and labels for debug
        for idx in range(img.shape[0]):
            plt.subplot(1,2,1)
            plt.imshow(img[idx, 0, 40, :, :].numpy())
            plt.subplot(1,2,2)
            plt.imshow(label[idx, 0, 40, :, :].numpy())

            plt.show()
        '''
        img = img.cuda(args.proc_idx, non_blocking=True)
        label = label.cuda(args.proc_idx, non_blocking=True).long()
        step = i + epoch * len(trainLoader) # global steps
        
        optimizer.zero_grad()

        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = net(img)
                
                loss = 0

                if isinstance(result, tuple) or isinstance(result, list):
                    # If use deep supervision, add all loss together 
                    for j in range(len(result)):
                        loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
                else:
                    loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            result = net(img)
            
            loss = 0
            if isinstance(result, tuple) or isinstance(result, list):
                # If use deep supervision, add all loss together 
                for j in range(len(result)):
                    loss += args.aux_weight[j] * (criterion(result[j], label.squeeze(1)) + criterion_dl(result[j], label))
            else:
                loss = criterion(result, label.squeeze(1)) + criterion_dl(result, label)


            loss.backward()
            optimizer.step()
        if args.ema:
            update_ema_variables(net, ema_net, args.ema_alpha, step)

        epoch_loss.update(loss.item(), img.shape[0])
        batch_time.update(time.time() - tic)
        tic = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if args.dimension == '3d':
            iter_num_per_epoch += 1
            if iter_num_per_epoch > args.iter_per_epoch:
                break

        #torch.cuda.empty_cache()

        if is_master(args):
            writer.add_scalar('Train/Loss', epoch_loss.avg, epoch+1)


    


def get_parser():
    parser = argparse.ArgumentParser(description='CBIM Meidcal Image Segmentation')
    parser.add_argument('--dataset', type=str, default='acdc', help='dataset name')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--dimension', type=str, default='2d', help='2d model or 3d model')
    parser.add_argument('--pretrain', action='store_true', help='if use pretrained weight for init')
    parser.add_argument('--amp', action='store_true', help='if use the automatic mixed precision for faster training')

    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--load', type=str, default=False, help='load pretrained model')
    parser.add_argument('--cp_path', type=str, default='./exp/', help='the path to save checkpoint and logging info')
    parser.add_argument('--log_path', type=str, default='./log/', help='the path to save tensorboard log')
    parser.add_argument('--unique_name', type=str, default='test', help='unique experiment name')
    
    parser.add_argument('--gpu', type=str, default='0,1,2,3')

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
        logging.info(f"Model loaded from {args.load}")

    if args.ema:
        ema_net = get_model(args, pretrain=args.pretrain)
        logging.info("Use EMA model for evaluation")
    else:
        ema_net = None

    return net, ema_net 





def main_worker(proc_idx, ngpus_per_node, fold_idx, args, result_dict=None, trainset=None, testset=None):
    # seed each process
    if args.reproduce_seed is not None:
        random.seed(args.reproduce_seed)
        np.random.seed(args.reproduce_seed)
        torch.manual_seed(args.reproduce_seed)

        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # set process specific info
    args.proc_idx = proc_idx
    args.ngpus_per_node = ngpus_per_node

    # suppress printing if not master
    if args.multiprocessing_distributed and args.proc_idx != 0:
        def print_pass(*args, **kwargs):
            pass

        builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + proc_idx
        
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=f"{args.dist_url}",
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.cuda.set_device(args.proc_idx)

        # adjust data settings according to multi-processing
        args.batch_size = int(args.batch_size / args.ngpus_per_node)
        args.workers = int((args.num_workers + args.ngpus_per_node - 1) / args.ngpus_per_node)


    args.cp_dir = f"{args.cp_path}/{args.dataset}/{args.unique_name}"
    os.makedirs(args.cp_dir, exist_ok=True)
    configure_logger(args.rank, args.cp_dir+f"/fold_{fold_idx}.txt")
    save_configure(args)

    logging.info(
        f"\nDataset: {args.dataset},\n"
        + f"Model: {args.model},\n"
        + f"Dimension: {args.dimension}"
    )
    

    net, ema_net = init_network(args)
    
    net.to('cuda')
    if args.ema:
        ema_net.to('cuda')
    if args.distributed:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net = DistributedDataParallel(net, device_ids=[args.proc_idx], find_unused_parameters=True)
        # set find_unused_parameters to True if some of the parameters is not used in forward
        
        if args.ema:
            ema_net = nn.SyncBatchNorm.convert_sync_batchnorm(ema_net)
            ema_net = DistributedDataParallel(ema_net, device_ids=[args.proc_idx], find_unused_parameters=False)
            
            for p in ema_net.parameters():
                p.requires_grad_(False)


    logging.info(f"Created Model")
    best_Dice, best_HD, best_ASD = train_net(net, trainset, testset, args, ema_net, fold_idx=fold_idx)
    
    logging.info(f"Training and evaluation on Fold {fold_idx} is done")
    
    if args.distributed:
        if is_master(args):
            # collect results from the master process
            result_dict['best_Dice'] = best_Dice
            result_dict['best_HD'] = best_HD
            result_dict['best_ASD'] = best_ASD
    else:
        return best_Dice, best_HD, best_ASD
        

        



if __name__ == '__main__':
    #mp.set_start_method('fork')
    #mp.set_sharing_strategy('file_system')
    
    # parse the arguments
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.log_path = args.log_path + '%s/'%args.dataset

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed


    ngpus_per_node = torch.cuda.device_count()
    
    
    Dice_list, HD_list, ASD_list = [], [], []
    for fold_idx in range(args.k_fold):
        if args.multiprocessing_distributed:
            with mp.Manager() as manager:
            # use the Manager to gather results from the processes
                result_dict = manager.dict()
                    
                # Since we have ngpus_per_node processes per node, the total world_size
                # needs to be adjusted accordingly
                args.world_size = ngpus_per_node * args.world_size
                trainset = get_dataset(args, mode='train', fold_idx=fold_idx)
                testset = get_dataset(args, mode='test', fold_idx=fold_idx)
                # Use torch.multiprocessing.spawn to launch distributed processes:
                # the main_worker process function
                mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, fold_idx, args, result_dict, trainset, testset))
                best_Dice = result_dict['best_Dice']
                best_HD = result_dict['best_HD']
                best_ASD = result_dict['best_ASD']
            args.world_size = 1
        else:
            trainset = get_dataset(args, mode='train', fold_idx=fold_idx)
            testset = get_dataset(args, mode='test', fold_idx=fold_idx)
            # Simply call main_worker function
            best_Dice, best_HD, best_ASD = main_worker(0, ngpus_per_node, fold_idx, args, trainset=trainset, testset=testset)



        Dice_list.append(best_Dice)
        HD_list.append(best_HD)
        ASD_list.append(best_ASD)
    
    #############################################################################################
    # Save the cross validation results
    total_Dice = np.vstack(Dice_list)
    total_HD = np.vstack(HD_list)
    total_ASD = np.vstack(ASD_list)
    

    with open(f"{args.cp_path}/{args.dataset}/{args.unique_name}/cross_validation.txt",  'w') as f:
        np.set_printoptions(precision=4, suppress=True) 
        f.write('Dice\n')
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {Dice_list[i]}\n")
        f.write(f"Each Class Dice Avg: {np.mean(total_Dice, axis=0)}\n")
        f.write(f"Each Class Dice Std: {np.std(total_Dice, axis=0)}\n")
        f.write(f"All classes Dice Avg: {total_Dice.mean()}\n")
        f.write(f"All classes Dice Std: {np.mean(total_Dice, axis=1).std()}\n")

        f.write("\n")

        f.write("HD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {HD_list[i]}\n")
        f.write(f"Each Class HD Avg: {np.mean(total_HD, axis=0)}\n")
        f.write(f"Each Class HD Std: {np.std(total_HD, axis=0)}\n")
        f.write(f"All classes HD Avg: {total_HD.mean()}\n")
        f.write(f"All classes HD Std: {np.mean(total_HD, axis=1).std()}\n")

        f.write("\n")

        f.write("ASD\n")
        for i in range(args.k_fold):
            f.write(f"Fold {i}: {ASD_list[i]}\n")
        f.write(f"Each Class ASD Avg: {np.mean(total_ASD, axis=0)}\n")
        f.write(f"Each Class ASD Std: {np.std(total_ASD, axis=0)}\n")
        f.write(f"All classes ASD Avg: {total_ASD.mean()}\n")
        f.write(f"All classes ASD Std: {np.mean(total_ASD, axis=1).std()}\n")



        
    print(f'All {args.k_fold} folds done.')

    sys.exit(0)

