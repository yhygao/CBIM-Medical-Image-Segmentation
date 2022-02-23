import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def get_optimizer(args, net):
    if args.optimizer == 'sgd':
        return optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        return optim.Adam(net.parameters(), lr=args.base_lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        return optim.AdamW(net.parameters(), lr=args.base_lr, betas=args.betas, weight_decay=args.weight_decay)


def log_evaluation_result(writer, dice_list, ASD_list, HD_list, name, epoch, args):
    
    writer.add_scalar('Dice/%s_AVG'%name, dice_list.mean(), epoch+1)
    for idx in range(args.classes-1):
        writer.add_scalar('Dice/%s_Dice%d'%(name, idx+1), dice_list[idx], epoch+1)
    writer.add_scalar('ASD/%s_AVG'%name, ASD_list.mean(), epoch+1)
    for idx in range(args.classes-1):
        writer.add_scalar('ASD/%s_ASD%d'%(name, idx+1), ASD_list[idx], epoch+1)
    writer.add_scalar('HD/%s_AVG'%name, HD_list.mean(), epoch+1)
    for idx in range(args.classes-1):
        writer.add_scalar('HD/%s_HD%d'%(name, idx+1), HD_list[idx], epoch+1)

def multistep_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, lr_decay_epoch, max_epoch, gamma=0.1):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    flag = False
    for i in range(len(lr_decay_epoch)):
        if epoch == lr_decay_epoch[i]:
            flag = True
            break

    if flag == True:
        lr = init_lr * gamma**(i+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    else:
        return optimizer.param_groups[0]['lr']

    return lr

def exp_lr_scheduler_with_warmup(optimizer, init_lr, epoch, warmup_epoch, max_epoch):

    if epoch >= 0 and epoch <= warmup_epoch:
        lr = init_lr * 2.718 ** (10*(float(epoch) / float(warmup_epoch) - 1.))
        if epoch == warmup_epoch:
            lr = init_lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    else:
        lr = init_lr * (1 - epoch / max_epoch)**0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return lr




def update_ema_variables(model, ema_model, alpha, global_step):
    
    alpha = min((1 - 1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    for ema_buffer, m_buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(m_buffer)
