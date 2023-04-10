import os
import logging
import torch
import torch.distributed as dist
import pdb

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logger(rank, log_path=None):
    if log_path:
        log_dir = os.path.dirname(log_path)
        os.makedirs(log_dir, exist_ok=True)

    # only master process will print & write
    level = logging.INFO if rank in {-1, 0} else logging.WARNING
    handlers = [logging.StreamHandler()]
    if rank in {0, -1} and log_path:
        handlers.append(logging.FileHandler(log_path, "w"))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
        force=True,
    )


def save_configure(args):
    if hasattr(args, "distributed"):
        if (args.distributed and is_master(args)) or (not args.distributed):
            with open(f"{args.cp_dir}/config.txt", 'w') as f:
                for name in args.__dict__:
                    f.write(f"{name}: {getattr(args, name)}\n")
    else:
        with open(f"{args.cp_dir}/config.txt", 'w') as f:
            for name in args.__dict__:
                f.write(f"{name}: {getattr(args, name)}\n")
            
def resume_load_optimizer_checkpoint(optimizer, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def resume_load_model_checkpoint(net, ema_net, args):
    assert args.load != False, "Please specify the load path with --load"
    
    checkpoint = torch.load(args.load)
    net.load_state_dict(checkpoint['model_state_dict'])
    args.start_epoch = checkpoint['epoch']

    if args.ema:
        ema_net.load_state_dict(checkpoint['ema_model_state_dict'])



class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)



class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1)) 
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]" 


def is_master(args):
    return args.rank % args.ngpus_per_node == 0
