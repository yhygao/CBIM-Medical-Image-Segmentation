import numpy as np

def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

    else:
        if args.dataset == 'acdc':
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
        elif args.dataset == 'lits':
            from .dim3.dataset_lits import LiverDataset

            return LiverDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'bcv':
            from .dim3.dataset_bcv import BCVDataset

            return BCVDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'kits':
            from .dim3.dataset_kits import KidneyDataset

            return KidneyDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'amos_ct':
            from .dim3.dataset_amos_ct import AMOSDataset

            return AMOSDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'amos_mr':
            from .dim3.dataset_amos_mr import AMOSDataset

            return AMOSDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'msd_lung':
            from .dim3.dataset_msd_lung import LungDataset

            return LungDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
            



class DALIInputCallable(object):
    def __init__(self, dataset, bs, shard_id=0, num_shards=1):
        self.dataset = dataset
        self.batch_size = bs
        
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.shard_size = len(self.dataset) // num_shards
        self.shard_offset = self.shard_size * shard_id

        self.full_iterations = self.shard_size // self.batch_size

        self.perm = None
        self.last_seen_epoch = None

    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()
        
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=42 + sample_info.epoch_idx).permutation(len(self.dataset))
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        img, lab = self.dataset.getitem_dali(sample_idx)

        return img, lab
