

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



class DALIInputCallable(object):
    def __init__(self, dataset, bs):
        self.dataset = dataset
        self.batch_size = bs

        self.full_iterations = len(self.dataset) // self.batch_size

    def __call__(self, sample_info):
        sample_idx = sample_info.idx_in_epoch
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration()

        img, lab = self.dataset.getitem_dali(sample_idx)

        return img, lab
