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
            



