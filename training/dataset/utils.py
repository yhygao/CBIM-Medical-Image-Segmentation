


def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.seed)

    else:
        if args.dataset == 'acdc':
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode)


