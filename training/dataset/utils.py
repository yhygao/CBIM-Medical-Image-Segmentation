


def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.seed)
        elif args.dataset == 'synapse':
            from .dim2.dataset_synapse import SynapseDataset

            return SynapseDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.seed)


    else:
        if args.dataset == 'acdc':
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode)

        elif args.dataset == 'synapse':
            from .dim3.dataset_synapse import SynapseDataset
            
            return SynapseDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.seed)
        

