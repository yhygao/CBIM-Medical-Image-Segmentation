import numpy as np
import torch
import torch.nn as nn
import pdb

def get_model(args, pretrain=False):
    
    if args.dimension == '2d':
        if args.model == 'unet':
            from .dim2 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        if args.model == 'unet++':
            from .dim2 import UNetPlusPlus
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNetPlusPlus(args.in_chan, args.classes, args.base_chan)
        if args.model == 'attention_unet':
            from .dim2 import AttentionUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return AttentionUNet(args.in_chan, args.classes, args.base_chan)

        elif args.model == 'resunet':
            from .dim2 import UNet 
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.classes, args.base_chan, block=args.block)
        elif args.model == 'daunet':
            from .dim2 import DAUNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return DAUNet(args.in_chan, args.classes, args.base_chan, block=args.block)

        elif args.model in ['medformer']:
            from .dim2 import MedFormer
            if pretrain:
                raise ValueError('No pretrain model available')
            return MedFormer(args.in_chan, args.classes, args.base_chan, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, map_size=args.map_size, proj_type=args.proj_type, act=nn.ReLU, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, aux_loss=args.aux_loss)


        elif args.model == 'transunet':
            from .dim2 import VisionTransformer as ViT_seg
            from .dim2.transunet import CONFIGS as CONFIGS_ViT_seg
            config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
            config_vit.n_classes = args.classes
            config_vit.n_skip = 3
            config_vit.patches.grid = (int(args.training_size[0]/16), int(args.training_size[1]/16))
            net = ViT_seg(config_vit, img_size=args.training_size[0], num_classes=args.classes)

            if pretrain:
                net.load_from(weights=np.load(args.init_model))

            return net
        
        elif args.model == 'swinunet':
            from .dim2 import SwinUnet
            from .dim2.swin_unet import SwinUnet_config
            config = SwinUnet_config()
            net = SwinUnet(config, img_size=224, num_classes=args.classes)
            
            if pretrain:
                net.load_from(args.init_model)

            return net



    elif args.dimension == '3d':
        if args.model == 'vnet':
            from .dim3 import VNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return VNet(args.in_chan, args.classes, scale=args.downsample_scale, baseChans=args.base_chan)
        elif args.model == 'resunet':
            from .dim3 import UNet
            if pretrain:
                raise ValueError('No pretrain model available')
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'unet':
            from .dim3 import UNet
            return UNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'unet++':
            from .dim3 import UNetPlusPlus
            return UNetPlusPlus(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)
        elif args.model == 'attention_unet':
            from .dim3 import AttentionUNet
            return AttentionUNet(args.in_chan, args.base_chan, num_classes=args.classes, scale=args.down_scale, norm=args.norm, kernel_size=args.kernel_size, block=args.block)

        elif args.model == 'medformer':
            from .dim3 import MedFormer

            return MedFormer(args.in_chan, args.classes, args.base_chan, map_size=args.map_size, conv_block=args.conv_block, conv_num=args.conv_num, trans_num=args.trans_num, num_heads=args.num_heads, fusion_depth=args.fusion_depth, fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, expansion=args.expansion, attn_drop=args.attn_drop, proj_drop=args.proj_drop, proj_type=args.proj_type, norm=args.norm, act=args.act, kernel_size=args.kernel_size, scale=args.down_scale, aux_loss=args.aux_loss)
    
        elif args.model == 'unetr':
            from .dim3 import UNETR
            model = UNETR(args.in_chan, args.classes, args.training_size, feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, pos_embed='perceptron', norm_name='instance', res_block=True)
            
            return model
        elif args.model == 'vtunet':
            from .dim3 import VTUNet
            model = VTUNet(args, args.classes)

            if pretrain:
                model.load_from(args)
            return model
        elif args.model == 'swin_unetr':
            from .dim3 import SwinUNETR
            model = SwinUNETR(args.window_size, args.in_chan, args.classes, feature_size=args.base_chan)

            if args.pretrain:
                weights = torch.load('/research/cbim/vast/yg397/ConvFormer/ConvFormer/initmodel/model_swinvit.pt')
                model.load_from(weights=weights)

            return model
        elif args.model == 'nnformer':
            from .dim3 import nnFormer
            model = nnFormer(args.window_size, input_channels=args.in_chan, num_classes=args.classes, deep_supervision=args.aux_loss)

            return model
    else:
        raise ValueError('Invalid dimension, should be \'2d\' or \'3d\'')

