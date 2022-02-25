## CBIM-Medical-Image-Segmentation

This repo is a PyTorch-based framework for medical image segmentation, whose goal is to provide an easy-to-use framework for academic researchers to develop and 
evaluate deep learning models. It provides fair evaluation and comparison of CNNs and Transformers on multiple medical image datasets. 

### Features:
- Cover the whole process of model design, including dataset processing, model definition, model configuration, training and evaluation.
- Provide SOTA models as baseline for comparison. Model definition, training and evaluation code are simple with no complex code encapsulation.
- Provide models, losses, metrics, and etc. for 2D, 3D, multiple modalities and multiple tasks.


### Supportting models
- [UTNetV2](https://arxiv.org/abs/2107.00781) (Official implementation)
- UNet. Including 2D, 3D with different building block, e.g. double conv, Residual BasicBlock, Bottleneck, MBConv, or ConvNeXt block.
- [UNet++](https://arxiv.org/abs/1807.10165)
- [Attention UNet](https://arxiv.org/abs/1804.03999)
- [Dual Attention](https://arxiv.org/abs/1809.02983)
- [TransUNet](https://arxiv.org/abs/2102.04306)
- [SwinUNet](https://arxiv.org/abs/2105.05537)
- [UNETR](https://arxiv.org/abs/2103.10504)
- [VT-UNet](https://arxiv.org/pdf/2111.13300.pdf)
- More models are comming soon ... 
