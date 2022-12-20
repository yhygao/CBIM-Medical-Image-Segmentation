## CBIM-Medical-Image-Segmentation

This repo is a PyTorch-based framework for medical image segmentation, whose goal is to provide an easy-to-use framework for academic researchers to develop and 
evaluate deep learning models. It provides fair evaluation and comparison of CNNs and Transformers on multiple medical image datasets. 

### News
We've released a new version paper of MedFormer, including more experiments and analysis, see in [MedFormer](https://arxiv.org/abs/2203.00131)
Detail updates can be found in docs/change.md

- **Support PyTorch DDP and AMP training. (Dec. 19. 2022)**
- **Add a tutorial on the usage of this repo: docs/tutorial.md (Still updating, Dec. 19. 2022)**
- Supports for BCV and LiTS dataset have been updated.

### Features
- Cover the whole process of model design, including dataset processing, model definition, model configuration, training and evaluation.
- Provide SOTA models as baseline for comparison. Model definition, training and evaluation code are simple with no complex code encapsulation.
- Provide models, losses, metrics, augmentation and etc. for 2D, 3D data, multiple modalities and multiple tasks.
- Optimized training techniques for SOTA performance.


### Supporting models
- [MedFormer](https://arxiv.org/abs/2203.00131) (Official implementation)
- UNet. Including 2D, 3D with different building block, e.g. double conv, Residual BasicBlock, Bottleneck, MBConv, or ConvNeXt block.
- [UNet++](https://arxiv.org/abs/1807.10165)
- [Attention UNet](https://arxiv.org/abs/1804.03999)
- [Dual Attention](https://arxiv.org/abs/1809.02983)
- [TransUNet](https://arxiv.org/abs/2102.04306)
- [SwinUNet](https://arxiv.org/abs/2105.05537)
- [UNETR](https://arxiv.org/abs/2103.10504)
- [VT-UNet](https://arxiv.org/pdf/2111.13300.pdf)
- More models are comming soon ... 

### Supporting Datasets
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) Cardiac MRI
- [BCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) Abdomen Organ CT
- [LiTS](https://competitions.codalab.org/competitions/17094) Liver and Tumor CT
- More datasets are comming soon


### Usage
We provide flexible usage. If you just want to use the models in your own framework, you can directly find the corresponding models in the `model/` folder and use them in your own framework. For the definition of specific models, please refer to `model/utils.py` in the `get_model` function. The models we provide do not have complex dependencies and encapsulation. The modules used by each model are defined in its own `xxx_utils.py` file. For example, the model definition of UNet only depend on `unet.py`, `unet_utils.py` and `conv_layers.py`.

If you want to use our framework, please follow below steps.

#### Install requirements
Create a new virtual environment and install all dependencies by:
```
pip install -r requirement.txt
```
#### Data preparation
Download the origin dataset from their corresponding official website.

Enter the `dataset_conversion` fold and find the dataset you want to use and the corresponding dimension (2d or 3d)

Edit the `src_path` and `tgt_path` the in `xxxdataset.py`, where the `src_path` is the path to the origin dataset, and `tgt_path` is the target path to store the processed dataset.

Then, `python xxxdataset.py`

After processing is finished, put the processed dataset into `dataset/` folder or use a soft link.

#### Configuration
Enter `config/xxxdataset/` and find the model and dimension (2d or 3d) you want to use. The training details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, etc., can be altered here. You can try your own congiguration or use the default configure, the one we used in the MedFormer paper, which should have a decent performance. The only thing to care is the `data_root`, make sure it points to the processed dataset directory.

#### Training
After the data, model and training strategy are configured, we can start training. Several arguments can be parsed in command line. You need to specify the model, the dimension, the dataset, whether use pretrain weights, batch size, and the experiment name. Our code will find the corresponding configuration for training. Here is an example to train 3D MedFormer on acdc.

```
python train.py --model medformer --dimension 3d --dataset acdc --batch_size 3 --unique_name acdc_3d_medformer --gpu 0
```

The training process will be logged by tensorboard, including training loss, testing performance, and etc. You can find it in the `log/unique_name` folder and use it to monitor the training process. After the training is done, the result of cross-validation will be stored in the `exp/exp_unique_name` folder.


### To Do
Add multi-GPU parallelism support.

Add MSD dataset support.

We'll continously maintain this repo to add more SOTA models, and add more dataset support. 

Performance comparison results of the supported models and dataset

Hope this repo can serves as a solid baseline for the future medical imaging model design.

### Citation
If you find this repo helps, please kindly cite our paper, thanks!
```
@inproceedings{gao2021utnet,
  title={UTNet: a hybrid transformer architecture for medical image segmentation},
  author={Gao, Yunhe and Zhou, Mu and Metaxas, Dimitris N},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={61--71},
  year={2021},
  organization={Springer}
}

@article{gao2022data,
  title={A data-scalable transformer for medical image segmentation: architecture, model efficiency, and benchmark},
  author={Gao, Yunhe and Zhou, Mu and Liu, Di and Yan, Zhennan and Zhang, Shaoting and Metaxas, Dimitris N},
  journal={arXiv preprint arXiv:2203.00131},
  year={2022}
}

```
