## Getting started


### Introduction

Welcome to the `CBIM-Medical-Image-Segmentation`, whose goal is to provide an easy-to-use framework for academic researchers to develop and evaluate deep learning models. It serves as a benchmark for fair evaluation and comparison of CNNs and Transformers on multiple public medical image datasets.


### Motivation

In the vision community, many benchmarks, e.g. ImageNet, CoCo, ADE20K, Cityscapes, enable fair and standardized evaluation pipelines. Researchers can directly find results of the baseline methods' results from the benchmarks for comparison. In contrast, medical image segmentation tasks don't have such standardized benchnarks due to the diversity and complexity of medical imaging. Researchers usually have to develop specific algorithms for diseases or datasets and reproduce the baseline models on their own. The motivation of this repository is to provide a framework for academic researchers covering the whole process of algorithm development, including data preprocessing, model definition, data augmentation, model training, and model evaluation. A variaty of representative CNN and Transformer segmentation models are provided as baselines. We also provide the support of the entire pipeline of these models on some commonly used public datasets. Users can reproduce the SOTA performance of the baselines on these datasets with only a few lines of commands. The simple, flexible and modular API makes it easy for researchers to adapt our framework for their own tasks without much efforts in debugging the over-encapsulated framework. **The ultimate goal of this repository is to enable researchers in medical image analysis to reproduce the SOTA baselines easily, design and develop models quickly, and evaluate and compair algorithms fairly without bias.**

### Differece with other frameworks

There are some well-designed frameworks for medical image analysis, such as nnUNet and MONAI. The following is the differece between our design ideas and theirs. nnUNet is an excellent framework. After tons of experiments on diverse dataests and challenges, nnUNet found that the design of pipeline and training strategy has great impact on the performance. So nnUNet designed rules for automatically making a training plan for a new task. Our repo refers many features from nnUNet. For example, we usually use similar data preprocessing strategy on the public medical datasets, e.g. data resampling, normalization and etc. The difference is that we believe that after selecting a good training strategy, the model design also has a great impact on performance. Therefore, we also provide many CNN and Transformer models as baselines. Users can easily reproduce SOTA performance with those baselines on the public datasets. 

MONAI is a framework for researchers and industry, providing a whole AI workflow including annotating, building, training, deploying and optimizing. In contrast, we only target AI researchers and provide concise, flexible, and modular design including data preprocessing, model training, and model evaluation functions. As we don't optimized for industry and deployment, our code is more flat and has no complicated encapsulation. Users can master every step of model development without spending a lot of time looking for dependent functions inside the framework for debugging when problems occur.
Such features make our framework more suitable for researchers to adapt our framework to their own tasks and develop models with more control.


### Getting Started

The tutorial will be two parts. In the first part, I'll show how to use *CBIM-Medical-Image-Segmentation* on the supported public datasets. In the second part, I'll show how to adapt our framework for your own tasks.


### Usage

Here we show how to use our framework to train and evaluate the provided models on the supported public datasets. First `clone` the repo.

#### Install requirements

Create a new virtual environment and install all dependencies by:

`pip install -r requirements.txt`

#### Data preparation

Download the origin dataset from their corresponding official website.

Enter the `dataset_conversion/` folder and find the dataset you want to use and the corresponding dimension (2d or 3d) you want to use.

Edit the `src_path` and `tgt_path` in the `xxxdataset.py`, where the `src_path` is the path to the downloaded original dataset, and `tgt_path` is the target path to store the processed dataset.

Then, `python xxxdataset.py`

After the processing is finished, put the processed dataset into the `dataset/` folder or use a soft link.

#### Training and evaluation configuration

Enter the `config/xxxdataset/` folder and find the model and dimension (2d or 3d) you want to use. The training and evaluation details, e.g. model hyper-parameters, training epochs, learning rate, optimizer, data augmentation, cross validation and etc., can be altered here. You can try your own configuration or use the default one, which we used in the MedFormer paper. The only thing to care is the `data_root`, make sure it points to the processed dataset path. The config of distributed training is also in this `.yaml` file. If you train the model using one gpu, then you can ignore the DDP section. If you want use multi-gpu training, please set the `dist_url` and set `multiprocessing_distributed` to `true`. You can also set the `reproduce_seed` to some specific seeds for reproducing issues.

#### Start training

We can start training after the data and configuration is done. Several arguments can be parsed in the command line, see in the `get_parser()` function in the `train.py` and `train_ddp.py`. You need to specify the model, the dimension, the dataset, whether use pretrain weights, batch size, and the unique experiment name. Our code will find the corresponding configuration and dataset for training.

Here is an example to train 3D MedFormer with one gpu on ACDC:

`python train.py --model medformer --dimension 3d --dataset acdc --batch_size 3 --unique_name acdc_3d_medformer --gpu 0`

This command will start the cross validation on ACDC. The training loss and evaluation performance will be logged by tensorboard. You can find them in the `log/dataset/unique_name` folder. All the standard output, the configuration, and model weigths will be saved in the `exp/dataset/unique_name` folder. The results of cross validation will be saved in `exp/dataset/unique_name/cross_validation.txt`.

Besides training with a single GPU, we also provide distributed training (DDP) and automatic mixed precision (AMP) training in the `train_ddp.py`. The `train_ddp.py` is the same as `train.py` except it supports DDP and AMP. We recomend you to start with `train.py` to make sure the whole train and eval pipeline is correct, and then use `train_ddp.py` for faster training or larger batch size.

Example of using DDP:

`python train_ddp.py --model medformer --dimension 3d --dataset acdc --batch_size 16 --unique_name acdc_3d_medformer_ddp --gpu 0,1,2,3`

Example of using DDP and AMP:

`python train_ddp.py --model medformer --dimension 3d --dataset acdc --batch_size 32 --unique_name acdc_3d_medformer_ddp_amp --gpu 0,1,2,3`

We have not fully benchmark if AMP can speed up training, but AMP can reduce the GPU memory consumption a lot.


### Adapt to your own task.

In this section, I'll show you how to modify our framework for your own task or dataset. The steps consist of data preparation, model design, configuration, training and evaluation.

#### Data preparation

In this step, you need to preprocess the data and write the PyTorch Dataset. You can refer to our examples for public datasets in `dataset_concersion/` and `training/dataset/`.

Actually, you can write at your will as long as the PyTorch Dataset can correctly sample data during training. But we recomand you to follow our examples to make the processed dataset. First, resample all data samples and corresponding labels to the median spacing. Resample the images with BSpline resampling. Resample the labels with NearestNeighbor resampling. Then store all the processed images and labels into the target path using the following structure:

|-list/  
|- name_data01.nii.gz  
|- name_data01_gt.nii.gz  
|- name_data02.nii.gz  
|- name_data02_gt.nii.gz  
|- ...  
|- name_datan.nii.gz  
|- name_datan_gt.nii.gz

All the names are saved as a yaml file in the `list/dataset.yaml`. You can save the data into your familar format, the `.nii.gz` is just an example. 

#### Write Dataset

After processing the data, you need to write the PyTorch Dataset, and save it into `training/dataset/`. Costumized `__init__()` `__len__()` and `__getitem__()` functions are needed. As 3D medical images are usually large and I/O takes a long time, I usually load all data into memory to reduce I/O time. The augmentation are done on the fly during training, you need to specify the augmentation used in the `__getitem__()` function. We provide some commonly used augmentation operations in `training/augmentation.py`.

After the Dataset is written, add it to the `get_dataset()` function in the `/training/dataset/utils.py`, such that the dataset can be initialized by our training script.


#### Model design

The model definitions are in the `model/` folder. Put your own model into the corresponding dimension (dim2 or dim3) folder. It is recomanded that your model definition in a `model.py` file, and put all support modules and functions in to a separate `model_utils.py` file. At last, add your model into the `get_model()` function in the `/model/utils.py`, such that the model can be initialized by our training script.

#### Training and evaluation

After all steps are finished, you can train the customized model and dataset as our examples for public datasets. If you need customized inference or evaluation, please see in `inferece/` and `training/validation.py`.





