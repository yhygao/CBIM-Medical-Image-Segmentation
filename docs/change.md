## Recent Changes

#### Mar. 1, 2023
- Support AMOS CT and MR dataset
- Support using GPU for data augmentation
    - The affine transformation (rotation, scaling, translation, shearing) for 3D image is computational intensive. Previously, we use multiple CPU workers to perform augmentation, which is slow (5-6 seconds for 160\*160\*160 image)
    - Now, we support two ways to use GPU to acesslerate augmentation (0.1-0.3 s for 160\*160\*160 image, with more GPU memory consumption 1-2 G)
    - We support use PyTorch cuda operation. You can simply activate this function my setting 'aug\_device' in the config to 'gpu'.
    - We support NVIDIA DALI to perform augmentation. You can set the 'dataloader' to 'dali' to use DALI operations for augmentation. We also provide commonly used DALI augmentation functions in the *training/augmentation\_dali.py*.
    - In my own experimence, using PyTorch cuda operation already provides huge acceleration. DALI has more advantages in the cpu mode. In the gpu mode, DALI has limited advantages, but needs a lot of time to learn its APIs.
- Add inference code
    - We provide *prediction.py* to make prediction on new testing images and save the corresponding predicted labels into .nii.gz files.
    - The *prediction.py* includes pre-processing, ensembled prediction, and post-processing, functions. ** You need to modify the target\_spacing and pre-processing function to make sure the testing image is resampled and normalized the same as training. **


#### Feb. 10, 2023
- Sypport KiTS19 kidney and tumor CT segmentation dataset


#### Dec. 19, 2022
- Support distributed training with PyTorch DDP
    - We provide a new training script: *train\_ddp.py*, which supports PyTorch distributed training
    - The original single process training script: *train.py* is still preserved.
    - As the debug for multi-processing is hard, you can develop your algorithm with *train.py*, then using *train\_ddp.py* for faster training or larger batch size

- Support Automatic Mixed Precision (AMP)
    - We provide an option for using half precision training in *train\_ddp*
    - We have not benchmark the speed with or without AMP yet, but we find AMP can greatly reduce the GPU memory consumption. So if you want to train large 3D models, AMP is an option.

- We made several improvment on code quality and readablity 
    - Using Python logging instead of print
    - Save all log information to a .txt file
    - Save the configuration of each training with the log
    - Use better log format
