## Recent Changes

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
