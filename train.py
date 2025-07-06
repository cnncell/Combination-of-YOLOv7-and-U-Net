#-------------------------------------#
#       Train the dataset
#-------------------------------------#
import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (download_weights, get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

'''
Key points when training your own object detection model:
1. Before training, carefully check if your format meets requirements. This library requires the dataset in VOC format, including input images and labels:
   - Input images are .jpg files, no fixed size required (automatically resized during training).
   - Grayscale images are automatically converted to RGB, no manual modification needed.
   - If input images have a suffix other than .jpg, batch convert them to .jpg before training.

   - Labels are .xml files containing target information, corresponding to input images.

2. The loss value indicates convergence. What matters is the convergence trend (validation loss decreasing). If validation loss stabilizes, the model is converging.
   The specific loss value is meaningless; its magnitude depends on the loss function (not necessarily close to 0). To make losses look smaller, divide by 10000 in the loss function.
   Training losses are saved in logs/loss_%Y_%m_%d_%H_%M_%S.

3. Trained weights are saved in the logs folder. Each epoch contains multiple steps (gradient descents).
   Weights won't be saved if only a few steps are trained. Understand the concepts of Epoch and Step.
'''
if __name__ == "__main__":
    #---------------------------------#
    #   Cuda: Use Cuda or not
    #   Set to False without GPU
    #---------------------------------#
    Cuda            = True
    #----------------------------------------------#
    #   Seed: Fix random seed for reproducible results
    #----------------------------------------------#
    seed            = 11
    #---------------------------------------------------------------------#
    #   distributed: Enable single-machine multi-GPU distributed training
    #               Terminal commands only support Ubuntu. CUDA_VISIBLE_DEVICES specifies GPUs on Ubuntu.
    #               Windows uses DP mode by default, not supporting DDP.
    #   DP mode:
    #       Set     distributed = False
    #       Run     CUDA_VISIBLE_DEVICES=0,1 python train.py in terminal
    #   DDP mode:
    #       Set     distributed = True
    #       Run     CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py in terminal
    #---------------------------------------------------------------------#
    distributed     = False
    #---------------------------------------------------------------------#
    #   sync_bn: Use synchronized BN (available for DDP mode)
    #---------------------------------------------------------------------#
    sync_bn         = False
    #---------------------------------------------------------------------#
    #   fp16: Use mixed-precision training
    #         Reduces memory usage by ~half, requires PyTorch 1.7.1+
    #---------------------------------------------------------------------#
    fp16            = False
    #---------------------------------------------------------------------#
    #   classes_path: Path to txt in model_data, related to your dataset
    #               Must be modified to match your dataset before training
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path: Txt file for prior boxes (usually unchanged)
    #   anchors_mask: Helps code find corresponding prior boxes (usually unchanged)
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   Pretrained weights can be downloaded via the link in README. The model's pretrained weights are universal for different datasets due to generic features.
    #   The most important part of pretrained weights is the backbone feature extraction network, used for feature extraction.
    #   Pretrained weights are essential for 99% of cases; random backbone weights yield poor feature extraction and training results.
    #
    #   To resume training from a checkpoint, set model_path to the weights in the logs folder and adjust freeze/unfreeze parameters for continuity.
    #   Set model_path = '' to not load any weights.
    #
    #   Here, the entire model's weights are loaded in train.py.
    #   To train from scratch, set model_path = '' and Freeze_Train = False (trains from scratch without freezing the backbone).
    #   
    #   Training from scratch typically yields poor results due to random weights. It's highly discouraged!
    #   Two approaches for training from scratch:
    #   1. Leverage Mosaic data augmentation: Set UnFreeze_Epoch large (300+), batch size large (16+), and use abundant data (10k+).
    #      Set mosaic=True and train with random initialization, though results still lag behind pretrained models. (Suitable for large datasets like COCO)
    #   2. First train a classification model on ImageNet to obtain backbone weights, which are generic to this model.
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/yolov7_weights.pth'
    #------------------------------------------------------#
    #   input_shape: Input shape, must be a multiple of 32
    #------------------------------------------------------#
    input_shape     = [640, 640]
    #------------------------------------------------------#
    #   phi: YOLOv7 version, options:
    #        l: YOLOv7
    #        x: YOLOv7_x
    #------------------------------------------------------#
    phi             = 'l'
    #----------------------------------------------------------------------------------------------------------------------------#
    #   pretrained: Use pretrained backbone weights (loaded during model construction).
    #               If model_path is set, pretrained is irrelevant.
    #               If model_path is not set and pretrained = True, only the backbone is loaded for training.
    #               If model_path is not set and pretrained = False with Freeze_Train = False, trains from scratch without freezing.
    #----------------------------------------------------------------------------------------------------------------------------#
    pretrained      = False
    #------------------------------------------------------------------#
    #   mosaic: Enable Mosaic data augmentation.
    #   mosaic_prob: Probability of using Mosaic per step (default 50%).
    #
    #   mixup: Enable mixup data augmentation (only valid if mosaic=True).
    #          Applies mixup to Mosaic-augmented images.
    #   mixup_prob: Probability of using mixup after Mosaic (default 50%).
    #               Total mixup probability: mosaic_prob * mixup_prob.
    #
    #   special_aug_ratio: Refer to YoloX. Mosaic images deviate from real-world distributions, so Mosaic is enabled within special_aug_ratio.
    #                       Default: first 70% of epochs (70/100 epochs).
    #------------------------------------------------------------------#
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    #------------------------------------------------------------------#
    #   label_smoothing: Label smoothing (typically <0.01, e.g., 0.01, 0.005)
    #------------------------------------------------------------------#
    label_smoothing     = 0

    #----------------------------------------------------------------------------------------------------------------------------#
    #   Training has two stages: frozen and unfrozen. Freezing accelerates training for limited hardware.
    #   Frozen training uses less VRAM; for poor GPUs, set Freeze_Epoch = UnFreeze_Epoch and Freeze_Train = True for only frozen training.
    #   
    #   Suggested parameter configurations (adjust flexibly):
    #   (1) Train from entire model's pretrained weights:
    #       Adam:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 100, Freeze_Train = True, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 100, Freeze_Train = False, optimizer_type = 'adam', Init_lr = 1e-3, weight_decay = 0. (Unfrozen)
    #       SGD:
    #           Init_Epoch = 0, Freeze_Epoch = 50, UnFreeze_Epoch = 300, Freeze_Train = True, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (Frozen)
    #           Init_Epoch = 0, UnFreeze_Epoch = 300, Freeze_Train = False, optimizer_type = 'sgd', Init_lr = 1e-2, weight_decay = 5e-4. (Unfrozen)
    #       Note: UnFreeze_Epoch can be 100-300.
    #   (2) Train from scratch:
    #       Init_Epoch = 0, UnFreeze_Epoch >= 300, Unfreeze_batch_size >= 16, Freeze_Train = False (unfrozen training)
    #       Note: UnFreeze_Epoch should be >= 300. optimizer_type = 'sgd', Init_lr = 1e-2, mosaic = True.
    #   (3) Batch size settings:
    #       Larger is better within GPU memory limits. Out of memory (OOM) indicates reducing batch_size.
    #       Batch size must be >= 2 (affected by BatchNorm, cannot be 1).
    #       Freeze_batch_size is typically 1-2x Unfreeze_batch_size for stable learning rate adjustment.
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Frozen training parameters
    #   The backbone is frozen, feature extraction network unchanged
    #   Less VRAM usage, fine-tune the network
    #   Init_Epoch: Starting epoch (can be > Freeze_Epoch for resume, e.g., Init_Epoch = 60, Freeze_Epoch = 50, UnFreeze_Epoch = 100 skips freezing)
    #   Freeze_Epoch: Frozen training epochs (ignored if Freeze_Train=False)
    #   Freeze_batch_size: Batch size for frozen training (ignored if Freeze_Train=False)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   Unfrozen training parameters
    #   The backbone is unfrozen, feature extraction network changes
    #   More VRAM usage, all parameters are updated
    #   UnFreeze_Epoch: Total training epochs (SGD needs more epochs to converge)
    #   Unfreeze_batch_size: Batch size for unfrozen training
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train: Enable frozen training
    #               Default: freeze first, then unfreeze
    #------------------------------------------------------------------#
    Freeze_Train        = True

    #------------------------------------------------------------------#
    #   Other training parameters: learning rate, optimizer, learning rate decay
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr: Maximum learning rate
    #   Min_lr: Minimum learning rate (default: 0.01 * Init_lr)
    #------------------------------------------------------------------#
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type: Optimizer type (adam, sgd)
    #                   Use Init_lr=1e-3 for Adam, Init_lr=1e-2 for SGD
    #   momentum: Momentum parameter for optimizer
    #   weight_decay: Weight decay to prevent overfitting (set to 0 for Adam)
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type: Learning rate decay type (step, cos)
    #------------------------------------------------------------------#
    lr_decay_type       = "cos"
    #------------------------------------------------------------------#
    #   save_period: Save weights every save_period epochs
    #------------------------------------------------------------------#
    save_period         = 10
    #------------------------------------------------------------------#
    #   save_dir: Folder for weights and logs
    #------------------------------------------------------------------#
    save_dir            = 'logs'
    #------------------------------------------------------------------#
    #   eval_flag: Evaluate during training (on validation set)
    #   eval_period: Evaluate every eval_period epochs (avoid frequent evaluation)
    #   Note: mAP here differs from get_map.py for two reasons:
    #   (1) This is mAP on the validation set.
    #   (2) Evaluation parameters are conservative for speed.
    #------------------------------------------------------------------#
    eval_flag           = True
    eval_period         = 10
    #------------------------------------------------------------------#
    #   num_workers: Number of worker threads for data loading
    #               Increase for faster loading, but more memory (set to 2 or 0 for low memory)
    #------------------------------------------------------------------#
    num_workers         = 4

    #------------------------------------------------------#
    #   train_annotation_path: Training image and label paths
    #   val_annotation_path: Validation image and label paths
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    seed_everything(seed)
    #------------------------------------------------------#
    #   Set available GPUs
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    #------------------------------------------------------#
    #   Get classes and anchors
    #------------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #----------------------------------------------------#
    #   Download pretrained weights
    #----------------------------------------------------#
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi)  
            dist.barrier()
        else:
            download_weights(phi)
            
    #------------------------------------------------------#
    #   Create YOLO model
    #------------------------------------------------------#
    model = YoloBody(anchors_mask, num_classes, phi, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        #------------------------------------------------------#
        #   Weights download link in README
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   Load weights based on key matching
        #------------------------------------------------------#
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   Display unmatched keys
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44mNote: Missing head weights are normal; missing backbone weights are errors.\033[0m")

    #----------------------#
    #   Get loss function
    #----------------------#
    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, anchors_mask, label_smoothing)
    #----------------------#
    #   Record loss history
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path