import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    #---------------------------------#
    # Whether or not Cuda uses Cuda
    # If you don't have a GPU, you can set it to False
    #--------------------------------- #
    Cuda = True
    #---------------------------------------------------------------------#
    # distributed is used to specify whether to use a single node and multiple cards for distributed operation
    # Terminal commands are only available on Ubuntu. CUDA_VISIBLE_DEVICES is used to specify the graphics card under Ubuntu.
    # By default, all graphics cards are invoked in DP mode on Windows, and DDP is not supported.
    # DP Mode:
    # Set distributed = False
    # Enter CUDA_VISIBLE_DEVICES=0,1 python train.py in the terminal
    # DDP Mode:
    # Set distributed = True
    # Enter CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py in the terminal
    #--------------------------------------------------------------------- #
    distributed     = False
    #---------------------------------------------------------------------#
    # sync_bn Whether to use sync_bn, DDP mode is available for multiple cards
    #--------------------------------------------------------------------- #
    sync_bn         = False
    #---------------------------------------------------------------------#
    # Whether fp16 uses mixed-precision training
    # It can reduce the video memory by about half, and requires pytorch 1.7.1 or higher
    #--------------------------------------------------------------------- #
    fp16            = False
    #-----------------------------------------------------#
    # num_classes You must modify your own dataset to train yourself
    # The number of categories you need is +1, e.g. 2+1
    #----------------------------------------------------- #
    num_classes = 2
    #-----------------------------------------------------#
    #   vgg
    #-----------------------------------------------------#
    backbone    = "vgg"
    #----------------------------------------------------------------------------------------------------------------------------#
    # Whether pretrained uses the pre-trained weights of the backbone network, the weights of the backbone are used here, so they are loaded when the model is built.
    # If model_path is set, the weight of the trunk does not need to be loaded, and the value of pretrained is meaningless.
    # If you do not set the model_path, pretrained = True, only the trunk will be loaded to start training.
    # If you do not set model_path, pretrained = False, Freeze_Train = Fasle, the training starts from 0 and there is no process of freezing the trunk.
    #---------------------------------------------------------------------------------------------------------------------------- #
    pretrained  = False
    #----------------------------------------------------------------------------------------------------------------------------#
    #   The pre-trained weight of the model 
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path  = "model_data/unet_vgg_voc.pth"
    #-----------------------------------------------------#
    #  input_shape Enter the size of the image, in multiples of 32
    #-----------------------------------------------------#
    input_shape = [96, 96]
    Init_Epoch          = 0
    Freeze_Epoch        = 60
    Freeze_batch_size   = 4
    UnFreeze_Epoch      = 60
    Unfreeze_batch_size = 4
    Freeze_Train        = True
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'cos'
    save_period         = 5
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 5
    
    #------------------------------#
    #   Dataset path
    #------------------------------#
    VOCdevkit_path  = 'VOCdevkit_u-net'
    #------------------------------------------------------------------#
    # Suggested Options:
    # If there are few types (several types), set to True
    # If there are more than a dozen types of batch_size, set to True if the  is large (10 or more).
    # If the batch_size is small (less than 10) if there are more than 10 types, set it to False
    #------------------------------------------------------------------ #
    dice_loss       = True
    #------------------------------------------------------------------#
    # Whether to use focal loss to prevent imbalance between positive and negative samples
    #------------------------------------------------------------------ #
    focal_loss      = False
    #------------------------------------------------------------------#
    # Whether to assign different loss weights to different types is balanced by default.
    # If you set it, please note that it is set to numpy, and the length is the same as that of the num_classes.
    # Such as:
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #------------------------------------------------------------------#
    cls_weights     = np.ones([num_classes], np.float32)
    #------------------------------------------------------------------#
    # num_workers Sets whether to use multithreading to read data, 1 indicates that multithreading is disabled
    # When enabled, it will speed up data reading, but it will take up more memory
    # Multithreading in keras is sometimes much slower
    # Enable multi-threading when IO is the bottleneck, that is, the GPU computing speed is much faster than the speed of reading images.
    #------------------------------------------------------------------ #
    num_workers     = 0

    #------------------------------------------------------#
    # Set up the graphics card you want to use
    #------------------------------------------------------ #
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

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weights_init(model)
    if model_path != '':

        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        

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

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))



    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
   
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )

    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            model.freeze_backbone()
            

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)


        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training, so expand the dataset。")

        train_dataset   = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)
        

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
 
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size


                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training, so expand the dataset")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
