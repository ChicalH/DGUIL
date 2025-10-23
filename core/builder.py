from typing import Callable

import torch
import torch.optim
from torch import nn
from torchpack.utils.config import configs
from torchpack.utils.typing import Dataset, Optimizer, Scheduler

import pdb

__all__ = [
    'make_dataset', 'make_model', 'make_criterion', 'make_optimizer', 'make_scheduler'
]


def get_kitti(phase):
    from core.datasets import SemanticKITTI
    dataset_config = configs.src_dataset if phase == 'train' else configs.tgt_dataset
    dataset = SemanticKITTI(root=dataset_config.root,
                            num_points=dataset_config.num_points,
                            voxel_size=dataset_config.voxel_size)
    return dataset[phase]

def get_kitti_inf(phase):
    from core.datasets import SemanticKITTI_inf
    dataset_config = configs.src_dataset if phase == 'train' else configs.tgt_dataset
    dataset = SemanticKITTI_inf(root=dataset_config.root,
                            num_points=dataset_config.num_points,
                            voxel_size=dataset_config.voxel_size)
    return dataset[phase]

def get_poss():
    from core.datasets import POSSDataset
    dataset_config = configs.tgt_dataset
    dataset = POSSDataset(data_path=dataset_config.root,
                            num_points=dataset_config.num_points,
                            voxel_size=dataset_config.voxel_size)
    return dataset

def get_nuscenes():
    from core.datasets import nuScenesDataset
    dataset_config = configs.tgt_dataset
    dataset = nuScenesDataset(data_path=dataset_config.root,
                            num_points=dataset_config.num_points,
                            voxel_size=dataset_config.voxel_size)
    return dataset

def get_synlidar():
    from core.datasets import SynLiDAR
    dataset_config = configs.src_dataset
    dataset = SynLiDAR(root=configs.src_dataset.root,
                       num_points=configs.src_dataset.num_points,
                       voxel_size=configs.src_dataset.voxel_size,
                       src=configs.tgt_dataset.name)
    return dataset['train']


def get_stf(phase='test'):
    from core.datasets import SemanticSTF
    dataset_config = configs.src_dataset if phase == 'train' else configs.tgt_dataset
    dataset = SemanticSTF(root=dataset_config.root,
                          num_points=dataset_config.num_points,
                          voxel_size=dataset_config.voxel_size)
    return dataset[phase]


def make_dataset() -> Dataset:
    # source dataset
    if configs.src_dataset.name == 'synlidar':
        src_dataset = get_synlidar()
    elif configs.src_dataset.name == 'semantickitti':
        src_dataset = get_kitti(phase='train')
    elif configs.src_dataset.name == 'semanticstf':
        src_dataset = get_stf(phase='train')
    else:
        raise NotImplementedError(configs.dataset.name)

    # target dataset
    if configs.tgt_dataset.name == 'semantickitti':
        tgt_dataset = get_kitti_inf(phase='test')
    elif configs.tgt_dataset.name == 'poss':
        tgt_dataset = get_poss()
    elif configs.tgt_dataset.name == 'nuscenes':
        tgt_dataset = get_nuscenes()
    elif configs.tgt_dataset.name == 'semanticstf':
        tgt_dataset = get_stf()
    else:
        raise NotImplementedError(configs.dataset.name)

    dataset = {}
    dataset['train'] = src_dataset
    dataset['test'] = tgt_dataset
    return dataset


def make_model() -> nn.Module:  
    if configs.model.name == 'minkunet':
        from core.models.semantic_kitti import MinkUNet_DR as MinkUNet
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        if 'intervention' in configs.model:
            model = MinkUNet(num_classes=configs.data.num_classes, cr=cr,
            inte=configs.model.intervention,nc= configs.model.nc)
        else:
            model = MinkUNet(num_classes=configs.data.num_classes, cr=cr)
    elif configs.model.name == 'ct':
        from core.models.semantic_kitti.unet_spherical_transformer import  Semantic as ct
        if 'cr' in configs.model:
            cr = configs.model.cr
        else:
            cr = 1.0
        patch_size = np.array([args.voxel_size[i] * args.patch_size for i in range(3)]).astype(np.float32)
        window_size = args.patch_size * args.window_size
        window_size_sphere = np.array(args.window_size_sphere)
        model = ct(
            input_c=configs.model.input_c, 
            m=configs.model.m,
            classes=configs.model.classes, 
            block_reps=configs.model.block_reps, 
            block_residual=configs.model.block_residual, 
            layers=configs.model.layers, 
            window_size=window_size, 
            window_size_sphere=window_size_sphere, 
            quant_size=window_size / configs.model.quant_size_scale, 
            quant_size_sphere=window_size_sphere / configs.model.quant_size_scale, 
            rel_query=configs.model.rel_query, 
            rel_key=configs.model.rel_key, 
            rel_value=configs.model.rel_value, 
            drop_path_rate=configs.model.drop_path_rate, 
            window_size_scale=configs.model.window_size_scale, 
            grad_checkpoint_layers=configs.model.grad_checkpoint_layers, 
            sphere_layers=configs.model.sphere_layers,
            a=configs.model.a,
        )
    else:
        raise NotImplementedError(configs.model.name)
    return model


def make_criterion() -> Callable:
    if configs.criterion.name == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=configs.criterion.ignore_index)
    else:
        raise NotImplementedError(configs.criterion.name)
    return criterion


def make_optimizer(model: nn.Module) -> Optimizer:
    if configs.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=configs.optimizer.lr,
                                    momentum=configs.optimizer.momentum,
                                    weight_decay=configs.optimizer.weight_decay,
                                    nesterov=configs.optimizer.nesterov)
    elif configs.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    elif configs.optimizer.name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=configs.optimizer.lr,
            weight_decay=configs.optimizer.weight_decay)
    else:
        raise NotImplementedError(configs.optimizer.name)
    return optimizer


def make_scheduler(optimizer: Optimizer) -> Scheduler:
    if configs.scheduler.name == 'none':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda epoch: 1)
    elif configs.scheduler.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.num_epochs)
    elif configs.scheduler.name == 'cosine_warmup':
        from functools import partial

        from core.schedulers import cosine_schedule_with_warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=partial(cosine_schedule_with_warmup,
                              num_epochs=configs.num_epochs,
                              batch_size=configs.batch_size,
                              dataset_size=configs.data.training_size))
    else:
        raise NotImplementedError(configs.scheduler.name)
    return scheduler
