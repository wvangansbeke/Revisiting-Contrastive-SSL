#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import cv2
import numpy as np
import os
import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.models.resnet import resnet50, resnet18
from utils.collate import collate_custom


def get_backbone(p):
    if os.path.exists(p['model_kwargs']['state_dict']):
        print('Load state dict {}'.format(p['model_kwargs']['state_dict']))
        model = resnet50(pretrained=False)        
        state_dict = torch.load(p['model_kwargs']['state_dict'], map_location='cpu')['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        return model

    else:
        raise ValueError('Path does not exist {}'.format(p['model_kwargs']['state_dict']))


def get_model(p):
    if p['train_db_name'] == 'VOCSegmentation':
        # We use a FCN model with dilation 6 in the head.
        from models.model import FCN
        return FCN(get_backbone(p), p['num_classes'] + int(p['has_bg']), dilation=6)

    elif p['train_db_name'] == 'cityscapes':
        # We use a FCN model with dilation 1 in the head.
        from models.model import FCN
        return FCN(get_backbone(p), p['num_classes'] + int(p['has_bg']), dilation=1)

    else:
        raise ValueError('No model for train dataset {}'.format(p['train_db_name']))


def get_train_dataset(p, transform=None):
    if p['train_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split=p['train_db_kwargs']['split'], transform=transform)

    elif p['train_db_name'] == 'cityscapes':
        from data.dataloaders.cityscapes import Cityscapes
        dataset = Cityscapes(split='train', transform=transform)
    
    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    return dataset


def get_val_dataset(p, transform=None):
    if p['val_db_name'] == 'VOCSegmentation':
        from data.dataloaders.pascal_voc import VOC12
        dataset = VOC12(split='val', transform=transform)
    
    elif p['val_db_name'] == 'cityscapes':
        from data.dataloaders.cityscapes import Cityscapes
        dataset = Cityscapes(split='val', transform=transform)
    
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))
    
    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['train_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['val_db_kwargs']['batch_size'], pin_memory=True, 
            collate_fn=collate_custom, drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['train_db_name'] == 'VOCSegmentation':
        import data.dataloaders.fblib_transforms as fblib_tr
        return transforms.Compose([fblib_tr.RandomHorizontalFlip(),
                                       fblib_tr.ScaleNRotate(rots=(-5,5), scales=(.75,1.25),
                                        flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                       fblib_tr.FixedResize(resolutions={'image': tuple((512,512)), 'semseg': tuple((512,512))},
                                        flagvals={'semseg': cv2.INTER_NEAREST, 'image': cv2.INTER_CUBIC}),
                                       fblib_tr.ToTensor(),
                                        fblib_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    elif p['train_db_name'] == 'cityscapes':
        import data.dataloaders.vanilla_transforms as tr
        return transforms.Compose([tr.RandomScale([0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0]), 
                                    tr.RandomCrop((768, 768)), tr.RandomHorizontalFlip(),
                                    tr.ToTensor(), tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    else:
        raise NotImplementedError

    
def get_val_transformations(p):
    if p['val_db_name'] == 'VOCSegmentation':
        import data.dataloaders.fblib_transforms as fblib_tr
        return transforms.Compose([fblib_tr.FixedResize(resolutions={'image': tuple((512,512)), 
                                                            'semseg': tuple((512,512))},
                                                flagvals={'image': cv2.INTER_CUBIC, 'semseg': cv2.INTER_NEAREST}),
                                    fblib_tr.ToTensor(),
                                    fblib_tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    
    elif p['val_db_name'] == 'cityscapes':
        import data.dataloaders.vanilla_transforms as tr
        return transforms.Compose([tr.ToTensor(), tr.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    else:
        raise NotImplementedError


def get_optimizer(p, parameters):
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(parameters, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'poly':
        lambd = pow(1-(epoch/p['epochs']), 0.9)
        lr = lr * lambd

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
