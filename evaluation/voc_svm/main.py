#
# Authors: Wouter Van Gansbeke, Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import numpy as np
import os
import torch

from metrics import get_precision_recall
from sklearn.svm import LinearSVC
from torchvision import transforms, datasets, models
from voc import Voc2007Classification


# Arguments
parser = argparse.ArgumentParser('Arguments VOC SVM')
parser.add_argument('--data', type=str, required=True, 
                    help='Location where data is saved')
parser.add_argument('--pretrained-weights', type=str, required=True,
                    help='Location where MoCo pretrained weights are saved')
args = parser.parse_args()


def main():
    print('Arguments {}'.format(args))

    # Create train and validation set
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    train_dataset = Voc2007Classification(args.data, set='trainval',transform = transform)
    val_dataset = Voc2007Classification(args.data, set='test',transform = transform)
    print('Train/val dataset contain {}/{} samples'.format(len(train_dataset), len(val_dataset)))

    # Create model
    print('Create ResNet-50 model')
    model = models.__dict__['resnet50'](num_classes=128, pretrained=False)
            
    # Load pre-trained weights
    print('Load weights from {}'.format(args.pretrained_weights))
    if os.path.exists(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location='cpu')['state_dict']
        
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            del state_dict[k]  
        sg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model.fc = torch.nn.Identity()
        print("=> loaded pre-trained model '{}'".format(args.pretrained))

    else:
        raise ValueError("=> no checkpoint found at '{}'".format(args.pretrained))

    model.cuda()   
    model.eval()   

    # Compute val features
    print('Compute val features')
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                        batch_size=16, shuffle=False, num_workers=4)
    val_feats, val_labels = [], []
    
    with torch.no_grad():
        for images, target in val_loader:
            feat = model(images.cuda()).cpu()
            val_feats.append(feat)
            val_labels.append(target)

    val_feats = torch.cat(val_feats,0).numpy()
    val_labels = torch.cat(val_labels,0).numpy()
    val_feats_norm = np.linalg.norm(val_feats, axis=1)
    val_feats = val_feats / (val_feats_norm + 1e-5)[:, np.newaxis]
    
    val_labels[val_labels==0] = -1    
       
    # Compute train features
    print('Compute train features')
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                        batch_size=16, shuffle=False, num_workers=4)
    train_feats, train_labels = [], []
    
    with torch.no_grad():
        for images, target in train_loader:
            feat = model(images.cuda()).cpu()
            train_feats.append(feat)
            train_labels.append(target)

    train_feats = torch.cat(train_feats,0).numpy()
    train_labels = torch.cat(train_labels,0).numpy()
    train_feats_norm = np.linalg.norm(train_feats, axis=1)
    train_feats = train_feats / (train_feats_norm + 1e-5)[:, np.newaxis]
    
    train_labels[train_labels==0] = -1

    # Fit SVM per class
    print('Fit SVM per class and compute mAP')
    cls_ap = np.zeros((train_dataset.get_number_classes(), 1))
    for cls in range(train_dataset.get_number_classes()):
        clf = LinearSVC(C=0.5, class_weight={1: 2, -1: 1}, intercept_scaling=1.0,
            penalty='l2', loss='squared_hinge', tol=1e-4, dual=True, max_iter=2000, 
            random_state=None)
        clf.fit(train_feats, train_labels[:,cls])
        prediction = clf.decision_function(test_feats)                                      
        P, R, score, ap = get_precision_recall(test_labels[:,cls], prediction)
        cls_ap[cls][0] = ap*100
    mean_ap = np.mean(cls_ap, axis=0)
    print('Class AP: {}'.format(cls_ap))
    print('Mean AP: {}'.format(mean_ap))

if __name__=='__main__':
    main()
