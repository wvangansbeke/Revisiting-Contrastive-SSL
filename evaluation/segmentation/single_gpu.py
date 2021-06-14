#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import sys
import torch

from datetime import datetime
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate
from utils.evaluate_utils import eval_segmentation_supervised_online, save_results_to_disk,\
                                 eval_segmentation_supervised_offline
from termcolor import colored
from utils.logger import Logger
from utils.utils import AverageMeter, ProgressMeter, SemsegMeter


# Parser
parser = argparse.ArgumentParser(description='Finetuning for semantic segmentation.')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--run-idx', type=int, 
                    help='Support for multiple runs')
args = parser.parse_args()


def main():
    cv2.setNumThreads(1)
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.run_idx)
    sys.stdout = Logger(p['log_file'])
    print('Python script is {}'.format(os.path.abspath(__file__)))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    print(model)
    model = model.cuda()
    
    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model.parameters())
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    train_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms) 
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # Validation set at original resolution 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)), 'yellow'))

    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model.cuda()
        start_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        best_iou = checkpoint['best_iou']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        best_epoch = 0
        best_iou = 0
        model = model.cuda()

    # Main loop
    print(colored('Starting main loop', 'blue'))
    
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train 
        print('Train ...')
        start = datetime.now()
        train(p, train_dataloader, model, criterion, optimizer, epoch)
        print('Training took {}'.format(datetime.now()-start))

        # Batch eval
        print('Evaluate ...')
        start = datetime.now()
        eval_val = eval_segmentation_supervised_online(p, val_dataloader, model)
        print('Validation took {}'.format(datetime.now()-start))
        if eval_val['mIoU'] > best_iou:
            print('Found new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
            best_iou = eval_val['mIoU']
            best_epoch = epoch
            torch.save(model.state_dict(), p['best_model'])
        
        else:
            print('No new best model: %.2f -> %.2f (mIoU)' %(100*best_iou, 100*eval_val['mIoU']))
            print('Last best model was found in epoch %d' %(best_epoch))

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_iou': best_iou}, 
                    p['checkpoint'])

    # Evaluate best model at the end at original resolution.
    # This is only required for PASCAL, as the resolution differs per image.
    # In this case, we can not employ the faster batch eval. 
    if p['val_db_name'] == 'VOCSegmentation':
        print(colored('Evaluating best model at the end', 'blue'))
        model.load_state_dict(torch.load(p['best_model']))
        save_results_to_disk(p, val_dataloader, model)
        eval_stats = eval_segmentation_supervised_offline(p, true_val_dataset, verbose=True)

    else:
        pass

def train(p, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('Loss', ':.4e')
    semseg_meter = SemsegMeter(p['num_classes'], train_loader.dataset.get_class_names(),
                            p['has_bg'], ignore_index=255)
    progress = ProgressMeter(len(train_loader),
                    [losses], prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['semseg'].cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, targets)
        losses.update(loss.item())
        semseg_meter.update(torch.argmax(output, dim=1), targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            progress.display(i)

    eval_results = semseg_meter.return_score(verbose = True)

if __name__ == "__main__":
    main()
