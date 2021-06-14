"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import builtins
import math
import os, sys
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.constrained_cropping import CustomMultiCropDataset, CustomMultiCropping
from utils.dataset import ImageFolder
from utils.logger import Logger
from datetime import datetime

import builder.loader
import builder.builder_ours

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', 
                    help='output directory')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

## crops
parser.add_argument("--num_crops", type=int, default=[2, 4], nargs="+",
                            help="amount of crops")
parser.add_argument("--size_crops", type=int, default=[224, 96], nargs="+",
                            help="resolution of inputs")
parser.add_argument("--min_scale_crops", type=float, default=[0.14, 0.05], nargs="+",
                            help="min area of crops")
parser.add_argument("--max_scale_crops", type=float, default=[1, 0.14], nargs="+",
                            help="max area of crops")

## 
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)') 

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# extra options
parser.add_argument('--constrained-cropping', action='store_true',
                    help='condition small crops on key crop')
parser.add_argument('--auto-augment', type=int, default=[], nargs='+',
                    help='Apply auto-augment 50 % of times to the selected crops')

# auxiliary options
parser.add_argument('--aux-weight', type=float, default=0.4,
                    help='Weight for auxiliary loss (default: 0.4)')
parser.add_argument('--aux-topk', type=int, default=50,
                    help='Top-K to use for loss (default: 50)')

def main():
    args = parser.parse_args()
    assert(args.mlp) # Just a safety check.
    assert(args.cos)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    if args.gpu == 0:
        sys.stdout = Logger(os.path.join(args.output_dir, 'log_file.txt'))
        print(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = builder.builder_ours.MoCo(
        models.__dict__[args.arch], args.aux_topk,
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.distributed:
        if args.gpu is not None:
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optionally resume from a checkpoint
    chkpt = os.path.join(args.output_dir, 'checkpoint.pth.tar')
    if os.path.isfile(chkpt):
        print("=> loading checkpoint '{}'".format(chkpt))
        if args.gpu is None:
            checkpoint = torch.load(chkpt)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(chkpt, map_location=loc)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(chkpt, checkpoint['epoch']))
    
    else:
        print("=> no checkpoint found at '{}'".format(chkpt))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    
    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    if args.constrained_cropping:
        print('We apply constrained cropping here - Use custom multi-cropping dataset')
        assert(len(args.size_crops) == 2)
        assert(args.num_crops[0] == 2) # only tested in this way
        assert(args.num_crops[1] == 4) # only testes in this way
        crop_transform = CustomMultiCropping(size_large = args.size_crops[0],
                        scale_large=(args.min_scale_crops[0], args.max_scale_crops[0]),
                        size_small=args.size_crops[1],
                        scale_small=(args.min_scale_crops[1], args.max_scale_crops[1]),
                        N_large=args.num_crops[0], N_small=args.num_crops[1], 
                        condition_small_crops_on_key=True)
        assert(crop_transform.N_large == 2)
        assert(crop_transform.N_small == 4)
       
        if len(args.auto_augment) == 0: 
            print('No auto augment - Apply regular moco v2 as secondary transform')
            secondary_transform = transforms.Compose([
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])

        else:
            from utils.auto_augment.auto_augment import AutoAugment
            from utils.auto_augment.random_choice import RandomChoice
            print('Auto augment - Apply custom auto-augment strategy')
            counter = 0
            secondary_transform = []
        
            for i in range(len(args.size_crops)):
                for j in range(args.num_crops[i]):
                    if not counter in set(args.auto_augment):
                        print('Crop {} - Apply regular secondary transform'.format(counter))
                        secondary_transform.extend([transforms.Compose([
                            transforms.RandomApply([
                                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                            ], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=0.5),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])])

                    else:
                        print('Crop {} - Apply auto-augment/regular secondary transform'.format(counter))
                        trans1 = transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            AutoAugment(),
                                            transforms.ToTensor(),
                                            normalize])

                        trans2 = transforms.Compose([
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=0.5),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize])
                   
                        secondary_transform.extend([RandomChoice([trans1, trans2])])
                    
                    counter += 1                        
    
        
        base_dataset = ImageFolder(args.data)
        train_dataset = CustomMultiCropDataset(base_dataset, crop_transform, 
                            secondary_transform, return_crop_params=False) 
    
        print('CustomMultiCrop is {}'.format(train_dataset.multi_crop))
        print('Secondary transform is {}'.format(train_dataset.secondary_transform))

    else:
        print('No constrained cropping is desired - Implement augmentations as list of tf')
        trans = []
        assert(len(args.auto_augment) == 0) # not tested

        for i in range(len(args.size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(args.size_crops[i],
                                        scale=(args.min_scale_crops[i], args.max_scale_crops[i]))
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([builder.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])] * args.num_crops[i])

        print('Transformations are {}'.format(trans))
        train_dataset = builder.loader.MultiCropDataset(args.data, trans)

    print('Train dataset contains {} files'.format(len(train_dataset)))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        start = datetime.now()
        train(train_loader, model, criterion, optimizer, epoch, args)
        print('Epoch took {}'.format(datetime.now()-start))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(args.output_dir, 'checkpoint.pth.tar'.format(epoch)))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    aux_losses = AverageMeter('Aux Loss', ':.4e')
    total_losses = AverageMeter('Total Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5-7', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, total_losses, losses, aux_losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, inputs in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        b, c, h, w = inputs[0].shape

        if len(args.num_crops) > 1:
            input_small = inputs[args.num_crops[0]].unsqueeze(1)
            for j in range(1, args.num_crops[1]):
                input_small = torch.cat((input_small, inputs[j+args.num_crops[0]].unsqueeze(1)), dim=1)
            input_small = input_small.view(-1, c, args.size_crops[1], args.size_crops[1]).cuda(non_blocking=True)
        else:
            input_small = None
        
        output, target, aux_loss = model(inputs[0].cuda(non_blocking=True), 
                               inputs[1].cuda(non_blocking=True), input_small)
        loss = criterion(output, target)
        total_loss = loss + args.aux_weight * aux_loss

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs[0].size(0))
        aux_losses.update(aux_loss.item(), inputs[0].size(0))
        total_losses.update(total_loss.item(), inputs[0].size(0))
        top1.update(acc1[0], inputs[0].size(0))
        top5.update(acc5[0], inputs[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
