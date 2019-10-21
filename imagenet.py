import torch
import torch.nn as nn
import os
import argparse
from torchvision import transforms
from torchvision import datasets
import numpy as np

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)
        
    return tensor, targets

parser = argparse.ArgumentParser(description='PyTorch Imagenet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--model', default='norm', type=str, help='model to train')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--mixed-precision', action='store_true', help='Use mixed-precision')
args = parser.parse_args()

# set seed
from catalyst.dl import SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from catalyst.dl.callbacks.checkpoint import IterationCheckpointCallback

set_global_seed(args.seed)
prepare_cudnn(deterministic=True)

# experiment setup
logdir = "./logdir_" + args.model + "_" + str(args.seed)
num_epochs = args.epochs

# Data loading code
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# data
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=fast_collate)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True,
    sampler=val_sampler,
    collate_fn=fast_collate)

loaders = {"train": train_loader, "valid": val_loader}

# model, criterion, optimizer
print ('Model : ', args.model)
if args.model == 'resnet50_norm':
    from torchvision.models import resnet50
    model = resnet50()
elif args.model == 'resnet50_zerocenter':
    from imagenet_models.zerocenter import resnet50
    model = resnet50()
elif args.model == 'resnet50_zerocenter2':
    from imagenet_models.zerocenter2 import resnet50
    model = resnet50()
else:
    print('unknown model')
    pass

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.1)

# model runner
runner = SupervisedRunner()

# mixed precision
if args.mixed_precision == True:
    from apex import amp
    print('Use mixed precision')
    opt_level = 'O1'
    model = model.to('cuda')
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# model training
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    logdir=logdir,
    callbacks=[
        IterationCheckpointCallback(
            save_n_last=100, num_iters=5000)
    ],
    num_epochs=num_epochs,
    verbose=True
)