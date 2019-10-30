'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
from arguments import args

from utils import progress_bar

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if args.random_seed is not None:
    print('Set manual seed :', args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

# Data
print('==> Preparing data..')
num_workers = args.num_workers

if os.name == 'nt':
    num_workers = 0

from datasets import get_datasets
trainset, trainloader, testset, testloader, num_classes = get_datasets(
    args.dataset, args.batch_size, num_workers)

# Model
print('==> Building model..')
#from torchvision.models import resnet18
if args.model == 'lasso':
    from my_models.lasso_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'norm':
    from my_models.norm_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'norm2':
    from my_models.norm2_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'ws':
    from my_models.ws_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'zerocenter':
    from my_models.zerocenter_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'zerocenter2':
    from my_models.zerocenter2_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm':
    from my_models.doublenorm_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm2':
    from my_models.doublenorm2_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm3':
    from my_models.doublenorm3_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm4':
    from my_models.doublenorm4_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm5':
    from my_models.doublenorm5_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm6':
    from my_models.doublenorm6_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'doublenorm7':
    from my_models.doublenorm7_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'avgpoolnorm':
    from my_models.avgpoolnorm_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'ws_doublenorm':
    from my_models.ws_doublenorm_resnet import resnet18
    net = resnet18(num_classes=num_classes)
elif args.model == 'wc_doublenorm':
    from my_models.wc_doublenorm_resnet import resnet18
    net = resnet18(num_classes=num_classes)

print('num_classes : ', num_classes)
dir_name = args.model + '_18_' + str(args.batch_size)

if args.random_seed is not None:
    dir_name += '_' + str(args.random_seed)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(dir_name), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(dir_name, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

from sklearn.model_selection import StratifiedKFold
fold = StratifiedKFold(n_splits=500, shuffle=True, random_state=args.random_seed)
splits = fold.split(trainset.data, trainset.targets)

datasets = [s[1] for i, s in enumerate(splits) if i < 5]
    
for batch in datasets:
    inputs = torch.stack([trainset[i][0] for i in batch])
    outputs = torch.tensor([trainset[i][1] for i in batch])

from hooks import add_hooks, init_cond_stats, add_cond_stats

add_hooks(net)

results = {}

for epoch in range(5, 91, 5):
    checkpoint = torch.load(os.path.join(dir_name, 'ckpt_{0}.pth'.format(epoch)))
    net.load_state_dict(checkpoint)
    print('epoch {0} loaded batch_size : {1}'.format(epoch, args.batch_size))
    
    stats = init_cond_stats(net)

    for step, batch in enumerate(trainloader):
        inputs, targets = batch
        
        net.zero_grad()
        loss = criterion(net(inputs.cuda()), targets.cuda())
        loss.backward()
        add_cond_stats(net, stats)
        
        if step > 300 / args.batch_size:
            break
    
    print('epoch {0} done'.format(epoch))
    results[epoch] = stats

torch.save(results, os.path.join(dir_name, 'forwards_backwards_{0}.stats'.format(args.batch_size)))
