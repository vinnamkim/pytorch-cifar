'''Train CIFAR10 with PyTorch.'''
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import get_datasets
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

trainset, trainloader, testset, testloader, num_classes = get_datasets(
    args.dataset, args.batch_size, num_workers)

# Model
print('==> Building model..')
# from torchvision.models import resnet18
if args.model == 'lasso':
    from my_models.lasso_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'norm':
    from my_models.norm_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'norm2':
    from my_models.norm2_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'ws':
    from my_models.ws_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'zerocenter':
    from my_models.zerocenter_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'zerocenter2':
    from my_models.zerocenter2_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm':
    from my_models.doublenorm_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm2':
    from my_models.doublenorm2_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm3':
    from my_models.doublenorm3_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm4':
    from my_models.doublenorm4_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm5':
    from my_models.doublenorm5_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm6':
    from my_models.doublenorm6_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'doublenorm7':
    from my_models.doublenorm7_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'avgpoolnorm':
    from my_models.avgpoolnorm_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'ws_doublenorm':
    from my_models.ws_doublenorm_resnet import resnet50
    net = resnet50(num_classes=num_classes)
elif args.model == 'wc_doublenorm':
    from my_models.wc_doublenorm_resnet import resnet50
    net = resnet50(num_classes=num_classes)

print('num_classes : ', num_classes)
dir_name = args.model + '_50_' + str(args.batch_size)

if args.random_seed is not None:
    dir_name += '_' + str(args.random_seed)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


def get_path(batch_size):
    return os.path.join(
        'cifar100_50_{0}'.format(batch_size),
        args.model + '_50_{0}_10'.format(batch_size),
        'ckpt_90.pth')


checkpoint_sb = torch.load(get_path(32))
checkpoint_lb = torch.load(get_path(64))


def test(alpha, loader):
    checkpoint = {}

    with torch.no_grad():
        for key in checkpoint_sb:
            if 'var' in key:
                x = alpha * checkpoint_lb[key].sqrt() + \
                    (1 - alpha) * checkpoint_sb[key].sqrt()
                checkpoint[key] = x * x
            else:
                checkpoint[key] = alpha * checkpoint_lb[key] + \
                    (1 - alpha) * checkpoint_sb[key]

    net.load_state_dict(checkpoint)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / (batch_idx + 1), 100. * correct / total


grid_size = 50
alpha_range = np.linspace(-1, 2, grid_size)

stats = {
    'train': np.ones([grid_size, 2]),
    'valid': np.ones([grid_size, 2])
}

for i, alpha in enumerate(alpha_range):
    stats['train'][i] = np.array(test(alpha, trainloader))
    stats['valid'][i] = np.array(test(alpha, testloader))

    print('alpha : {0} done'.format(alpha))


plt.plot(stats['train'])
plt.show()
plt.plot(stats['valid'])
plt.show()
torch.save(stats, args.model + '_sharpness.stats')
