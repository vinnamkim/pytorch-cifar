'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='length of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--random_seed', default=None, type=int, help='random seed')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='norm', type=str, help='model to train')
parser.add_argument('--spectral-penalty', default=-1., type=float, help='spectral_penalty')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if args.random_seed is not None:
    print('Set manual seed :', args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

num_workers = 2

if os.name == 'nt':
    num_workers = 0

print('batch_size : ', args.batch_size)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
#from torchvision.models import resnet18
if args.model == 'lasso':
    from my_models.lasso_resnet import resnet18
    net = resnet18()
elif args.model == 'norm':
    from my_models.norm_resnet import resnet18
    net = resnet18()
elif args.model == 'ws':
    from my_models.ws_resnet import resnet18
    net = resnet18()

dir_name = args.model + '_18_' + str(args.batch_size)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
assert os.path.isdir(dir_name), 'Error: no checkpoint directory found!'
checkpoint = torch.load(os.path.join(dir_name, 'ckpt.pth'))
#net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

def get_d_params(ckpt, percentage):
    from collections import OrderedDict
    d_params = OrderedDict()
    
    o_num_params = 0
    n_num_params = 0

    with torch.no_grad():
        for key in ckpt:
            if len(ckpt[key].shape) > 1:
                U, S, V = torch.svd(ckpt[key].flatten(1))
                n = int(len(S) * percentage)
                UU = U[:, :n]
                SS = S[:n]
                VV = V[:, :n]
                dp = UU.mm(SS.diag()).mm(VV.T)
                d_params[key] = dp.reshape_as(ckpt[key])

                o_num_params += ckpt[key].shape.numel()
                n_num_params += UU.shape.numel() + SS.shape.numel() + VV.shape.numel()
            else:
                d_params[key] = ckpt[key]
                o_num_params += ckpt[key].shape.numel()
                n_num_params += ckpt[key].shape.numel()
    
    return d_params, o_num_params, n_num_params
    

criterion = nn.CrossEntropyLoss()

def only_test(percentage):
    d_params, o_num_params, n_num_params = get_d_params(checkpoint['net'], percentage)
    net.load_state_dict(d_params)
    print(n_num_params / o_num_params)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return percentage, test_loss/(batch_idx+1), 100. * correct / total, n_num_params / o_num_params

stats = []

import numpy as np

for i in np.linspace(0.5, 1.0, 20):
    print("Percentage : ", i)
    stats.append(only_test(i))

torch.save(stats, args.model + '_reduction.stats')
