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
parser.add_argument('--cosine-penalty', default=-1., type=float, help='cosine_penalty')
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
elif args.model == 'zerocenter':
    from my_models.zerocenter_resnet import resnet18
    net = resnet18()
elif args.model == 'doublenorm':
    from my_models.doublenorm_resnet import resnet18
    net = resnet18()
elif args.model == 'avgpoolnorm':
    from my_models.avgpoolnorm_resnet import resnet18
    net = resnet18()

dir_name = args.model + '_18_' + str(args.batch_size)

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
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if args.spectral_penalty > 0.:
    from my_models.spectral_penalty import SpectralPenalty
    spectral_penalty = SpectralPenalty(weight=args.spectral_penalty)
else:
    spectral_penalty = None

if args.cosine_penalty > 0.:
    from my_models.cosine_penalty import CosinePenalty
    cosine_penalty = CosinePenalty(weight=args.cosine_penalty)
else:
    cosine_penalty = None

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        origin_loss = loss.item()

        if spectral_penalty is not None:
            loss += spectral_penalty(net)
        
        if cosine_penalty is not None:
            loss += cosine_penalty(net)

        loss.backward()
        optimizer.step()

        train_loss += origin_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    return train_loss / (batch_idx + 1), 100. * correct / total

def test(epoch):
    global best_acc
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

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        torch.save(state, os.path.join(dir_name, 'ckpt.pth'))
        best_acc = acc

    return test_loss / (batch_idx + 1), 100. * correct / total

stats = {
    'train' : [],
    'test' : []
}

for epoch in range(start_epoch, start_epoch + args.epoch):
    stats['train'].append(train(epoch))
    stats['test'].append(test(epoch))
    scheduler.step()
    torch.save(stats, os.path.join(dir_name, 'stats.pth'))
