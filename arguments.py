import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--epoch', default=90, type=int, help='length of epochs to train')
parser.add_argument('--batch-size', default=128, type=int, help='training batch size')
parser.add_argument('--random-seed', default=None, type=int, help='random seed')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='norm', type=str, help='model to train')
parser.add_argument('--cosine-penalty', default=-1., type=float, help='cosine_penalty')
parser.add_argument('--spectral-penalty', default=-1., type=float, help='spectral_penalty')
parser.add_argument('--num-workers', default=2, type=int, help='num_workers')
parser.add_argument('--save', '-s', action='store_true', help='save every 10 epoch')
args = parser.parse_args()
