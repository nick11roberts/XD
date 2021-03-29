import json
import os
import pdb
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.mnist_pixel.utils import data_generator
from TCN.mnist_pixel.model import TCN
import numpy as np
import argparse
from tensorboardX import SummaryWriter

from chrysalis import Chrysalis
from nas import MixedOptimizer, clip_grad_norm


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
parser.add_argument('--patch-conv', action='store_true')
parser.add_argument('--kmatrix-depth', type=int, default=1)
parser.add_argument('--max-kernel-size', type=int, default=1)
parser.add_argument('--base', type=int, default=2)
parser.add_argument('--perturb', type=float, default=0.0)
parser.add_argument('--arch-optim', type=str, default='Adam')
parser.add_argument('--arch-lr', type=float, default=2E-3)
parser.add_argument('--save-dir', type=str, default='test')
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--cooldown', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
batch_size = args.batch_size
n_classes = 10
input_channels = 1
seq_length = int(784 / input_channels)
epochs = args.epochs
steps = 0

print(args)
train_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

model = Chrysalis.metamorphosize(model, in_place=True)
if args.patch_conv:
    X = next(iter(train_loader))
    model.patch_conv(
                     X[0].view(-1, input_channels, seq_length)[:1], 
                     verbose=True,
                     kmatrix_depth=args.kmatrix_depth,
                     max_kernel_size=args.max_kernel_size,
                     padding_mode='zeros',
                     base=args.base,
                     perturb=args.perturb,
                     )

if args.cuda:
    model.cuda()
    permute = permute.cuda()

lr = args.lr
optimizers = [getattr(optim, args.optim)(model.model_weights(), lr=lr)]
if args.patch_conv:
    if args.arch_optim == 'SGD':
        optimizers.append(optim.SGD(model.arch_params(), lr=args.arch_lr, momentum=0.9))
    elif args.arch_optim == 'Adam':
        optimizers.append(optim.Adam(model.arch_params(), lr=args.arch_lr))
    else:
        raise(NotImplementedError)
optimizer = MixedOptimizer(optimizers)

writer = SummaryWriter(args.save_dir)
with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

def metrics(epoch):

    if args.patch_conv:
        mods = [m for m in model.modules() if hasattr(m, 'distance_from')]
        for mod in mods:
            mod.weight = mod.weight.cuda()
        for metric, metric_kwargs in [
                                      ('euclidean', {}),
                                      ('frobenius', {'approx': 16}),
                                      ('averaged', {'approx': 16, 'samples': 10}),
                                      ]:
            writer.add_scalar('/'.join(['conv', metric+'-dist']),
                              sum(m.distance_from('conv', metric=metric, relative=True, **metric_kwargs) for m in mods) / len(mods),
                              epoch)
            if not metric == 'averaged':
                writer.add_scalar('/'.join(['conv', metric+'-norm']),
                                  sum(getattr(m, metric)(**metric_kwargs) for m in mods) / len(mods),
                                  epoch)
        writer.add_scalar('conv/weight-norm', sum(m.weight.data.norm() for m in mods) / len(mods), epoch)


def train(ep):
    global steps
    train_loss, total_loss = 0, 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, seq_length)
        if args.permute:
            data = data[:, :, permute]
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.model_weights(), args.clip)
        optimizer.step()
        train_loss += loss
        total_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0
    
    writer.add_scalar('train/loss', total_loss.item() / len(train_loader), ep)


def test(ep):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, seq_length)
            if args.permute:
                data = data[:, :, permute]
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        writer.add_scalar('test/loss', test_loss, ep)
        writer.add_scalar('test/acc', 100. * correct / len(test_loader.dataset), ep)
        return test_loss


if __name__ == "__main__":
    metrics(0)
    for epoch in range(1, epochs+1):
        lr = optimizer.optimizers[0].param_groups[0]['lr']
        writer.add_scalar('hyper/lr', optimizer.optimizers[0].param_groups[0]['lr'], epoch)
        if args.patch_conv:
            writer.add_scalar('hyper/arch', optimizer.optimizers[1].param_groups[0]['lr'], epoch)
            for param_group in optimizer.optimizers[1].param_groups:
                if epoch <= args.warmup:
                    param_group['lr'] = 0.0
                    model.set_arch_requires_grad(False)
                elif epoch == args.warmup+1:
                    param_group['lr'] = args.arch_lr * lr / args.lr
                    model.set_arch_requires_grad(True)
        train(epoch)
        test(epoch)
        if epoch % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.
        if args.patch_conv and epoch == args.epochs - args.cooldown:
            for param_group in optimizer.optimizers[1].param_groups:
                param_group['lr'] = 0.0
            model.save_arch(os.path.join(args.save_dir, 'arch.th'))
            model.set_arch_requires_grad(False)
        metrics(epoch)

    writer.flush()
