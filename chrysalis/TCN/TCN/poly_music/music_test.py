import argparse
import json
import os
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.poly_music.model import TCN
from TCN.poly_music.utils import data_generator
import numpy as np
from tensorboardX import SummaryWriter
from torch.nn import functional as F

from chrysalis import Chrysalis
from nas import MixedOptimizer, clip_grad_norm


parser = argparse.ArgumentParser(description='Sequence Modeling - Polyphonic Music')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.25,
                    help='dropout applied to layers (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.2,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--data', type=str, default='Nott',
                    help='the dataset to run (default: Nott)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--patch-conv', action='store_true')
parser.add_argument('--kmatrix-depth', type=int, default=1)
parser.add_argument('--max-kernel-size', type=int, default=1)
parser.add_argument('--base', type=int, default=2)
parser.add_argument('--perturb', type=float, default=0.0)
parser.add_argument('--arch-optim', type=str, default='Adam')
parser.add_argument('--arch-lr', type=float, default=2E-3)
parser.add_argument('--save-dir', type=str, default='test')
parser.add_argument('--cooldown', type=int, default=0)
parser.add_argument('--causal-stack', action='store_true')
parser.add_argument('--history', type=int, default=40)
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--warmup-dir', type=str, default='')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
input_size = 88
X_train, X_valid, X_test = data_generator(args.data)

n_channels = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(input_size, input_size, n_channels, kernel_size, dropout=args.dropout)
if args.warmup and args.warmup_dir:
    for param, old in zip(model.parameters(),
                          torch.load(os.path.join(args.warmup_dir, 'model'+str(args.warmup)+'.pt')).cpu().parameters()):
        param.data = old.data

model = Chrysalis.metamorphosize(model, in_place=True, attrs=['forward'])
if args.patch_conv:
    model.patch_conv(
                     torch.zeros(1, args.history, input_size),
                     verbose=True,
                     kmatrix_depth=args.kmatrix_depth,
                     max_kernel_size=args.max_kernel_size,
                     padding_mode='zeros',
                     base=args.base,
                     perturb=args.perturb,
                     )

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
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
                                      #('frobenius', {'approx': 4}),
                                      #('averaged', {'approx': 4, 'samples': 5}),
                                      ]:
            writer.add_scalar('/'.join(['conv', metric+'-dist']),
                              sum(m.distance_from('conv', metric=metric, relative=True, **metric_kwargs) for m in mods) / len(mods),
                              epoch)
            if not metric == 'averaged':
                writer.add_scalar('/'.join(['conv', metric+'-norm']),
                                  sum(getattr(m, metric)(**metric_kwargs) for m in mods) / len(mods),
                                  epoch)
        writer.add_scalar('conv/weight-norm', sum(m.weight.data.norm() for m in mods) / len(mods), epoch)

def evaluate(X_data, name='Eval'):
    model.eval()
    eval_idx_list = np.arange(len(X_data), dtype="int32")
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx in eval_idx_list:
            data_line = X_data[idx]
            x, y = Variable(data_line[:-1]), Variable(data_line[1:])
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            if args.causal_stack:
                causal_stack = torch.vstack([F.pad(x[None,max(0, j-args.history):j], (0, 0, max(0, args.history-j), 0)) for j in range(1, len(x)+1)])
                output = model(causal_stack)[:,-1]
            else:
                output = model(x.unsqueeze(0)).squeeze(0)
            loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                                torch.matmul((1-y), torch.log(1-output).float().t()))
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss


def train(ep):
    model.train()
    train_loss, total_loss = 0, 0
    count, total_count = 0, 0
    train_idx_list = np.arange(len(X_train), dtype="int32")
    np.random.shuffle(train_idx_list)
    for idx in train_idx_list:
        data_line = X_train[idx]
        x, y = Variable(data_line[:-1]), Variable(data_line[1:])
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        if args.causal_stack:
            causal_stack = torch.vstack([F.pad(x[None,max(0, j-args.history):j], (0, 0, max(0, args.history-j), 0)) for j in range(1, len(x)+1)])
            output = model(causal_stack)[:,-1]
        else:
            output = model(x.unsqueeze(0)).squeeze(0)
        loss = -torch.trace(torch.matmul(y, torch.log(output).float().t()) +
                            torch.matmul((1 - y), torch.log(1 - output).float().t()))
        train_loss += loss.item()
        total_loss += loss
        count += output.size(0)
        total_count += output.size(0)

        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.model_weights(), args.clip)
        optimizer.step()
        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = train_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, optimizer.optimizers[0].param_groups[0]['lr'], cur_loss))
            train_loss = 0.0
            count = 0

    writer.add_scalar('train/loss', total_loss.item() /total_count, ep)


if __name__ == "__main__":
    model_name = "poly_music_{0}.pt".format(args.data)
    if args.warmup and args.warmup_dir:
        loaded = torch.load(os.path.join(args.warmup_dir, 'optim'+str(args.warmup)+'.pt'))
        optimizer.optimizers[0].load_state_dict(loaded['optimizer'].state_dict())
        vloss_list = loaded['vloss_list'][:args.warmup]
        best_vloss = min(vloss_list)
        eloss = loaded['eloss']
        start_epoch = args.warmup+1
    else:
        metrics(0)
        best_vloss = 1e8
        vloss_list = []
        start_epoch = 1

    eloss = float('inf')
    for ep in range(start_epoch, args.epochs+1):
        lr = optimizer.optimizers[0].param_groups[0]['lr']
        writer.add_scalar('hyper/lr', optimizer.optimizers[0].param_groups[0]['lr'], ep)
        if args.patch_conv:
            writer.add_scalar('hyper/arch', optimizer.optimizers[1].param_groups[0]['lr'], ep)
            for param_group in optimizer.optimizers[1].param_groups:
                if ep <= args.warmup:
                    param_group['lr'] = 0.0
                    model.set_arch_requires_grad(False)
                elif ep == args.warmup+1:
                    param_group['lr'] = args.arch_lr * lr / args.lr
                    model.set_arch_requires_grad(True)
        train(ep)
        vloss = evaluate(X_valid, name='Validation')
        tloss = evaluate(X_test, name='Test')
        writer.add_scalar('valid/loss', vloss, ep)
        writer.add_scalar('test/loss', tloss, ep)
        metrics(ep)

        if np.isnan(vloss):
            break

        model.train()
        if vloss < best_vloss:
            torch.save(model, os.path.join(args.save_dir, model_name))
            print("Saved model!\n")
            best_vloss = vloss
            eloss = tloss
        if ep > start_epoch+9 and vloss > max(vloss_list[-3:]):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.
        if args.patch_conv and ep == args.epochs - args.cooldown:
            for param_group in optimizer.optimizers[1].param_groups:
                param_group['lr'] = 0.0
            model.save_arch(os.path.join(args.save_dir, 'arch.th'))
            model.set_arch_requires_grad(False)

        if ep in {10, 25, 50, 75, 100}:
            torch.save(model, os.path.join(args.save_dir, 'model'+str(ep)+'.pt'))
            torch.save({'optimizer': optimizer, 
                        'vloss_list': vloss_list, 
                        'best_vloss': best_vloss,
                        'eloss': eloss},
                        os.path.join(args.save_dir, 'optim'+str(ep)+'.pt'))

        vloss_list.append(vloss)

    print('-' * 89)
    print("Eval loss: {:.5f}".format(eloss))
    writer.add_scalar('eval/loss', eloss, ep)
    writer.flush()
