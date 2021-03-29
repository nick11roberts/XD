import argparse
import json
import os
import pdb
import time
import math
from glob import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.word_cnn.utils import *
from TCN.word_cnn.model import *
import pickle
from random import randint
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

from chrysalis import Chrysalis
from nas import MixedOptimizer, clip_grad_norm


parser = argparse.ArgumentParser(description='Sequence Modeling - Word-level Language Modeling')

parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (default: 0.45)')
parser.add_argument('--emb_dropout', type=float, default=0.25,
                    help='dropout applied to the embedded layer (default: 0.25)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip, -1 means no clip (default: 0.35)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus (default: ./data/penn)')
parser.add_argument('--emsize', type=int, default=600,
                    help='size of word embeddings (default: 600)')
parser.add_argument('--levels', type=int, default=4,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=600,
                    help='number of hidden units per layer (default: 600)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the encoder-decoder weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=40,
                    help='valid sequence length (default: 40)')
parser.add_argument('--seq_len', type=int, default=80,
                    help='total sequence length, including effective history (default: 80)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
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
parser.add_argument('--offline', type=str, default='')
parser.add_argument('--warmup', type=int, default=0)
parser.add_argument('--warmup-dir', type=str, default='')
parser.add_argument('--accumulation-rounds', type=int, default=1)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

print(args)
corpus = data_generator(args)
eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, eval_batch_size, args)


n_words = len(corpus.dictionary)

num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied
model = TCN(args.emsize, n_words, num_chans, dropout=dropout, emb_dropout=emb_dropout, kernel_size=k_size, tied_weights=tied)
model = Chrysalis.metamorphosize(model, in_place=True, attrs=['forward'])
if args.warmup and args.warmup_dir:
    for param, old in zip(model.parameters(),
                          torch.load(os.path.join(args.warmup_dir, 'model'+str(args.warmup)+'.pt')).cpu().parameters()):
        param.data = old.data

if args.patch_conv:
    data, _ = get_batch(train_data, 0, args)
    model.patch_conv(
                     data[:1,:args.seq_len-args.validseqlen].cpu(),
                     verbose=True,
                     kmatrix_depth=args.kmatrix_depth,
                     max_kernel_size=args.max_kernel_size,
                     padding_mode='zeros',
                     base=args.base,
                     perturb=args.perturb,
                     )

if args.offline:
    model.load_arch(args.offline, verbose=True)
if args.cuda:
    model.cuda()

# May use adaptive softmax to speed up training
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
if args.offline:
    for param_group in optimizer.optimizers[1].param_groups:
        param_group['lr'] = 0.0
    model.set_arch_requires_grad(False)

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

def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    with torch.no_grad():
        for i in range(0, data_source.size(1) - 1, args.validseqlen):
            if i + args.seq_len - args.validseqlen >= data_source.size(1) - 1:
                continue
            data, targets = get_batch(data_source, i, args, evaluation=True)

            # Discard the effective history, just like in training
            eff_history = args.seq_len - args.validseqlen
            final_target = targets[:, eff_history:].contiguous().view(-1)
            if args.causal_stack:
                batchsize = data.shape[0]
                valseqlen = min(args.validseqlen, data.shape[1] - eff_history)
                causal_stack = torch.vstack([data[:,j:j+eff_history] for j in range(1, valseqlen+1)]).reshape(valseqlen, batchsize, eff_history)
                final_output = model(causal_stack.permute(1, 0, 2).reshape(len(final_target), eff_history))[:,-1].contiguous()
            else:
                output = model(data)
                final_output = output[:, eff_history:].contiguous().view(-1, n_words)
            loss = criterion(final_output, final_target)

            # Note that we don't add TAR loss here
            total_loss += (data.size(1) - eff_history) * loss.item()
            processed_data_size += data.size(1) - eff_history

        return total_loss / processed_data_size


def train(ep):
    # Turn on training mode which enables dropout.
    global train_data
    model.train()
    train_loss, total_loss = 0, 0
    start_time = time.time()
    for batch_idx, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        optimizer.zero_grad()

        # Discard the effective history part
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        if args.causal_stack:
            valseqlen = min(args.validseqlen, data.shape[1] - eff_history)
            #causal_stack = torch.vstack([data[:,j:j+eff_history] for j in range(1, valseqlen+1)]).reshape(valseqlen, args.batch_size, eff_history)
            #final_output = model(causal_stack.permute(1, 0, 2).reshape(len(final_target), eff_history))[:,-1].contiguous()
            causal_stack = torch.vstack([data[:,j:j+eff_history] for j in range(1, valseqlen+1)]).reshape(valseqlen, args.batch_size, eff_history).permute(1, 0, 2).reshape(len(final_target), eff_history)
            span = len(final_target) // args.accumulation_rounds
            intervals = [(offset, offset+span) for offset in range(0, len(final_target), span)]
            if len(intervals) > args.accumulation_rounds:
                intervals = intervals[:-1]
                intervals[-1] = (intervals[-1][0], len(final_target))
            for a, b in intervals:
                loss = criterion(model(causal_stack[a:b])[:,-1].contiguous(), final_target[a:b]) / args.accumulation_rounds
                loss.backward()
                train_loss += loss.item()
                total_loss += loss.item()
        else:
            output = model(data)
            final_output = output[:, eff_history:].contiguous().view(-1, n_words)
            loss = criterion(final_output, final_target)
            loss.backward()
            train_loss += loss.item()
            total_loss += loss.item()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.model_weights(), args.clip)
        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, optimizer.optimizers[0].param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            train_loss = 0
            start_time = time.time()

    writer.add_scalar('train/loss', total_loss / (batch_idx+1.), ep)


def load_sched(folder, epoch):

    acc = EventAccumulator(glob(os.path.join(args.warmup_dir, 'events.out*'))[0]).Reload()
    return [e.value for e in acc.Scalars('valid/loss')][:epoch], [e.value for e in acc.Scalars('hyper/lr')][epoch]


if __name__ == "__main__":
    best_vloss = 1e8

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if args.warmup and args.warmup_dir:
            all_vloss, lr = load_sched(args.warmup_dir, args.warmup)
            for param_group in optimizer.optimizers[0].param_groups:
                param_group['lr'] = lr
            best_vloss = min(all_vloss)
            start_epoch = args.warmup+1
        else:
            metrics(0)
            all_vloss = []
            start_epoch = 1

        for epoch in range(start_epoch, args.epochs+1):
            lr = optimizer.optimizers[0].param_groups[0]['lr']
            writer.add_scalar('hyper/lr', lr, epoch)
            if args.patch_conv:
                for param_group in optimizer.optimizers[1].param_groups:
                    if epoch <= args.warmup:
                        param_group['lr'] = 0.0
                        model.set_arch_requires_grad(False)
                    elif epoch == args.warmup+1:
                        param_group['lr'] = args.arch_lr * lr / args.lr
                        model.set_arch_requires_grad(True)
                writer.add_scalar('hyper/arch', optimizer.optimizers[1].param_groups[0]['lr'], epoch)
            epoch_start_time = time.time()
            train(epoch)
            val_loss = evaluate(val_data)
            test_loss = evaluate(test_data)
            writer.add_scalar('valid/loss', val_loss, epoch)
            writer.add_scalar('valid/ppl', math.exp(val_loss), epoch)
            writer.add_scalar('test/loss', test_loss, epoch)
            writer.add_scalar('test/ppl', math.exp(test_loss), epoch)
            metrics(epoch)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)

            model.train()
            # Save the model if the validation loss is the best we've seen so far.
            if val_loss < best_vloss:
                print('Save model!\n')
                torch.save(model, os.path.join(args.save_dir, 'model.pt'))
                best_vloss = val_loss

            if epoch in {25, 50, 75, 100}:
                torch.save(model, os.path.join(args.save_dir, 'model'+str(epoch)+'.pt'))

            # Anneal the learning rate if the validation loss plateaus
            if epoch > start_epoch+4 and val_loss >= max(all_vloss[-5:]):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / 2.
            if args.patch_conv and epoch == args.epochs - args.cooldown:
                for param_group in optimizer.optimizers[1].param_groups:
                    param_group['lr'] = 0.0
                model.save_arch(os.path.join(args.save_dir, 'arch.th'))
                model.set_arch_requires_grad(False)
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    model.save_arch(os.path.join(args.save_dir, 'arch.th'))

    # Load the best saved model.
    for param, old in zip(model.parameters(),
                          torch.load(os.path.join(args.save_dir, 'model.pt')).parameters()):
        param.data = old.data

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    writer.flush()
