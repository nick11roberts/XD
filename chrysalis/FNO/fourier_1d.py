"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *

import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, '..')
from chrysalis import Chrysalis
from nas import MixedOptimizer
import fire

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 1, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.irfft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width, arch='xd', s=1024):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        if arch == 'fno':
            self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
            self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        elif (arch == 'conv') or (arch == 'xd'):
            self.conv0 = nn.Conv1d(self.width, self.width, 
                kernel_size=self.modes1+1, padding=8, 
                padding_mode='circular', bias=False)
            self.conv1 = nn.Conv1d(self.width, self.width, 
                kernel_size=self.modes1+1, padding=8, 
                padding_mode='circular', bias=False)
            self.conv2 = nn.Conv1d(self.width, self.width, 
                kernel_size=self.modes1+1, padding=8, 
                padding_mode='circular', bias=False)
            self.conv3 = nn.Conv1d(self.width, self.width, 
                kernel_size=self.modes1+1, padding=8, 
                padding_mode='circular', bias=False)
        else:
            raise NotImplementedError

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm1d(self.width)
        self.bn1 = torch.nn.BatchNorm1d(self.width)
        self.bn2 = torch.nn.BatchNorm1d(self.width)
        self.bn3 = torch.nn.BatchNorm1d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net1d(nn.Module):
    def __init__(self, modes, width, arch='fno', s=1024):
        super(Net1d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock1d(modes, width, arch, s)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()

    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c


def main(sub=2**3, arch='fno', epochs=500, acc_steps=1, datapath='data/',
    arch_lr=0.001, arch_momentum=0.0, arch_sgd=False, 
    warmup_epochs=0, cooldown_epochs=0, start_epoch=0):
    ################################################################
    #  configurations
    ################################################################
    ntrain = 1000
    ntest = 100

    print(sub, arch)

    #sub = 1 #subsampling rate
    #h = 2**10 // sub
    #s = h
    #sub = 2**3 #subsampling rate
    h = 2**13 // sub
    s = h

    batch_size = 20 // acc_steps
    learning_rate = 0.001

    epochs = epochs
    step_size = 100
    gamma = 0.5

    modes = 16
    width = 64


    ################################################################
    # read data
    ################################################################
    dataloader = MatReader(datapath + 'burgers_data_R10.mat')
    x_data = dataloader.read_field('a')[:,::sub]
    y_data = dataloader.read_field('u')[:,::sub]

    x_train = x_data[:ntrain,:]
    y_train = y_data[:ntrain,:]
    x_test = x_data[-ntest:,:]
    y_test = y_data[-ntest:,:]

    # cat the locations information
    grid = np.linspace(0, 2*np.pi, s).reshape(1, s, 1)
    grid = torch.tensor(grid, dtype=torch.float)
    x_train = torch.cat([x_train.reshape(ntrain,s,1), grid.repeat(ntrain,1,1)], dim=2)
    x_test = torch.cat([x_test.reshape(ntest,s,1), grid.repeat(ntest,1,1)], dim=2)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # model
    model = Net1d(modes, width, arch, s)

    # Convert to NAS search space, if applicable
    if arch == 'xd':
        X = torch.zeros([batch_size, s, 2
        ])
        model, original = Chrysalis.metamorphosize(model), model

        arch_kwargs = {
            'kmatrix_depth': 1,
            'max_kernel_size': 1,
            'global_biasing': False, 
            'channel_gating': False,
            'base': 2,
        }

        # TODO filter out skip connect convs
        named_modules = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv1d) and '.conv' in name:
                named_modules.append((name, layer))

        # Only patch non-skip connect conv1d
        model.patch_conv(X[:1], named_modules=named_modules, **arch_kwargs)
    else:
        arch_lr = 0.0

    cudnn.benchmark = True
    model.cuda()
    print(model)
    print(model.count_params())


    ################################################################
    # training and evaluation
    ################################################################
    if arch == 'xd':
        momentum = partial(torch.optim.SGD, momentum=arch_momentum)
        arch_opt = momentum if arch_sgd else torch.optim.Adam
        opts = [torch.optim.Adam([{'params': list(model.xd_weights())},
                          {'params': list(model.nonxd_weights())}], 
                          lr=learning_rate, weight_decay=1e-4),
                arch_opt([{'params': list(model.arch_params())}],
                        lr=arch_lr, weight_decay=1e-4)]
    else:
        opts = [torch.optim.Adam(model.parameters(), 
            lr=learning_rate, weight_decay=1e-4)]
    optimizer = MixedOptimizer(opts, op_decay=None)

    #weight_sched = torch.optim.lr_scheduler.StepLR(optimizer, 
    #    step_size=step_size, gamma=gamma)

    def weight_sched(epoch):
        return gamma ** (epoch // step_size)
    
    def arch_sched(epoch):
        return 0.0 if (epoch < warmup_epochs) or (epoch > epochs-cooldown_epochs) else weight_sched(epoch)

    if arch == 'xd':
        sched_groups = [
            weight_sched if g['params'][0] in set(model.model_weights()) else arch_sched for g in optimizer.param_groups]
    else:
        sched_groups = [weight_sched]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=sched_groups, last_epoch=start_epoch-1)


    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    myloss = LpLoss(size_average=False)
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward() # use the l2 relative loss

            if (i + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            #optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)

    # torch.save(model, 'model/ns_fourier_burgers_8192')
    pred = torch.zeros(y_test.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1

    # scipy.io.savemat('pred/burger_test.mat', mdict={'pred': pred.cpu().numpy()})

if __name__ == '__main__':
    fire.Fire(main)