"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
"""


import torch
import torch.fft
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io

import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, '..')
from chrysalis import Chrysalis
from nas import MixedOptimizer
import fire

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul3d(a, b):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    op = partial(torch.einsum, "bixyz,ioxyz->boxyz")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
# 3d fourier layers
################################################################

class SpectralConv3d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d_fast, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.rfft(x, 3, normalized=True, onesided=True)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        #Return to physical space
        x = torch.irfft(out_ft, 3, normalized=True, onesided=True, 
            signal_sizes=(x.size(-3), x.size(-2), x.size(-1)))
        return x

class SimpleBlock3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, arch):
        super(SimpleBlock3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)

        if arch == 'fno':
            self.conv0 = SpectralConv3d_fast(
                self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv1 = SpectralConv3d_fast(
                self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv2 = SpectralConv3d_fast(
                self.width, self.width, self.modes1, self.modes2, self.modes3)
            self.conv3 = SpectralConv3d_fast(
                self.width, self.width, self.modes1, self.modes2, self.modes3)
        #elif arch == 'xd':
        #    self.conv0 = xd([64, 64, 40], self.width, self.width, 
        #        arch_init='conv', 
        #        max_kernel_size=[self.modes1+1, self.modes1+1, self.modes1+1], 
        #        padding=2)
        #    self.conv1 = xd([64, 64, 40], self.width, self.width, 
        #        arch_init='conv', 
        #        max_kernel_size=[self.modes1+1, self.modes1+1, self.modes1+1], 
        #        padding=2)
        #    self.conv2 = xd([64, 64, 40], self.width, self.width, 
        #        arch_init='conv', 
        #        max_kernel_size=[self.modes1+1, self.modes1+1, self.modes1+1], 
        #        padding=2)
        #    self.conv3 = xd([64, 64, 40], self.width, self.width, 
        #        arch_init='conv', 
        #        max_kernel_size=[self.modes1+1, self.modes1+1, self.modes1+1], 
        #        padding=2)
        elif (arch == 'conv') or (arch == 'xd'):
            self.conv0 = nn.Conv3d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=4, 
                padding_mode='circular', bias=False)
            self.conv1 = nn.Conv3d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=4, 
                padding_mode='circular', bias=False)
            self.conv2 = nn.Conv3d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=4, 
                padding_mode='circular', bias=False)
            self.conv3 = nn.Conv3d(self.width, self.width, 
                kernel_size=self.modes1 + 1, padding=4, 
                padding_mode='circular', bias=False)
        else:
            raise NotImplementedError
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm3d(self.width)
        self.bn1 = torch.nn.BatchNorm3d(self.width)
        self.bn2 = torch.nn.BatchNorm3d(self.width)
        self.bn3 = torch.nn.BatchNorm3d(self.width)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y, size_z = x.shape[1], x.shape[2], x.shape[3]

        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)

        x1 = self.conv0(x)
        x2 = self.w0(
            x.view(batchsize, self.width, -1)
            ).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        x1 = self.conv1(x)
        x2 = self.w1(
            x.view(batchsize, self.width, -1)
            ).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn1(x1 + x2)
        x = F.relu(x)
        x1 = self.conv2(x)
        x2 = self.w2(
            x.view(batchsize, self.width, -1)
            ).view(batchsize, self.width, size_x, size_y, size_z)
        x = self.bn2(x1 + x2)
        x = F.relu(x)
        x1 = self.conv3(x)
        x2 = self.w3(
            x.view(batchsize, self.width, -1)).view(
                batchsize, self.width, size_x, size_y, size_z)
        x = self.bn3(x1 + x2)


        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net3d(nn.Module):
    def __init__(self, modes, width, arch='fno', visc=3):
        super(Net3d, self).__init__()

        """
        A wrapper function
        """

        if visc == 5:
            self.conv1 = SimpleBlock3d(modes, modes, 4, width, arch)
        else:
            self.conv1 = SimpleBlock3d(modes, modes, modes, width, arch)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

def main(arch='fno', epochs=500, sub=1, acc_steps=1, datapath='data/',
    arch_lr=0.0025, arch_momentum=0.0, arch_sgd=False, 
    warmup_epochs=0, cooldown_epochs=0, start_epoch=0,
    visc=3, large=False): 
    ################################################################
    # configs
    ################################################################

    ntrain = 1000
    ntest = 200

    modes = 8 #4 #8
    width = 20

    batch_size = 10 // acc_steps
    batch_size2 = batch_size

    epochs = epochs # 10
    learning_rate = 0.0025

    if large:
        scheduler_step = 40
        scheduler_gamma = 0.5
        epochs = 200
    else:
        scheduler_step = 100
        scheduler_gamma = 0.5
        epochs = 500

    print(epochs, learning_rate, scheduler_step, scheduler_gamma)

    path = 'test'
    # path = 'ns_fourier_V100_N'+str(ntrain)+'_ep' + str(epochs) + '_m' + str(modes) + '_w' + str(width)
    path_model = 'model/'+path
    path_train_err = 'results/'+path+'train.txt'
    path_test_err = 'results/'+path+'test.txt'
    path_image = 'image/'+path


    runtime = np.zeros(2, )
    t1 = default_timer()


    # TRAIN_PATH = 'data/ns_data_V1000_N1000_train.mat'
    # TEST_PATH = 'data/ns_data_V1000_N1000_train_2.mat'
    # TRAIN_PATH = 'data/ns_data_V1000_N5000.mat'
    # TEST_PATH = 'data/ns_data_V1000_N5000.mat'
    #TRAIN_PATH = 'data/ns_data_V100_N1000_T50_1.mat'
    #TEST_PATH = 'data/ns_data_V100_N1000_T50_2.mat'

    # TODO
    #sub = 1
    S = 64 #// sub
    T_in = 10 #// sub

    if visc == 3:
        TRAIN_PATH = datapath + 'ns_V1e-3_N5000_T50.mat'
        TEST_PATH = datapath + 'ns_V1e-3_N5000_T50.mat'
        T = 40 #// sub
    elif visc == 4 and not large:
        #TRAIN_PATH = datapath + 'ns_data_V1e-4_N20_T50_R256test.mat'
        #TEST_PATH = datapath + 'ns_data_V1e-4_N20_T50_R256test.mat'
        TRAIN_PATH = datapath + 'ns_V1e-4_N10000_T30.mat'
        TEST_PATH = datapath + 'ns_V1e-4_N10000_T30.mat'
        T = 20 #// sub
    elif visc == 4 and large:
        TRAIN_PATH = datapath + 'ns_V1e-4_N10000_T30.mat'
        TEST_PATH = datapath + 'ns_V1e-4_N10000_T30.mat'
        T = 20 #// sub
        ntrain = 10000
    elif visc == 5:
        TRAIN_PATH = datapath + 'NavierStokes_V1e-5_N1200_T20.mat'
        TEST_PATH = datapath + 'NavierStokes_V1e-5_N1200_T20.mat'
        T = 10 #// sub
    else:
        raise NotImplementedError

    ################################################################
    # load data
    ################################################################
    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('u')[:ntrain,::sub,::sub,:T_in]
    train_u = reader.read_field('u')[:ntrain,::sub,::sub,T_in:T+T_in]

    reader = MatReader(TEST_PATH)
    test_a = reader.read_field('u')[-ntest:,::sub,::sub,:T_in]
    test_u = reader.read_field('u')[-ntest:,::sub,::sub,T_in:T+T_in]

    print(train_u.shape)
    print(test_u.shape)
    assert (S == train_u.shape[-2])
    assert (T == train_u.shape[-1])


    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)
    train_u = y_normalizer.encode(train_u)

    train_a = train_a.reshape(ntrain,S,S,1,T_in).repeat([1,1,1,T,1])
    test_a = test_a.reshape(ntest,S,S,1,T_in).repeat([1,1,1,T,1])

    # pad locations (x,y,t)
    gridx = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S), dtype=torch.float)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1, T+1)[1:], dtype=torch.float)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])

    train_a = torch.cat((gridx.repeat([ntrain,1,1,1,1]), gridy.repeat([ntrain,1,1,1,1]),
                        gridt.repeat([ntrain,1,1,1,1]), train_a), dim=-1)
    test_a = torch.cat((gridx.repeat([ntest,1,1,1,1]), gridy.repeat([ntest,1,1,1,1]),
                        gridt.repeat([ntest,1,1,1,1]), test_a), dim=-1)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
    
    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    ################################################################
    # training and evaluation
    ################################################################
    model = Net3d(modes, width, arch, visc)
    # model = torch.load('model/ns_fourier_V100_N1000_ep100_m8_w20')

    # Convert to NAS search space, if applicable
    if arch == 'xd':
        X = torch.zeros([batch_size, S, S, T, 13])
        model, original = Chrysalis.metamorphosize(model), model

        arch_kwargs = {
            'kmatrix_depth': 1,
            'max_kernel_size': 1,
            'global_biasing': False, 
            'channel_gating': False,
            'base': 2,
        }

        named_modules = []
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.Conv3d):
                named_modules.append((name, layer))

        # Only patch conv3d
        model.patch_conv(X[:1], named_modules=named_modules, **arch_kwargs)
    else:
        arch_lr = 0.0


    cudnn.benchmark = True
    model.cuda()
    print(model)
    print(model.count_params())

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

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    def weight_sched(epoch):
        return scheduler_gamma ** (epoch // scheduler_step)
    
    def arch_sched(epoch):
        return 0.0 if (epoch < warmup_epochs) or (epoch > epochs-cooldown_epochs) else weight_sched(epoch)

    if arch == 'xd':
        sched_groups = [
            weight_sched if g['params'][0] in set(model.model_weights()) else arch_sched for g in optimizer.param_groups]
    else:
        sched_groups = [weight_sched]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=sched_groups, last_epoch=start_epoch-1)


    myloss = LpLoss(size_average=False)
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        optimizer.zero_grad()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            #optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out, y, reduction='mean')
            # mse.backward()

            y = y_normalizer.decode(y)
            out = y_normalizer.decode(out)
            l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            l2.backward()

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
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_mse /= len(train_loader)
        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print(ep, t2-t1, train_mse, train_l2, test_l2)
    # torch.save(model, path_model)


    pred = torch.zeros(test_u.shape)
    index = 0
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
    
    model.eval()
    torch.cuda.synchronize()

    t1 = default_timer()
    with torch.no_grad():
        for x, y in test_loader:
            test_l2 = 0
            x, y = x.cuda(), y.cuda()

            out = model(x)
            out = y_normalizer.decode(out)
            pred[index] = out

            test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
            print(index, test_l2)
            index = index + 1
    t2 = default_timer()
    print("Average inference time:", (t2-t1)/ntest)

    scipy.io.savemat('pred/'+path+'.mat', mdict={'pred': pred.cpu().numpy()})


if __name__ == '__main__':
    fire.Fire(main)