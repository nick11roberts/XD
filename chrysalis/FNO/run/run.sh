#!/bin/bash

# Use --arch={xd, fno, conv} for the desired architecture

# 1D
python -u python fourier_1d.py --arch=xd --sub=32 --datapath <path_to_data> |& tee <results_file>

# 2D
#python -u python fourier_2d.py --arch=xd --sub=5 --datapath <path_to_data> |& tee <results_file>

# 3D
#python -u python fourier_3d.py --arch=xd --visc=5 --datapath <path_to_data> |& tee <results_file>

#python -u python fourier_3d.py --arch=xd --visc=4 --datapath <path_to_data> |& tee <results_file>
