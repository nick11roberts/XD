#!/bin/bash

for ARCH in conv xd 
do
    python -u fourier_3d.py --arch=${ARCH} --visc=5 --datapath ${DATA} --warmup_epochs=0 --arch_lr=0.005 --arch_sgd=True --arch_momentum=0.5 |& tee results/fourier_3d_v5_${ARCH}.log

    python -u fourier_3d.py --arch=${ARCH} --visc=4 --datapath ${DATA} --warmup_epochs=0 --arch_lr=0.001 --arch_sgd=True --arch_momentum=0.5 |& tee results/fourier_3d_v5_${ARCH}.log
done
