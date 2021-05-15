#!/bin/bash

for SUB in 32 16 8 4 2 1
do 
    for ARCH in conv xd 
    do
        python -u fourier_1d.py --arch=${ARCH} --sub=${SUB} --datapath ${DATA} --warmup_epochs=0 --arch_lr=0.001 |& tee results/fourier_1d_${ARCH}_${SUB}.log
    done
done
