#!/bin/bash

for SUB in 5 4 3 2 1
do 
    for ARCH in conv xd 
    do
        python -u fourier_2d.py --arch=${ARCH} --sub=${SUB} --datapath ${DATA} --warmup_epochs=0 --arch_lr=0.1 --arch_sgd=True --arch_momentum=0.5 |& tee results/fourier_2d_${ARCH}_${SUB}.log
    done
done
