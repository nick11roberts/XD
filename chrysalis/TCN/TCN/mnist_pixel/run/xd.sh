#!/bin/bash

export PYTHONPATH="../../.."

DEPTH=3
OPTIM=Adam
LR=2E-4
WARMUP=0

for SEED in 1 2 3 ; do

  python pmnist_test.py --dropout 0.0 --permute --seed $SEED \
    --save-dir results/permute/xd/depth$DEPTH/$OPTIM/$LR/warmup$WARMUP/$SEED \
    --patch-conv --warmup $WARMUP \
    --arch-optim $OPTIM --arch-lr $LR --kmatrix-depth $DEPTH

done
