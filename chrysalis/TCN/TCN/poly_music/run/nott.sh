#!/bin/bash

export PYTHONPATH="../../.."

OPTIM=Adam
WARMUP=0
LR=2E-3

for SEED in 1 2 3 ; do

  python music_test.py --ksize 6 --levels 4 --dropout 0.2 --clip 0.4 --seed $SEED --save-dir results/nott/backbone/$SEED

  python music_test.py --ksize 6 --levels 4 --dropout 0.2 --clip 0.4 --seed $SEED \
    --save-dir results/nott/xd/depth1/$OPTIM/$LR/warmup$WARMUP/$SEED \
    --patch-conv --causal-stack \
    --arch-optim $OPTIM --arch-lr $LR \
    --warmup $WARMUP --warmup-dir results/nott/backbone/$SEED

done
