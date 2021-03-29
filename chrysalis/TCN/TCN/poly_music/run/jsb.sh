#!/bin/bash

export PYTHONPATH="../../.."

OPTIM=Adam
WARMUP=25
LR=2E-4

for SEED in 1 2 3 ; do

  python music_test.py --data JSB --ksize 2 --levels 3 --dropout 0.5 --clip 0.4 --seed $SEED --save-dir results/jsb/backbone/$SEED

  python music_test.py --data JSB --ksize 2 --levels 3 --dropout 0.5 --clip 0.4 --seed $SEED \
    --save-dir results/jsb/xd/depth1/$OPTIM/$LR/warmup$WARMUP/$SEED \
    --patch-conv --causal-stack \
    --arch-optim $OPTIM --arch-lr $LR \
    --warmup $WARMUP --warmup-dir results/jsb/backbone/$SEED

done
