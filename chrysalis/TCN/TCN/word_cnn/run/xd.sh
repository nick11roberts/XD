#!/bin/bash

export PYTHONPATH="../../.."

OPTIM=Adam
LR=2E-6
WARMUP=0

for SEED in 1 2 3 ; do

  python word_cnn_test.py --dropout 0.5 --clip 0.4  --seed $SEED \
    --save-dir results/ptb/xd/depth1/$OPTIM/$LR/warmup$WARMUP/$SEED \
    --patch-conv --causal-stack --accumulation-rounds 2 \
    --arch-optim $OPTIM --arch-lr $LR

done
