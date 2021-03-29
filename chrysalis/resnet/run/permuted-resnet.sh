#!/bin/bash

export PYTHONPATH=".."

DEPTH=3

for SEED in 1 2 3 ; do

  # backbone
  DIR=results/permuted/resnet/backbone/$SEED
  python trainer.py --arch resnet20 --seed $SEED \
    --save-dir $DIR \
    --permute

  # DARTS 
  LR=1E-1
  WARMUP=0
  PERTURB=0.875
  DIR=results/permuted/resnet/darts/perturb$PERTURB/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch resnet20 --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --darts --perturb $PERTURB \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --permute

  # XD 
  LR=1E-3
  WARMUP=0
  DIR=results/permuted/resnet/xd/depth$DEPTH/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch resnet20 --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --kmatrix-depth $DEPTH \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --permute

done
