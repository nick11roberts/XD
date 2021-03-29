#!/bin/bash

export PYTHONPATH=".."

DEPTH=3

for SEED in 1 2 3 ; do

  # backbone
  DIR=results/permuted/lenet/backbone/$SEED
  python trainer.py --arch lenet --seed $SEED \
    --save-dir $DIR \
    --lr 0.01 --permute

  # DARTS 
  LR=1E-1
  WARMUP=50
  PERTURB=0.875
  DIR=results/permuted/lenet/darts/perturb$PERTURB/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch lenet --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --darts --perturb $PERTURB \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --lr 0.01 --permute

  # XD 
  LR=1E-3
  WARMUP=0
  DIR=results/permuted/lenet/xd/depth$DEPTH/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch lenet --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --kmatrix-depth $DEPTH \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --lr 0.01 --permute

done
