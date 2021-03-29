#!/bin/bash

export PYTHONPATH=".."

DEPTH=3

for SEED in 1 2 3 ; do

  # backbone
  DIR=results/cifar10/resnet/backbone/$SEED
  python trainer.py --arch resnet20 --seed $SEED \
    --save-dir $DIR

  # DARTS 
  LR=1E-3
  WARMUP=0
  PERTURB=0.1
  DIR=results/cifar10/resnet/darts/perturb$PERTURB/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch resnet20 --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --darts --perturb $PERTURB \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP

  # XD 
  LR=1E-4
  WARMUP=50
  DIR=results/cifar10/resnet/xd/depth$DEPTH/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch resnet20 --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --kmatrix-depth $DEPTH \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP

done
