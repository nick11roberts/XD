#!/bin/bash

export PYTHONPATH=".."

DEPTH=3

for SEED in 1 2 3 ; do

  # backbone
  DIR=results/cifar10/lenet/backbone/$SEED
  python trainer.py --arch lenet --seed $SEED \
    --save-dir $DIR \
    --lr 0.01 

  # DARTS 
  LR=1E-1
  WARMUP=0
  PERTURB=0.1
  DIR=results/cifar10/lenet/darts/perturb$PERTURB/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch lenet --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --darts --perturb $PERTURB \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --lr 0.01

  # XD 
  LR=1E-4
  WARMUP=0
  DIR=results/cifar10/lenet/xd/depth$DEPTH/adam/$LR/warmup$WARMUP/$SEED
  python trainer.py --arch lenet --seed $SEED --save-dir $DIR \
    --patch-conv --patch-pool --kmatrix-depth $DEPTH \
    --arch-adam --arch-lr $LR --warmup-epochs $WARMUP \
    --lr 0.01

done
