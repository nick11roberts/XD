#!/bin/bash

export PYTHONPATH="../../.."

for SEED in 1 2 3 ; do

  python pmnist_test.py --dropout 0.0 --permute --seed $SEED --save-dir results/permute/backbone/$SEED

done
