#!/bin/bash

export PYTHONPATH="../../.."

for SEED in 1 2 3 ; do

  python word_cnn_test.py --dropout 0.5 --clip 0.4  --seed $SEED --save-dir results/ptb/backbone/$SEED

done
