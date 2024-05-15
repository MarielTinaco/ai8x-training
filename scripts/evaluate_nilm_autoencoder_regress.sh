#!/bin/sh
LOG_DIRECTORY="../ai8x-training/logs/2024.05.15-011246"

python train.py --model ai85nilmautoencoderregress --dataset UKDALE_small_autoencoder_regress --optimizer Adam --multitarget --batch-size 128 --evaluate --exp-load-weights-from $LOG_DIRECTORY/qat_best_quantized.pth.tar -8 --save-sample 1 --use-bias --device MAX78000 "$@"
