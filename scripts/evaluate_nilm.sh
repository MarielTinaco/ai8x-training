#!/bin/sh
LOG_DIRECTORY="../ai8x-training/logs/2024.04.23-091340"

python train.py --model ai85nilmnet --dataset UKDALE_small --optimizer Adam --multitarget --evaluate --exp-load-weights-from $LOG_DIRECTORY/qat_best_quantized.pth.tar -8 --save-sample 1 --use-bias --device MAX78000 "$@"
