#!/bin/sh
python train.py --model ai85netnilmseq2point128 --dataset UKDALE_128 --enable-nilm --multitarget --deterministic --optimizer Adam --use-bias --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-nilm-seq2point-128-qat8-q.pth.tar -8 --device MAX78000 --save-sample 1 --compiler-mode none "$@"
