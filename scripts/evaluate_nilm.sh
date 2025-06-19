#!/bin/sh
python train.py --model ai85netnilmseq2point --dataset UKDALE --enable-nilm --multitarget --deterministic --optimizer Adam --use-bias --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-nilm-seq2point-qat8-q.pth.tar -8 --device MAX78000 "$@"
