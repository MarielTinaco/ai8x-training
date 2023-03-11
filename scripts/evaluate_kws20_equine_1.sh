#!/bin/sh
python train.py --model ai85kws20kabayo --dataset EQUINE --confusion --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/qat_equine_1_best-q.pth.tar -8 --device MAX78000 "$@"
