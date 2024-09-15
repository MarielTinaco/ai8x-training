#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress policies/schedule-nilm-seq2point.yaml --model ai85nilmseq2pointmod --batch-size 256 --dataset UKDALE_small_512_regress --multitarget --qat-policy policies/qat_policy_nilm_seq2point.yaml --enable-tensorboard --use-bias --device MAX78000 "$@"