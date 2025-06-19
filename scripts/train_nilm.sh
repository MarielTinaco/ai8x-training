#!/bin/sh
python train.py --epochs 100 --batch-size 256 --optimizer Adam --lr 0.001 --wd 0.001 --use-bias --deterministic --model ai85netnilmseq2point --dataset UKDALE --enable-nilm --multitarget --compress policies/schedule-nilm.yaml --qat-policy policies/qat_policy_nilm.yaml --device MAX78000 --validation-split 0 "$@"
