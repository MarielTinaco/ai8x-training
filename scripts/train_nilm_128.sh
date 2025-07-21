#!/bin/sh
python train.py --epochs 100 --batch-size 256 --optimizer Adam --lr 0.001 --wd 0.0001 --use-bias  --deterministic --model ai85netnilmseq2point128 --dataset UKDALE_128_stratified_compand_wide --enable-nilm --multitarget --compress policies/schedule-nilm.yaml --qat-policy policies/qat_policy_nilm.yaml --device MAX78000 --validation-split 0 "$@"
