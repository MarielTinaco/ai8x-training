#!/bin/sh
# python train.py --epochs 2 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85kws20netv3 --dataset NILM --confusion --device MAX78000 "$@"

python train.py --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule_kws20.yaml --model ai85nilmnet --dataset UKDALE_small --multitarget --qat-policy policies/qat_policy_cnn1dnilm.yaml --use-bias --device MAX78000 "$@"
