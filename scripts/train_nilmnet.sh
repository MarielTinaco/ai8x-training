#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.001 --deterministic --compress policies/schedule-nilm.yaml --multitarget --model ai85nilmnet --batch-size 256 --dataset UKDALE_small --multitarget --qat-policy policies/qat_policy_cnn1dnilm.yaml --enable-tensorboard --use-bias --device MAX78000 "$@"