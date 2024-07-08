#!/bin/sh
python train.py --epochs 200 --optimizer Adam --lr 0.001 --deterministic --compress policies/schedule-nilm-slidingwindow-resnet.yaml --model ai85nilmslidingwindowressimplenet --batch-size 256 --dataset UKDALE_small_sliding_window --multitarget --qat-policy policies/qat_policy_nilm_resnet.yaml --enable-tensorboard --use-bias --device MAX78000 "$@"