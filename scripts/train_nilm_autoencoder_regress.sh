#!/bin/sh

python train.py --epochs 400 --optimizer Adam --lr 0.001 --deterministic --multitarget --model ai85nilmautoencoderregress --batch-size 256 --dataset UKDALE_small_autoencoder_regress --multitarget --qat-policy policies/qat_policy_nilm_autoencoder.yaml --enable-tensorboard --use-bias --device MAX78000 "$@"