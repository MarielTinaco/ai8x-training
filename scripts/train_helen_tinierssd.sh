#!/bin/sh
python train.py --deterministic --print-freq 200 --epochs 300 --optimizer Adam --lr 0.1e-5 --model ai85tinierssd --use-bias --momentum 0.9 --weight-decay 5e-4 --dataset HELEN_74 --device MAX78000 --obj-detection --obj-detection-params parameters/obj_detection_params.yaml --batch-size 128 --qat-policy policies/qat_policy_svhn.yaml --enable-tensorboard --validation-split 0"$@"
