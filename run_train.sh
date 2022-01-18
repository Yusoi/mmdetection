#!/bin/sh

CONFIG_FILE="configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py"

python tools/train.py \
    "${CONFIG_FILE}"
