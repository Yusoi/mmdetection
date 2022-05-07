#!/bin/sh

ORDERED=False
CREDIBILITY_THRESHOLD=0.5
DATASET=coco
ENSEMBLE_METHOD=network


for CHECKPOINT in 2 4 6 8 10 12 14 16 18 20
do
    python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD"  \
                                -e "$ENSEMBLE_METHOD" -ck "work_dirs/unet2_bitwise_or_img_ensemble_reduced_do_40_/SGD_lrelu_BCELoss/epoch_$CHECKPOINT.pt" \
                                > outputs/network/results/"UNet2_e$CHECKPOINT".txt
done