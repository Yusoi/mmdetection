#!/bin/sh

ORDERED=False

for DATASET in coco
do
    for ENSEMBLE_METHOD in "network"
    do
        for CHECKPOINT in "work_dirs/simplenet_1_1/sigmoid_BCELoss/epoch_25.pt"
        do
        
            python run_evaluation.py "$ORDERED" -data "$DATASET" -e "$ENSEMBLE_METHOD" -ck "$CHECKPOINT" \
                                        > outputs/network/results/simplenet_1_1.txt
        done
    done
done