#!/bin/sh

ORDERED=False
CREDIBILITY_THRESHOLD=0.5
DATASET=cityscapes

for ENSEMBLE_METHOD in average
do
    python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD"  \
                                -e "$ENSEMBLE_METHOD" > outputs/transfer_learning/results/"$ENSEMBLE_METHOD".txt
done