#!/bin/sh

ORDERED=False
CREDIBILITY_THRESHOLD=0.5
DATASET=coco

for ENSEMBLE_METHOD in bitwise_or
do
    python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD"  \
                                -e "$ENSEMBLE_METHOD" > outputs/aa_vc/results/"$ENSEMBLE_METHOD".txt
done