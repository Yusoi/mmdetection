#!/bin/sh

ORDERED=False

for DATASET in coco
do
    for CREDIBILITY_THRESHOLD in 0.55
    do
            python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" \
                                        > outputs/ct/results/"$CREDIBILITY_THRESHOLD".txt
    done
done