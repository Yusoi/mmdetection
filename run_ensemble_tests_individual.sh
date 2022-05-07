#!/bin/sh

ORDERED=False
DATASET=coco_filtered

for CREDIBILITY_THRESHOLD in 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.70 0.75 0.80 0.85 0.9 0.95 
do
    python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" \
                                    > outputs/ct/results/ct_"$CREDIBILITY_THRESHOLD".txt
done