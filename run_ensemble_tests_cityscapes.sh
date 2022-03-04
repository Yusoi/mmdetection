#!/bin/sh

ORDERED=False
CREDIBILITY_THRESHOLD=0.5
DATASET=cityscapes

for VIABLE_COUNTABILITY in -1 0 1
do
    for AVERAGE_ACCEPTABILITY in 0
    do
        python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                    > outputs/cityscapes/results/v"$VIABLE_COUNTABILITY"_a"$AVERAGE_ACCEPTABILITY".txt
    done
done

for AVERAGE_ACCEPTABILITY in -1 1
do
    for VIABLE_COUNTABILITY in 0
    do
        python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                    > outputs/cityscapes/results/v"$VIABLE_COUNTABILITY"_a"$AVERAGE_ACCEPTABILITY".txt
    done
done