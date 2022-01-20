#!/bin/sh

ORDERED=False

for DATASET in coco
do
    for VIABLE_COUNTABILITY in 0
    do
        for AVERAGE_ACCEPTABILITY in -1 0 1
        do
            python run_evaluation.py "$ORDERED" -data "$DATASET" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                        > outputs/aa_vc/results/v="$VIABLE_COUNTABILITY"_a="$AVERAGE_ACCEPTABILITY".txt
        done
    done

    for VIABLE_COUNTABILITY in -1 1
    do
        for AVERAGE_ACCEPTABILITY in 0
        do
            python run_evaluation.py "$ORDERED" -data "$DATASET" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                        > outputs/aa_vc/results/v="$VIABLE_COUNTABILITY"_a="$AVERAGE_ACCEPTABILITY".txt
        done
    done
done