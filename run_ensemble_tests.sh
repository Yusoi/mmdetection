#!/bin/sh

ORDERED=False

for DATASET in coco
do
    for CREDIBILITY_THRESHOLD in 0.55
    do
        for VIABLE_COUNTABILITY in 1
        do
            for AVERAGE_ACCEPTABILITY in -1 0 1
            do
                python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" \
                                            -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                            > outputs/aa_vc/results/v_"$VIABLE_COUNTABILITY"_a_"$AVERAGE_ACCEPTABILITY".txt
            done
        done
    done
done