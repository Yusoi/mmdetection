#!/bin/sh

ORDERED=False
CREDIBILITY_THRESHOLD=0.5
DATASET=coco

for ENSEMBLE_METHOD in average
do
    for VIABLE_COUNTABILITY in -1 0 1
    do
        for AVERAGE_ACCEPTABILITY in 0
        do
            python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                        -e "$ENSEMBLE_METHOD" > outputs/aa_vc/results/"$ENSEMBLE_METHOD"_v"$VIABLE_COUNTABILITY"_a"$AVERAGE_ACCEPTABILITY".txt
        done
    done

    for AVERAGE_ACCEPTABILITY in -1 1
    do
        for VIABLE_COUNTABILITY in 0
        do
            python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD" -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" \
                                        -e "$ENSEMBLE_METHOD" > outputs/aa_vc/results/"$ENSEMBLE_METHOD"_v"$VIABLE_COUNTABILITY"_a"$AVERAGE_ACCEPTABILITY".txt
        done
    done
done

for ENSEMBLE_METHOD in bitwise_or bitwise_and
do
    python run_evaluation.py "$ORDERED" -data "$DATASET" -c "$CREDIBILITY_THRESHOLD"  \
                                -e "$ENSEMBLE_METHOD" > outputs/aa_vc/results/"$ENSEMBLE_METHOD".txt
done