#!/bin/sh

ORDERED=False
FOLDER=mask_compare

for DATASET in coco
do
    for DEVIATION_THRESHOLD in 0.4 0.5 0.6 0.7 0.8 0.9
    do
        for VIABLE_COUNTABILITY in 0
        do
            for ENSEMBLE_METHOD in average
            do
                for AVERAGE_ACCEPTABILITY in 0
                do
                    python run_evaluation.py "$ORDERED" -data "$DATASET" -e "$ENSEMBLE_METHOD" -c "$CREDIBILITY_THRESHOLD" \
                                                -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" -d "$DEVIATION_THRESHOLD" \
                                                > results_"$FOLDER"/"$DATASET"_"$ENSEMBLE_METHOD"_c="$CREDIBILITY_THRESHOLD"_v="$VIABLE_COUNTABILITY"_d="$DEVIATION_THRESHOLD"_a="$AVERAGE_ACCEPTABILITY".txt
                done
            done
        done
    done
done

for DATASET in coco
do
    for DEVIATION_THRESHOLD in 0.5
    do
        for VIABLE_COUNTABILITY in -1 0 1
        do
            for ENSEMBLE_METHOD in average
            do
                for AVERAGE_ACCEPTABILITY in 0
                do
                    python run_evaluation.py "$ORDERED" -data "$DATASET" -e "$ENSEMBLE_METHOD" -c "$CREDIBILITY_THRESHOLD" \
                                                -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" -d "$DEVIATION_THRESHOLD" \
                                                > results_"$FOLDER"/"$DATASET"_"$ENSEMBLE_METHOD"_c="$CREDIBILITY_THRESHOLD"_v="$VIABLE_COUNTABILITY"_d="$DEVIATION_THRESHOLD"_a="$AVERAGE_ACCEPTABILITY".txt
                done
            done
        done
    done
done

for DATASET in coco
do
    for DEVIATION_THRESHOLD in 0.5
    do
        for VIABLE_COUNTABILITY in 0
        do
            for ENSEMBLE_METHOD in average
            do
                for AVERAGE_ACCEPTABILITY in -1 1
                do
                    python run_evaluation.py "$ORDERED" -data "$DATASET" -e "$ENSEMBLE_METHOD" -c "$CREDIBILITY_THRESHOLD" \
                                                -v "$VIABLE_COUNTABILITY" -a "$AVERAGE_ACCEPTABILITY" -d "$DEVIATION_THRESHOLD" \
                                                > results_"$FOLDER"/"$DATASET"_"$ENSEMBLE_METHOD"_c="$CREDIBILITY_THRESHOLD"_v="$VIABLE_COUNTABILITY"_d="$DEVIATION_THRESHOLD"_a="$AVERAGE_ACCEPTABILITY".txt
                done
            done
        done
    done
done