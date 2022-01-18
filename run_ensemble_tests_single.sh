#!/bin/sh

for ORDERED in False
do
    for CREDIBILITY_THRESHOLD in 0.62 0.64 0.66 0.68
    do
    
        python run_evaluation.py "$ORDERED" -c "$CREDIBILITY_THRESHOLD" \
                                 > results_singles/c="$CREDIBILITY_THRESHOLD".txt

    done
done