#!/bin/sh

for ORDERED in False
do
        python run_evaluation.py "$ORDERED" \
                                 > outputs/single_networks/results/results.txt
done