#!/bin/sh

CONFIG_FILE=
CHECKPOINT_FILE=
RESULT_FILE=
EVAL_METRICS=

python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --out ${RESULT_FILE} \
    --eval ${EVAL_METRICS}
