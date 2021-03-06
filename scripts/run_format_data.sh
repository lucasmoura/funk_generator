#!/bin/bash

#usage: ./scripts/run_format_data.sh

set -e

DATA_FOLDER='data'
DATASET_SAVE_PATH='song_dataset'
MIN_FREQUENCY=5
VALIDATION_PERCENT=0.0
TEST_PERCENT=0.0


python format_data.py \
    --data-folder=${DATA_FOLDER} \
    --dataset-save-path=${DATASET_SAVE_PATH} \
    --min-frequency=${MIN_FREQUENCY} \
    --validation-percent=${VALIDATION_PERCENT} \
    --test-percent=${TEST_PERCENT}
