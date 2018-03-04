#!/bin/bash

#usage: ./scripts/run_format_data.sh

set -e

DATA_FOLDER='data'
DATASET_SAVE_PATH='song_dataset'

python format_data.py \
    --data-folder=${DATA_FOLDER} \
    --dataset-save-path=${DATASET_SAVE_PATH}
