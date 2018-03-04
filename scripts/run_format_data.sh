#!/bin/bash

#usage: ./scripts/run_format_data.sh

set -e

DATA_FOLDER='data'

python format_data.py \
    --data-folder=${DATA_FOLDER}
