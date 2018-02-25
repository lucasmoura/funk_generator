#!/bin/bash

#usage: ./scripts/run_vagalume_crawler.sh

set -e

DATA_FOLDER='data/'
ARTIST_LIST_PATH=$DATA_FOLDER'artist_list.txt'


python vagalume_crawler.py \
    --data-folder=${DATA_FOLDER} \
    --artist-list-path=${ARTIST_LIST_PATH}
