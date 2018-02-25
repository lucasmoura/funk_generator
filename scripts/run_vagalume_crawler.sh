#!/bin/bash

#usage: ./scripts/run_vagalume_crawler.sh

set -e

ARTIST_LIST_PATH='data/artist_list.txt'


python vagalume_crawler.py \
    --artist-list-path=${ARTIST_LIST_PATH}
