#!/bin/bash

#usage: ./scripts/download_musics.sh

set -e

DATA_FOLDER='data/'
KEY_FILE_PATH=$DATA_FOLDER'config'
CODE_FILES_NAME='song_codes.txt'

python vagalume_downloader.py \
  --key-file-path=${KEY_FILE_PATH} \
  --data-folder=${DATA_FOLDER} \
  --code-files-name=${CODE_FILES_NAME}
