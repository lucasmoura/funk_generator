#!/bin/bash

set -e

#usage: ./scripts/create_sample.sh

INDEX2WORD_PATH='data/song_dataset/index2word.pkl'
WORD2INDEX_PATH='data/song_dataset/word2index.pkl'

CHECKPOINT_PATH='checkpoint'

VOCAB_SIZE=12551
EMBEDDING_SIZE=300
NUM_LAYERS=3
NUM_UNITS=728

python -u sample.py \
  --checkpoint-path=${CHECKPOINT_PATH} \
  --index2word-path=${INDEX2WORD_PATH} \
  --word2index-path=${WORD2INDEX_PATH} \
  --vocab-size=${VOCAB_SIZE} \
  --embedding-size=${EMBEDDING_SIZE} \
  --num-layers=${NUM_LAYERS} \
  --num-units=${NUM_UNITS}
