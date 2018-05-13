#!/bin/bash

set -e

#usage: ./scripts/create_sample.sh

ALL_INDEX2WORD_PATH='data/song_dataset/index2word.pkl'
ALL_WORD2INDEX_PATH='data/song_dataset/word2index.pkl'
ALL_CHECKPOINT_PATH='good_checkpoint'
ALL_VOCAB_SIZE=12399

KONDZILLA_INDEX2WORD_PATH='kondzilla/song_dataset/index2word.pkl'
KONDZILLA_WORD2INDEX_PATH='kondzilla/song_dataset/word2index.pkl'
KONDZILLA_CHECKPOINT_PATH='kondzilla_checkpoint'
KONDZILLA_VOCAB_SIZE=2192

PROIBIDAO_INDEX2WORD_PATH='proibidao-songs/song_dataset/index2word.pkl'
PROIBIDAO_WORD2INDEX_PATH='proibidao-songs/song_dataset/word2index.pkl'
PROIBIDAO_CHECKPOINT_PATH='proibidao_checkpoint'
PROIBIDAO_VOCAB_SIZE=1445

OSTENTACAO_INDEX2WORD_PATH='ostentacao-songs/song_dataset/index2word.pkl'
OSTENTACAO_WORD2INDEX_PATH='ostentacao-songs/song_dataset/word2index.pkl'
OSTENTACAO_CHECKPOINT_PATH='ostentacao_checkpoint'
OSTENTACAO_VOCAB_SIZE=2035

EMBEDDING_SIZE=300
NUM_LAYERS=3
NUM_UNITS=728
TEMPERATURE=0.7

PARAM=${1:-all}
if [ $PARAM == "all" ]; then
    echo "Creating sample for all songs model"
	INDEX2WORD_PATH=$ALL_INDEX2WORD_PATH
	WORD2INDEX_PATH=$ALL_WORD2INDEX_PATH
	CHECKPOINT_PATH=$ALL_CHECKPOINT_PATH
	VOCAB_SIZE=$ALL_VOCAB_SIZE
elif [ $PARAM == "kondzilla" ]; then
    echo "Creating sample for kondzilla songs model"
	INDEX2WORD_PATH=$KONDZILLA_INDEX2WORD_PATH
	WORD2INDEX_PATH=$KONDZILLA_WORD2INDEX_PATH
	CHECKPOINT_PATH=$KONDZILLA_CHECKPOINT_PATH
	VOCAB_SIZE=$KONDZILLA_VOCAB_SIZE
elif [ $PARAM == "proibidao" ]; then
    echo "Creating sample for probidao songs model"
	INDEX2WORD_PATH=$PROIBIDAO_INDEX2WORD_PATH
	WORD2INDEX_PATH=$PROIBIDAO_WORD2INDEX_PATH
	CHECKPOINT_PATH=$PROIBIDAO_CHECKPOINT_PATH
	VOCAB_SIZE=$PROIBIDAO_VOCAB_SIZE
elif [ $PARAM == "ostentacao" ]; then
    echo "Creating sample for ostentacao songs model"
	INDEX2WORD_PATH=$OSTENTACAO_INDEX2WORD_PATH
	WORD2INDEX_PATH=$OSTENTACAO_WORD2INDEX_PATH
	CHECKPOINT_PATH=$OSTENTACAO_CHECKPOINT_PATH
	VOCAB_SIZE=$OSTENTACAO_VOCAB_SIZE
fi

python -u sample.py \
  --checkpoint-path=${CHECKPOINT_PATH} \
  --index2word-path=${INDEX2WORD_PATH} \
  --word2index-path=${WORD2INDEX_PATH} \
  --vocab-size=${VOCAB_SIZE} \
  --embedding-size=${EMBEDDING_SIZE} \
  --num-layers=${NUM_LAYERS} \
  --num-units=${NUM_UNITS} \
  --temperature=${TEMPERATURE}
