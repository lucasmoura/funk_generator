#!/bin/bash

set -e

#usage: ./scripts/run_model.sh

ALL_TRAIN_FILE='data/song_dataset/train/train.tfrecord'
ALL_VALIDATION_FILE='data/song_dataset/validation/validation.tfrecord'
ALL_TEST_FILE='data/song_dataset/test/test.tfrecord'
ALL_INDEX2WORD_PATH='data/song_dataset/index2word.pkl'
ALL_WORD2INDEX_PATH='data/song_dataset/word2index.pkl'
ALL_CHECKPOINT_PATH='checkpoint'
ALL_VOCAB_SIZE=12551

KONDZILLA_TRAIN_FILE='kondzilla/song_dataset/train/train.tfrecord'
KONDZILLA_VALIDATION_FILE='kondzilla/song_dataset/validation/validation.tfrecord'
KONDZILLA_TEST_FILE='kondzilla/song_dataset/test/test.tfrecord'
KONDZILLA_INDEX2WORD_PATH='kondzilla/song_dataset/index2word.pkl'
KONDZILLA_WORD2INDEX_PATH='kondzilla/song_dataset/word2index.pkl'
KONDZILLA_CHECKPOINT_PATH='kondzilla_checkpoint'
KONDZILLA_VOCAB_SIZE=2192

PROIBIDAO_TRAIN_FILE='proibidao-songs/song_dataset/train/train.tfrecord'
PROIBIDAO_VALIDATION_FILE='proibidao-songs/song_dataset/validation/validation.tfrecord'
PROIBIDAO_TEST_FILE='proibidao-songs/song_dataset/test/test.tfrecord'
PROIBIDAO_INDEX2WORD_PATH='proibidao-songs/song_dataset/index2word.pkl'
PROIBIDAO_WORD2INDEX_PATH='proibidao-songs/song_dataset/word2index.pkl'
PROIBIDAO_CHECKPOINT_PATH='proibidao_checkpoint'
PROIBIDAO_VOCAB_SIZE=1445

OSTENTACAO_TRAIN_FILE='ostentacao-songs/song_dataset/train/train.tfrecord'
OSTENTACAO_VALIDATION_FILE='ostentacao-songs/song_dataset/validation/validation.tfrecord'
OSTENTACAO_TEST_FILE='ostentacao-songs/song_dataset/test/test.tfrecord'
OSTENTACAO_INDEX2WORD_PATH='ostentacao-songs/song_dataset/index2word.pkl'
OSTENTACAO_WORD2INDEX_PATH='ostentacao-songs/song_dataset/word2index.pkl'
OSTENTACAO_CHECKPOINT_PATH='ostentacao_checkpoint'
OSTENTACAO_VOCAB_SIZE=2035

USE_CHECKPOINT=1

LEARNING_RATE=0.002
NUM_EPOCHS=20
BATCH_SIZE=32

NUM_LAYERS=3
NUM_UNITS=728
EMBEDDING_SIZE=300
MIN_VAL=-1
MAX_VAL=1

EMBEDDING_DROPOUT=0.5
LSTM_OUTPUT_DROPOUT=0.5
LSTM_STATE_DROPOUT=0.5
LSTM_INPUT_DROPOUT=0.5
WEIGHT_DECAY=0.0000

NUM_BUCKETS=30
BUCKET_WIDTH=30
PREFETCH_BUFFER=8
PERFORM_SHUFFLE=1

PARAM=${1:-all}
if [ $PARAM == "all" ]; then
    echo "Running model for all songs"
	TRAIN_FILE=$ALL_TRAIN_FILE
	VALIDATION_FILE=$ALL_VALIDATION_FILE
	TEST_FILE=$ALL_TEST_FILE
	INDEX2WORD_PATH=$ALL_INDEX2WORD_PATH
	WORD2INDEX_PATH=$ALL_WORD2INDEX_PATH
	CHECKPOINT_PATH=$ALL_CHECKPOINT_PATH
	VOCAB_SIZE=$ALL_VOCAB_SIZE
elif [ $PARAM == "kondzilla" ]; then
    echo "Running model for kondzilla songs"
	TRAIN_FILE=$KONDZILLA_TRAIN_FILE
	VALIDATION_FILE=$KONDZILLA_VALIDATION_FILE
	TEST_FILE=$KONDZILLA_TEST_FILE
	INDEX2WORD_PATH=$KONDZILLA_INDEX2WORD_PATH
	WORD2INDEX_PATH=$KONDZILLA_WORD2INDEX_PATH
	CHECKPOINT_PATH=$KONDZILLA_CHECKPOINT_PATH
	VOCAB_SIZE=$KONDZILLA_VOCAB_SIZE
elif [ $PARAM == "proibidao" ]; then
    echo "Running model for probidao songs"
	TRAIN_FILE=$PROIBIDAO_TRAIN_FILE
	VALIDATION_FILE=$PROIBIDAO_VALIDATION_FILE
	TEST_FILE=$PROIBIDAO_TEST_FILE
	INDEX2WORD_PATH=$PROIBIDAO_INDEX2WORD_PATH
	WORD2INDEX_PATH=$PROIBIDAO_WORD2INDEX_PATH
	CHECKPOINT_PATH=$PROIBIDAO_CHECKPOINT_PATH
	VOCAB_SIZE=$PROIBIDAO_VOCAB_SIZE
elif [ $PARAM == "ostentacao" ]; then
    echo "Running model for ostentacao songs"
	TRAIN_FILE=$OSTENTACAO_TRAIN_FILE
	VALIDATION_FILE=$OSTENTACAO_VALIDATION_FILE
	TEST_FILE=$OSTENTACAO_TEST_FILE
	INDEX2WORD_PATH=$OSTENTACAO_INDEX2WORD_PATH
	WORD2INDEX_PATH=$OSTENTACAO_WORD2INDEX_PATH
	CHECKPOINT_PATH=$OSTENTACAO_CHECKPOINT_PATH
	VOCAB_SIZE=$OSTENTACAO_VOCAB_SIZE
fi


python -u model.py \
  --train-file=${TRAIN_FILE} \
  --validation-file=${VALIDATION_FILE} \
  --test-file=${TEST_FILE} \
  --checkpoint-path=${CHECKPOINT_PATH} \
  --use-checkpoint=${USE_CHECKPOINT} \
  --index2word-path=${INDEX2WORD_PATH} \
  --word2index-path=${WORD2INDEX_PATH} \
  --num-epochs=${NUM_EPOCHS} \
  --batch-size=${BATCH_SIZE} \
  --learning-rate=${LEARNING_RATE} \
  --num-layers=${NUM_LAYERS} \
  --num-units=${NUM_UNITS} \
  --vocab-size=${VOCAB_SIZE} \
  --embedding-size=${EMBEDDING_SIZE} \
  --embedding-dropout=${EMBEDDING_DROPOUT} \
  --lstm-output-dropout=${LSTM_OUTPUT_DROPOUT} \
  --lstm-input-dropout=${LSTM_INPUT_DROPOUT} \
  --lstm-state-dropout=${LSTM_STATE_DROPOUT} \
  --weight-decay=${WEIGHT_DECAY} \
  --min-val=${MIN_VAL} \
  --max-val=${MAX_VAL} \
  --num-buckets=${NUM_BUCKETS} \
  --bucket-width=${BUCKET_WIDTH} \
  --prefetch-buffer=${PREFETCH_BUFFER} \
  --perform-shuffle=${PERFORM_SHUFFLE}
