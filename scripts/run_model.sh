#!/bin/bash

set -e

#usage: ./scripts/run_model.sh

TRAIN_FILE='data/song_dataset/train/train.tfrecord'
VALIDATION_FILE='data/song_dataset/validation/validation.tfrecord'
TEST_FILE='data/song_dataset/test/test.tfrecord'

INDEX2WORD_PATH='data/song_dataset/index2word.pkl'
WORD2INDEX_PATH='data/song_dataset/word2index.pkl'

LEARNING_RATE=0.0001
NUM_EPOCHS=10
BATCH_SIZE=128

NUM_LAYERS=2
NUM_UNITS=128
VOCAB_SIZE=6448
EMBEDDING_SIZE=50
MIN_VAL=-1
MAX_VAL=1
LSTM_OUTPUT_DROPOUT=0.5


python model.py \
  --train-file=${TRAIN_FILE} \
  --validation-file=${VALIDATION_FILE} \
  --test-file=${TEST_FILE} \
  --index2word-path=${INDEX2WORD_PATH} \
  --word2index-path=${WORD2INDEX_PATH} \
  --num-epochs=${NUM_EPOCHS} \
  --batch-size=${BATCH_SIZE} \
  --learning-rate=${LEARNING_RATE} \
  --num-layers=${NUM_LAYERS} \
  --num-units=${NUM_UNITS} \
  --vocab-size=${VOCAB_SIZE} \
  --embedding-size=${EMBEDDING_SIZE} \
  --lstm-output-dropout=${LSTM_OUTPUT_DROPOUT} \
  --min-val=${MIN_VAL} \
  --max-val=${MAX_VAL}
