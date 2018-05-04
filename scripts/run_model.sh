#!/bin/bash

set -e

#usage: ./scripts/run_model.sh

TRAIN_FILE='data/song_dataset/train/train.tfrecord'
VALIDATION_FILE='data/song_dataset/validation/validation.tfrecord'
TEST_FILE='data/song_dataset/test/test.tfrecord'

INDEX2WORD_PATH='data/song_dataset/index2word.pkl'
WORD2INDEX_PATH='data/song_dataset/word2index.pkl'

CHECKPOINT_PATH='checkpoint'
USE_CHECKPOINT=1

LEARNING_RATE=0.002
NUM_EPOCHS=0
BATCH_SIZE=32

NUM_LAYERS=3
NUM_UNITS=728
VOCAB_SIZE=12551
EMBEDDING_SIZE=300
MIN_VAL=-1
MAX_VAL=1

EMBEDDING_DROPOUT=0.5
LSTM_OUTPUT_DROPOUT=0.7
LSTM_STATE_DROPOUT=0.9
LSTM_INPUT_DROPOUT=0.5
WEIGHT_DECAY=0.0000

NUM_BUCKETS=30
BUCKET_WIDTH=30
PREFETCH_BUFFER=8
PERFORM_SHUFFLE=0


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
