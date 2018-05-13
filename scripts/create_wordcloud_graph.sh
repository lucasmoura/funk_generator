#!/bin/bash

set -e

#usage: ./scripts/create_word_cloud.sh

ALL_SONGS_PATH='data/song_dataset/all_songs.pkl'
ALL_GRAPH_NAME='all-word-cloud-graph.png'

KONDZILLA_SONGS_PATH='kondzilla/song_dataset/all_songs.pkl'
KONDZILLA_GRAPH_NAME='kondzilla-word-cloud-graph.png'

PROIBIDAO_SONGS_PATH='proibidao-songs/song_dataset/all_songs.pkl'
PROIBIDAO_GRAPH_NAME='proibidao-word-cloud-graph.png'

OSTENTACAO_SONGS_PATH='ostentacao-songs/song_dataset/all_songs.pkl'
OSTENTACAO_GRAPH_NAME='ostentacao-word-cloud-graph.png'


PARAM=${1:-all}
if [ $PARAM == "all" ]; then
    echo "Creating word cloud graph for all songs"
    SONGS_PATH=$ALL_SONGS_PATH
    GRAPH_NAME=$ALL_GRAPH_NAME
elif [ $PARAM == "kondzilla" ]; then
    echo "Creating word cloud graph for kondzilla songs"
    SONGS_PATH=$KONDZILLA_SONGS_PATH
    GRAPH_NAME=$KONDZILLA_GRAPH_NAME
elif [ $PARAM == "proibidao" ]; then
    echo "Creating word cloud graph for proibidao songs"
    SONGS_PATH=$PROIBIDAO_SONGS_PATH
    GRAPH_NAME=$PROIBIDAO_GRAPH_NAME
elif [ $PARAM == "ostentacao" ]; then
    echo "Creating word cloud graph for ostentacao songs"
    SONGS_PATH=$OSTENTACAO_SONGS_PATH
    GRAPH_NAME=$OSTENTACAO_GRAPH_NAME
fi

python -u song_word_cloud.py \
    --songs-path=${SONGS_PATH} \
    --graph-name=${GRAPH_NAME}
