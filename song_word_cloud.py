import argparse
import nltk
import pickle

from PIL import Image
import numpy as np

from wordcloud import WordCloud


def create_argparse():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-sp',
                                 '--songs-path',
                                 type=str,
                                 help='Location of the all songs pickle file')

    argument_parser.add_argument('-gn',
                                 '--graph-name',
                                 type=str,
                                 help='Name of the word cloud graph')

    return argument_parser


def create_songs_str(songs_path):
    stopwords = nltk.corpus.stopwords.words('portuguese')
    more_words = ['pra', 'tá', 'pode', 'tô', 'hoje', 'Então', 'então', 'agora',
                  'tudo', 'porque', 'sempre', 'quero', 'quer', 'sei', 'Refrão',
                  '2x', 'assim', 'aqui', 'todo', 'vai', 'vem', 'nóis', 'vou',
                  'pro', 'ser', 'nois', 'ter', 'tao', 'la', 'tão', 'ta']
    stopwords.extend(more_words)

    with open(songs_path, 'rb') as f:
        all_songs = pickle.load(f)

    songs = [' '.join(s[1:-1]) for s in all_songs]
    all_songs_str = '\n'.join(songs)
    stopwords = set(stopwords)

    return all_songs_str, stopwords


def create_word_cloud_graph(all_songs_str, stopwords, graph_name):
    sarrada_mask = np.array(Image.open("masks/romano.png"))
    wc = WordCloud(background_color="white", max_words=2000, mask=sarrada_mask,
                   stopwords=stopwords)
    wc.generate(all_songs_str)
    wc.to_file(graph_name)


def main():
    argument_parser = create_argparse()
    user_args = vars(argument_parser.parse_args())
    songs_path = user_args['songs_path']
    graph_name = user_args['graph_name']

    all_songs_str, stopwords = create_songs_str(songs_path)
    create_word_cloud_graph(all_songs_str, stopwords, graph_name)


if __name__ == '__main__':
    main()
