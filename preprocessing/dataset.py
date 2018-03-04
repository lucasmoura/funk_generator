import os
import random
import pickle

from pathlib import Path
from collections import namedtuple


Song = namedtuple('Song', ['song_text', 'num_words'])


class MusicDataset:

    def __init__(self, data_folder, dataset_save_path):
        self.data_folder = Path(data_folder)
        self.dataset_save_path = self.data_folder / dataset_save_path

    def read_song(self, song_path):
        with song_path.open() as song_file:
            return song_file.read()

    def get_num_words_from_song(self, song):
        return len(song.replace('\n', ' ').split(' '))

    def get_songs(self):
        self.all_songs = []

        for artist in os.listdir(self.data_folder):
            artist_path = self.data_folder / artist

            if not artist_path.is_dir():
                continue

            for song in os.listdir(artist_path):
                if song == 'song_codes.txt' or song == 'song_names.txt':
                    continue

                song_path = artist_path / song
                song_text = self.read_song(song_path)
                num_words = self.get_num_words_from_song(song_text)

                song = Song(song_text, num_words)

                self.all_songs.append(song)

    def average_song_size(self):
        avg_size = 0
        for song in self.all_songs:
            avg_size += song.num_words

        return avg_size / len(self.all_songs)

    def load_dataset(self, dataset_type):
        dataset_path = self.dataset_save_path / dataset_type

        with dataset_path.open('rb') as dataset_file:
            return pickle.load(dataset_file)

    def load_datasets(self):
        if not self.dataset_save_path.is_dir():
            return False

        print('Loading datasets ...')
        self.train_dataset = self.load_dataset('train')
        self.validation_dataset = self.load_dataset('validation')
        self.test_dataset = self.load_dataset('test')

    def split_dataset(self, validation_percent, test_percent):
        total_size = validation_percent + test_percent

        random.shuffle(self.all_songs)
        train_size = len(self.all_songs) - int(len(self.all_songs) * total_size)
        self.train_dataset = self.all_songs[:train_size]

        validation_size = train_size + int(len(self.all_songs) * validation_percent)
        self.validation_dataset = self.all_songs[train_size: validation_size]

        self.test_dataset = self.all_songs[validation_size:]

    def create_dataset(self, validation_percent, test_percent):
        if not self.load_datasets():
            print('Creating dataset ...')
            self.get_songs()
            self.split_dataset(validation_percent, test_percent)
            self.save_dataset()

    def display_info(self):
        print('Total number of songs: {}'.format(len(self.all_songs)))
        print('Average size of songs: {} words'.format(int(self.average_song_size())))
