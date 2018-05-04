import os
import random
import pickle
import re

from pathlib import Path


class MusicDataset:

    def __init__(self, data_folder, dataset_save_path):
        self.data_folder = Path(data_folder)
        self.dataset_save_path = self.data_folder / dataset_save_path

    def read_song(self, song_path):
        with song_path.open() as song_file:
            return song_file.read()

    def get_num_words_from_song(self, song):
        return len(song.replace('\n', ' ').split(' '))

    def format_song_text(self, song_text):
        song_text = re.sub(r",", "", song_text)
        song_text = re.sub(r"!", " ! ", song_text)
        song_text = re.sub(r"\?", " ? ", song_text)
        song_text = re.sub(r"\)", "", song_text)
        song_text = re.sub(r"\(", "", song_text)
        song_text = re.sub(r"\}", "", song_text)
        song_text = re.sub(r"\{", "", song_text)
        song_text = re.sub(r":", "", song_text)
        song_text = re.sub(r"\.", "  ", song_text)
        song_text = re.sub(r"\n", " \n ", song_text)
        song_text = re.sub(r'"', " ", song_text)
        song_text = re.sub(r"\[.*\]", " ", song_text)

        song_text = '<begin> ' + song_text + ' <end>'
        song_text = re.sub(r'\s{2,}', ' ', song_text)

        song_text = song_text.split(' ')

        return song_text

    def get_songs(self):
        self.all_songs = []

        for artist in os.listdir(str(self.data_folder)):
            artist_path = self.data_folder / artist

            if not artist_path.is_dir():
                continue

            for song in os.listdir(str(artist_path)):
                if song == 'song_codes.txt' or song == 'song_names.txt':
                    continue

                song_path = artist_path / song
                song_text = self.read_song(song_path)

                song_text = self.format_song_text(song_text)
                self.all_songs.append(song_text)

    def average_song_size(self):
        avg_size = 0
        for song in self.all_songs:
            avg_size += len(song)

        return avg_size / len(self.all_songs)

    def load_dataset(self, dataset_type):
        dataset_path = self.dataset_save_path / dataset_type

        with dataset_path.open(mode='rb') as dataset_file:
            return pickle.load(dataset_file)

    def load_datasets(self):
        if not self.dataset_save_path.is_dir():
            return False

        print('Loading datasets ...')
        self.all_songs = self.load_dataset('all_songs.pkl')
        self.train_dataset = self.load_dataset('train/raw_train.pkl')
        self.validation_dataset = self.load_dataset('validation/raw_validation.pkl')
        self.test_dataset = self.load_dataset('test/raw_test.pkl')

        return True

    def save_dataset(self, dataset, dataset_path):
        with dataset_path.open(mode='wb') as dataset_file:
            pickle.dump(dataset, dataset_file)

    def create_dirs(self):
        if not self.dataset_save_path.is_dir():
            self.dataset_save_path.mkdir()

        train_dataset = self.dataset_save_path / 'train'
        if not train_dataset.is_dir():
            train_dataset.mkdir()

        validation_dataset = self.dataset_save_path / 'validation'
        if not validation_dataset.is_dir():
            validation_dataset.mkdir()

        test_dataset = self.dataset_save_path / 'test'
        if not test_dataset.is_dir():
            test_dataset.mkdir()

        return train_dataset, validation_dataset, test_dataset

    def save_datasets(self):

        (train_dataset_path, validation_dataset_path,
            test_dataset_path) = self.create_dirs()

        save_path = self.dataset_save_path / 'all_songs.pkl'
        self.save_dataset(self.all_songs, save_path)

        train_dataset_path = train_dataset_path / 'raw_train.pkl'
        self.save_dataset(self.train_dataset, train_dataset_path)

        validation_dataset_path = validation_dataset_path / 'raw_validation.pkl'
        self.save_dataset(self.validation_dataset, validation_dataset_path)

        test_dataset_path = test_dataset_path / 'raw_test.pkl'
        self.save_dataset(self.test_dataset, test_dataset_path)

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
            self.save_datasets()

    def display_info(self):
        print('Total number of songs: {}'.format(len(self.all_songs)))
        print('Average size of songs: {} words'.format(int(self.average_song_size())))

        print('Train size: {}'.format(len(self.train_dataset)))
        print('Validation size: {}'.format(len(self.validation_dataset)))
        print('Test size: {}'.format(len(self.test_dataset)))
