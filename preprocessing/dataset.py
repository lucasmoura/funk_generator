import os

from pathlib import Path
from collections import namedtuple


Song = namedtuple('Song', ['song_text', 'num_words'])


class MusicDataset:

    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)

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

    def display_info(self):
        print('Total number of songs: {}'.format(len(self.all_songs)))
        print('Average size of songs: {} words'.format(int(self.average_song_size())))
