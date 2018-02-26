import os
import time
import requests
import unicodedata

from bs4 import BeautifulSoup
from pathlib import Path


def remove_accented_characters(name):
    name = unicodedata.normalize('NFD', name).encode('ascii', 'ignore')
    return name.decode('ascii')


def clean_name(name):
    name = name.strip()
    name = name.lower()
    name = name.replace(' ', '-')
    name = name.replace('/', '')
    name = remove_accented_characters(name)

    return name


class MusicCrawler:

    def __init__(self, artist_list_path, data_folder):
        self.artist_list_path = artist_list_path
        self.data_folder = Path(data_folder)
        self.artists = None

        self.vagalume_url = 'https://www.vagalume.com.br/'

    def remove_accented_characters(self, artist_name):
        artist_name = unicodedata.normalize('NFD', artist_name).encode('ascii', 'ignore')
        return artist_name.decode('ascii')

    def parse_artist_name(self, artist_name):
        return clean_name(artist_name)

    def load_artists(self):
        with open(self.artist_list_path, 'r') as artist_file:
            artists = artist_file.readlines()

            self.artists = [self.parse_artist_name(artist) for artist in artists]

    def find_tracks_list(self, html_page):
        songs = []

        tracks = html_page.find('ul', {'class': 'tracks'})
        tracks_hrefs = tracks.findAll('a', href=True)

        for track_href in tracks_hrefs:
            name = track_href.get_text()
            code = track_href.get('data-song')

            if not code or not name:
                continue

            songs.append((name, code))

        return songs

    def get_artist_songs(self, artist_name):
        artist_url = self.vagalume_url + artist_name

        response = requests.get(artist_url)
        parsed_response = BeautifulSoup(response.content, 'html.parser')

        artist_songs = self.find_tracks_list(parsed_response)

        return artist_songs

    def save_data(self, save_path, save_list):
        with save_path.open(mode='w') as f:
            for name in save_list:
                f.write(name + '\n')

    def save_artist_songs_info(self, artist, artist_songs):
        song_names = [name for name, code in artist_songs]
        song_codes = [code for name, code in artist_songs]

        save_path = self.data_folder / artist

        if not save_path.exists():
            save_path.mkdir()

        save_path_names = save_path / 'song_names.txt'
        save_path_codes = save_path / 'song_codes.txt'

        self.save_data(save_path_names, song_names)
        self.save_data(save_path_codes, song_codes)

    def crawl_musics(self):
        self.load_artists()

        for artist in self.artists:
            print('Getting songs of {}...'.format(artist))

            artist_songs = self.get_artist_songs(artist)
            self.save_artist_songs_info(artist, artist_songs)


class MusicDownloader:

    def __init__(self, key_file_path, data_folder, code_file_name):
        self.key_file_path = key_file_path
        self.data_folder = Path(data_folder)
        self.code_file_name = code_file_name

        self.api_url = 'https://api.vagalume.com.br/search.php?musid={}&apikey{}'

    def load_api_key(self):
        with open(self.key_file_path, 'r') as key_file:
            self.api_key = key_file.read().strip()

    def load_codes(self, codes_path):
        with codes_path.open() as code_file:
            codes = code_file.readlines()
            codes = [code.strip() for code in codes]

        return codes

    def make_request(self, code):
        return requests.get(self.api_url.format(code, self.api_key))

    def save_songs(self, songs, artist_path):
        for song_name, song in songs:
            song_name += '.txt'
            song_path = artist_path / song_name

            with song_path.open(mode='w') as song_file:
                song_file.write(song)

    def download_songs(self, artist_name):
        codes_path = self.data_folder / artist_name / self.code_file_name
        codes = self.load_codes(codes_path)
        songs = []

        for code in codes:
            self.make_request(code)
            json_response = self.make_request(code)

            song_name, song = self.parse_json_response(json_response)
            songs.append((song_name, song))
            time.sleep(2)

        artist_path = self.data_folder / artist_name
        self.save_songs(songs, artist_path)

    def clean_song_name(self, song_name):
        return clean_name(song_name)

    def parse_json_response(self, json_response):
        json_dict = json_response.json()

        song = json_dict['mus'][0]['text']
        song_name = json_dict['mus'][0]['name']

        return self.clean_song_name(song_name), song

    def download_all_songs(self):
        self.load_api_key()

        for artist in os.listdir(self.data_folder):
            if artist in succedded_artists:
                continue

            print('Downloading songs from {}'.format(artist))
            self.download_songs(artist)
