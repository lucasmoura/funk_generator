import requests
import unicodedata

from bs4 import BeautifulSoup
from pathlib import Path


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
        artist_name = artist_name.strip()
        artist_name = artist_name.lower()
        artist_name = artist_name.replace(' ', '-')
        artist_name = self.remove_accented_characters(artist_name)

        return artist_name

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
