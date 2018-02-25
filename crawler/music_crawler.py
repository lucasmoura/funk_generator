class MusicCrawler:

    def __init__(self, artist_list_path):
        self.artist_list_path = artist_list_path
        self.artists = None

    def parse_artist_name(self, artist_name):
        artist_name = artist_name.strip()
        artist_name = artist_name.lower()
        artist_name = artist_name.replace(' ', '-')

        return artist_name

    def load_artists(self):
        with open(self.artist_list_path, 'r') as artist_file:
            artists = artist_file.readlines()

            self.artists = [self.parse_artist_name(artist) for artist in artists]

    def crawl_musics(self):
        self.load_artists()
