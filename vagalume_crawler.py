import argparse

from crawler.music_crawler import MusicCrawler


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d',
                        '--data-folder',
                        type=str,
                        help='Data folder path')

    parser.add_argument('-al',
                        '--artist-list-path',
                        type=str,
                        help='Path of the file containing the artists')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    artist_list_path = user_args['artist_list_path']
    data_folder = user_args['data_folder']

    music_crawler = MusicCrawler(artist_list_path, data_folder)
    music_crawler.crawl_musics()


if __name__ == '__main__':
    main()
