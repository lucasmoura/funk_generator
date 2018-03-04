import argparse

from preprocessing.dataset import MusicDataset


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-df',
                        '--data-folder',
                        type=str,
                        help='Location of the songs files')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    data_folder = user_args['data_folder']
    music_dataset = MusicDataset(data_folder)

    music_dataset.get_songs()
    music_dataset.display_info()


if __name__ == '__main__':
    main()
