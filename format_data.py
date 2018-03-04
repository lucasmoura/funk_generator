import argparse

from preprocessing.dataset import MusicDataset


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-df',
                        '--data-folder',
                        type=str,
                        help='Location of the songs files')

    parser.add_argument('-dsp',
                        '--dataset-save-path',
                        type=str,
                        help='Location to save the dataset files')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    data_folder = user_args['data_folder']
    dataset_save_path = user_args['dataset_save_path']
    music_dataset = MusicDataset(data_folder, dataset_save_path)

    music_dataset.create_dataset(validation_percent=0.1, test_percent=0.1)
    music_dataset.display_info()


if __name__ == '__main__':
    main()
