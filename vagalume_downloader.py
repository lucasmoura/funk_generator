import argparse

from crawler.music_crawler import MusicDownloader


def create_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-kfp',
                        '--key-file-path',
                        type=str,
                        help='Location of the file containing the Vagalume API key')

    parser.add_argument('-df',
                        '--data-folder',
                        type=str,
                        help='Location of the data files')

    parser.add_argument('-cfn',
                        '--code-files-name',
                        type=str,
                        help='Name of the file that contains the music ids')

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    key_file_path = user_args['key_file_path']
    data_folder = user_args['data_folder']
    code_files_name = user_args['code_files_name']

    music_downloader = MusicDownloader(key_file_path, data_folder, code_files_name)
    music_downloader.download_all_songs()


if __name__ == '__main__':
    main()
