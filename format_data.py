import argparse

from pathlib import Path

from preprocessing.dataset import MusicDataset
from preprocessing.tfrecord import SentenceTFRecord
from preprocessing.text_preprocessing import (get_vocabulary, create_word_dictionaties,
                                              save, replace_unk_words, replace_words_with_ids,
                                              create_labels, get_sizes_list)


DATA = 0
LABELS = 1
SIZES = 2


def save_dataset(dataset_all, dataset_type, dataset_save_path):
    save_path = dataset_save_path / dataset_type

    if not save_path.is_dir():
        save_path.mkdir()

    data = dataset_all[DATA]
    data_save_path = save_path / str(dataset_type + '_data.pkl')
    save(data, data_save_path)

    labels = dataset_all[LABELS]
    labels_save_path = save_path / str(dataset_type + '_labels.pkl')
    save(labels, labels_save_path)

    sizes = dataset_all[SIZES]
    sizes_save_path = save_path / str(dataset_type + '_sizes.pkl')
    save(sizes, sizes_save_path)


def create_tfrecord(dataset, dataset_type, dataset_save_path):
    save_path = dataset_save_path / dataset_type / str(dataset_type + '.tfrecord')
    sentence_tfrecord = SentenceTFRecord(dataset, str(save_path))
    sentence_tfrecord.parse_sentences()


def full_preprocessing(train, validation, test, data_folder,
                       dataset_save_path, min_frequency):

    data_folder = Path(data_folder)

    print('Creating vocabulary ...')
    vocabulary = get_vocabulary(train, min_frequency)
    print('Vocabulary lenght: {}'.format(len(vocabulary)))
    word2index, index2word = create_word_dictionaties(vocabulary)

    index2word_path = data_folder / dataset_save_path / 'index2word.pkl'
    save(index2word, index2word_path)
    word2index_path = data_folder / dataset_save_path / 'word2index.pkl'
    save(word2index, word2index_path)

    print('Replacing unknown words ...')
    replace_unk_words(train, word2index)
    replace_unk_words(validation, word2index)
    replace_unk_words(test, word2index)

    print('Turning words into word ids ...')
    train = replace_words_with_ids(train, word2index)
    validation = replace_words_with_ids(validation, word2index)
    test = replace_words_with_ids(test, word2index)

    print('Creating labels ...')
    train, train_labels = create_labels(train)
    validation, validation_labels = create_labels(validation)
    test, test_labels = create_labels(test)

    print('Creating size list ...')
    train_sizes = get_sizes_list(train)
    validation_sizes = get_sizes_list(validation)
    test_sizes = get_sizes_list(test)

    train_all = (train, train_labels, train_sizes)
    validation_all = (validation, validation_labels, validation_sizes)
    test_all = (test, test_labels, test_sizes)

    dataset_save_path = data_folder / dataset_save_path

    print('Saving datasets ...')
    save_dataset(train_all, 'train', dataset_save_path)
    save_dataset(validation_all, 'validation', dataset_save_path)
    save_dataset(test_all, 'test', dataset_save_path)

    print('Creating tfrecords ...')
    create_tfrecord(train_all, 'train', dataset_save_path)
    create_tfrecord(validation_all, 'validation', dataset_save_path)
    create_tfrecord(test_all, 'test', dataset_save_path)


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

    parser.add_argument('-mf',
                        '--min-frequency',
                        type=int,
                        help='Minimum word frequency required for a word to a part of the vocabulary')  # noqa

    return parser


def main():
    parser = create_argparse()
    user_args = vars(parser.parse_args())

    data_folder = user_args['data_folder']
    dataset_save_path = Path(user_args['dataset_save_path'])
    music_dataset = MusicDataset(data_folder, dataset_save_path)

    music_dataset.create_dataset(validation_percent=0.1, test_percent=0.1)
    music_dataset.display_info()

    train_dataset = music_dataset.train_dataset
    validation_dataset = music_dataset.validation_dataset
    test_dataset = music_dataset.test_dataset
    min_frequency = user_args['min_frequency']

    full_preprocessing(
        train=train_dataset,
        validation=validation_dataset,
        test=test_dataset,
        data_folder=data_folder,
        dataset_save_path=dataset_save_path,
        min_frequency=min_frequency)


if __name__ == '__main__':
    main()
