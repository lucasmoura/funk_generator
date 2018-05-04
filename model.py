import argparse
import contextlib
import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.rnn import RecurrentModel, RecurrentConfig
from model.song_generator import GreedySongGenerator, BeamSearchSongGenerator


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_argparse():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-tf',
                                 '--train-file',
                                 type=str,
                                 help='Location of the training file')

    argument_parser.add_argument('-vf',
                                 '--validation-file',
                                 type=str,
                                 help='Location of the validation file')

    argument_parser.add_argument('-tef',
                                 '--test-file',
                                 type=str,
                                 help='Location of the test file')

    argument_parser.add_argument('-chp',
                                 '--checkpoint-path',
                                 type=str,
                                 help="The path to save model's checkpoint")

    argument_parser.add_argument('-uch',
                                 '--use-checkpoint',
                                 type=int,
                                 help='If the model checkpoint should be loaded')

    argument_parser.add_argument('-i2w',
                                 '--index2word-path',
                                 type=str,
                                 help='Location of the index2word dict')

    argument_parser.add_argument('-w2i',
                                 '--word2index-path',
                                 type=str,
                                 help='Location of word2index dict')

    argument_parser.add_argument('-ne',
                                 '--num-epochs',
                                 type=int,
                                 help='Number of epochs to train')

    argument_parser.add_argument('-bs',
                                 '--batch-size',
                                 type=int,
                                 help='Batch size to use in the model')

    argument_parser.add_argument('-lr',
                                 '--learning-rate',
                                 type=float,
                                 help='Learning rate to use when training')

    argument_parser.add_argument('-nl',
                                 '--num-layers',
                                 type=int,
                                 help='Number of lstm layers to use')

    argument_parser.add_argument('-nu',
                                 '--num-units',
                                 type=int,
                                 help='Number of units to use in the lstm cell')

    argument_parser.add_argument('-vs',
                                 '--vocab-size',
                                 type=int,
                                 help='Size of the vocabulary')

    argument_parser.add_argument('-es',
                                 '--embedding-size',
                                 type=int,
                                 help='Dimension of the embedding matrix')

    argument_parser.add_argument('-ed',
                                 '--embedding-dropout',
                                 type=float,
                                 help='Embedding dropout')

    argument_parser.add_argument('-lod',
                                 '--lstm-output-dropout',
                                 type=float,
                                 help='LSTM output dropout')

    argument_parser.add_argument('-lid',
                                 '--lstm-input-dropout',
                                 type=float,
                                 help='LSTM input dropout')

    argument_parser.add_argument('-lsd',
                                 '--lstm-state-dropout',
                                 type=float,
                                 help='LSTM state dropout')

    argument_parser.add_argument('-wd',
                                 '--weight-decay',
                                 type=float,
                                 help='Weight decay')

    argument_parser.add_argument('-minv',
                                 '--min-val',
                                 type=int,
                                 help='Min value to use when initializing weights')

    argument_parser.add_argument('-maxv',
                                 '--max-val',
                                 type=int,
                                 help='Max value to use when initializing weights')

    argument_parser.add_argument('-nbc',
                                 '--num-buckets',
                                 type=int,
                                 help='Number of buckets to use')

    argument_parser.add_argument('-bcw',
                                 '--bucket-width',
                                 type=int,
                                 help='Number of elements allowed in bucket')

    argument_parser.add_argument('-pb',
                                 '--prefetch-buffer',
                                 type=int,
                                 help='Size of prefetch buffer')

    argument_parser.add_argument('-pf',
                                 '--perform-shuffle',
                                 type=int,
                                 help='If we shoudl shuffle the batches when training the model')

    return argument_parser


@contextlib.contextmanager
def initialize_session(user_config):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    checkpoint = tf.train.latest_checkpoint(user_config.checkpoint_path)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        if user_config.use_checkpoint:
            print('Load checkpoint: {}'.format(checkpoint))
            saver.restore(sess, checkpoint)
        else:
            print('Creating new model')
            if not os.path.exists(user_config.checkpoint_path):
                os.makedirs(user_config.checkpoint_path)

            sess.run(tf.global_variables_initializer())

        yield (sess, saver)


def main():
    argument_parser = create_argparse()
    user_args = vars(argument_parser.parse_args())

    train_file = user_args['train_file']
    validation_file = user_args['validation_file']
    test_file = user_args['test_file']
    batch_size = user_args['batch_size']
    num_buckets = user_args['num_buckets']
    bucket_width = user_args['bucket_width']
    prefetch_buffer = user_args['prefetch_buffer']
    perform_shuffle = True if user_args['perform_shuffle'] == 1 else False

    dataset = InputPipeline(
        train_files=train_file,
        validation_files=validation_file,
        test_files=test_file,
        batch_size=batch_size,
        perform_shuffle=perform_shuffle,
        bucket_width=bucket_width,
        num_buckets=num_buckets,
        prefetch_buffer=prefetch_buffer)

    dataset.build_pipeline()

    config = RecurrentConfig(user_args)
    model = RecurrentModel(dataset, config)
    model.build_graph()

    with initialize_session(config) as (sess, saver):
        model.fit(sess, saver)

        generator = GreedySongGenerator(model)
        print('Generating song (Greedy) ...')
        generator.generate(sess)

        beam_generator = BeamSearchSongGenerator(model)
        print('Generating song (Beam Search) ...')
        beam_generator.generate(sess)


if __name__ == '__main__':
    main()
