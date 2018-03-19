import argparse
import os

import tensorflow as tf

from model.input_pipeline import InputPipeline
from model.rnn import RecurrentModel, ModelConfig


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

    argument_parser.add_argument('-lod',
                                 '--lstm-output-dropout',
                                 type=float,
                                 help='LSTM output dropout')

    argument_parser.add_argument('-minv',
                                 '--min-val',
                                 type=int,
                                 help='Min value to use when initializing weights')

    argument_parser.add_argument('-maxv',
                                 '--max-val',
                                 type=int,
                                 help='Max value to use when initializing weights')

    return argument_parser


def main():
    argument_parser = create_argparse()
    user_args = vars(argument_parser.parse_args())

    train_file = user_args['train_file']
    validation_file = user_args['validation_file']
    test_file = user_args['test_file']
    batch_size = user_args['batch_size']

    dataset = InputPipeline(
        train_files=train_file,
        validation_files=validation_file,
        test_files=test_file,
        batch_size=batch_size,
        perform_shuffle=True,
        bucket_width=30,
        num_buckets=30)

    dataset.build_pipeline()

    config = ModelConfig(user_args)
    model = RecurrentModel(dataset, config)

    with tf.Session() as sess:
        model.build_graph()

        init = tf.global_variables_initializer()
        init.run()

        model.fit(sess)


if __name__ == '__main__':
    main()
