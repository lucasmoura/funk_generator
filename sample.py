import argparse
import os

from collections import defaultdict

from model.rnn import RecurrentModel, RecurrentConfig
from model.song_generator import GreedySongGenerator
from utils.session_manager import initialize_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_argparse():
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument('-chp',
                                 '--checkpoint-path',
                                 type=str,
                                 help="The path to save model's checkpoint")

    argument_parser.add_argument('-i2w',
                                 '--index2word-path',
                                 type=str,
                                 help='Location of the index2word dict')

    argument_parser.add_argument('-w2i',
                                 '--word2index-path',
                                 type=str,
                                 help='Location of word2index dict')

    argument_parser.add_argument('-vs',
                                 '--vocab-size',
                                 type=int,
                                 help='Size of the vocabulary')

    argument_parser.add_argument('-es',
                                 '--embedding-size',
                                 type=int,
                                 help='Dimension of the embedding matrix')

    argument_parser.add_argument('-nl',
                                 '--num-layers',
                                 type=int,
                                 help='Number of lstm layers to use')

    argument_parser.add_argument('-nu',
                                 '--num-units',
                                 type=int,
                                 help='Number of units to use in the lstm cell')

    argument_parser.add_argument('-t',
                                 '--temperature',
                                 type=float,
                                 help='Logits temperature')

    return argument_parser


def main():
    argument_parser = create_argparse()
    user_args = vars(argument_parser.parse_args())
    user_args['use_checkpoint'] = True

    user_args = defaultdict(int, user_args)

    config = RecurrentConfig(user_args)
    model = RecurrentModel(None, config)
    model.build_placeholders()
    model.build_generate_graph(reuse=False)

    with initialize_session(config) as (sess, saver):
        generator = GreedySongGenerator(model)
        temperature = user_args['temperature']

        print('Generating song (Greedy) ...')
        generator.generate(sess, temperature=temperature)


if __name__ == '__main__':
    main()
