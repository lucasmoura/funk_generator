from model.rnn import RecurrentModel, RecurrentConfig
from model.song_generator import GreedySongGenerator

import tensorflow as tf


def create_sample(model_config, html=False):
    sample_config = RecurrentConfig(model_config)
    graph = tf.Graph()

    with graph.as_default():
        model = RecurrentModel(None, sample_config)
        model.build_placeholders()
        model.build_generate_graph(reuse=False)

        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(config=config)

        checkpoint = tf.train.latest_checkpoint(sample_config.checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        generator = GreedySongGenerator(model)

    def sample(prime_words, html):
        prime_words = prime_words.split()
        with graph.as_default():
            return generator.generate(sess, prime_words=prime_words, html=html)

    return sample
