import contextlib
import os

import tensorflow as tf


@contextlib.contextmanager
def initialize_session(user_config, use_gpu=True):
    if use_gpu:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})

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
