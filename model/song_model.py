import numpy as np
import tensorflow as tf


class ModelConfig:

    def __init__(self, model_params):
        self.vocab_size = model_params['vocab_size']
        self.embedding_size = model_params['embedding_size']
        self.learning_rate = model_params['learning_rate']
        self.num_epochs = model_params['num_epochs']


class SongLyricsModel:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def add_embedding_op(self, data_batch):
        raise NotImplementedError

    def add_logits_op(self, data_batch, size_batch, reuse=False):
        raise NotImplementedError

    def add_loss_op(self, logits, labels_batch, size_batch):
        raise NotImplementedError

    def add_train_op(self, loss):
        raise NotImplementedError

    def run_epoch(self, sess, ops, training=True):
        costs = 0
        num_iters = 0

        while True:
            try:

                if training:
                    _, batch_loss, batch_num_words = sess.run(ops)
                else:
                    batch_loss, batch_num_words = sess.run(ops)

                costs += batch_loss
                num_iters += 1

            except tf.errors.OutOfRangeError:
                return np.exp(costs / num_iters)

    def fit(self, sess):
        for i in range(self.config.num_epochs):
            print('Running epoch: {}'.format(i + 1))
            ops = [self.train_op, self.train_loss, self.train_num_words]
            sess.run(self.train_iterator.initializer)
            train_perplexity = self.run_epoch(sess, ops)
            print('Train perplexity: {:.3f}'.format(train_perplexity))

            ops = [self.validation_loss, self.validation_num_words]
            sess.run(self.validation_iterator.initializer)
            val_perplexity = self.run_epoch(sess, ops, training=False)
            print('Validation perplexity: {:.3f}'.format(val_perplexity))

            print()

    def build_graph(self):

        with tf.name_scope('iterator'):
            self.train_iterator = self.dataset.train_iterator
            self.validation_iterator = self.dataset.validation_iterator
            self.test_iterator = self.dataset.test_iterator

        with tf.name_scope('train_data'):
            train_data, train_labels, train_size = self.train_iterator.get_next()

        with tf.name_scope('validation_data'):
            (validation_data, validation_labels,
                validation_size) = self.validation_iterator.get_next()

        with tf.name_scope('test_data'):
            test_data, test_labels, test_size = self.test_iterator.get_next()

        with tf.name_scope('train'):
            train_logits = self.add_logits_op(train_data, train_size)
            self.train_loss = self.add_loss_op(train_logits, train_labels, train_size)
            self.train_num_words = tf.reduce_sum(train_size)
            self.train_op = self.add_train_op(self.train_loss)

        self.config.lstm_output_dropout = 1.0
        with tf.name_scope('validation'):
            validation_logits = self.add_logits_op(validation_data, validation_size, reuse=True)
            self.validation_loss = self.add_loss_op(
                validation_logits, validation_labels, validation_size)
            self.validation_num_words = tf.reduce_sum(validation_size)
