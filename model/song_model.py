import pickle

import numpy as np
import tensorflow as tf


class ModelConfig:

    def __init__(self, model_params):
        self.vocab_size = model_params['vocab_size']
        self.embedding_size = model_params['embedding_size']
        self.learning_rate = model_params['learning_rate']
        self.num_epochs = model_params['num_epochs']
        self.index2word_path = model_params['index2word_path']
        self.word2index_path = model_params['word2index_path']


class SongLyricsModel:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        self.index2word = self.load_dict(self.config.index2word_path)
        self.word2index = self.load_dict(self.config.word2index_path)

    def load_dict(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

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
                    feed_dict = self.create_train_feed_dict()
                    _, batch_loss = sess.run(ops, feed_dict=feed_dict)
                else:
                    feed_dict = self.create_validation_feed_dict()
                    batch_loss = sess.run(ops, feed_dict=feed_dict)

                costs += batch_loss
                num_iters += 1

            except tf.errors.OutOfRangeError:
                return np.exp(costs / num_iters)

    def fit(self, sess):
        best_perplexity = 10000000
        saver = tf.train.Saver()

        for i in range(self.config.num_epochs):
            print('Running epoch: {}'.format(i + 1))
            ops = [self.train_op, self.train_loss]
            sess.run(self.train_iterator.initializer)
            train_perplexity = self.run_epoch(sess, ops)
            print('Train perplexity: {:.3f}'.format(train_perplexity))

            ops = self.validation_loss
            sess.run(self.validation_iterator.initializer)
            val_perplexity = self.run_epoch(sess, ops, training=False)
            print('Validation perplexity: {:.3f}'.format(val_perplexity))

            if val_perplexity < best_perplexity:
                best_perplexity = val_perplexity
                print('New best perplexity found !  {:.3f}'.format(best_perplexity))

                saver.save(sess, 'best_model/song_model.ckpt')

            print('Song generated for epoch: {}'.format(i + 1))
            print(' '.join(self.generate(sess)))

            print()

    def generate(self, sess, num_out=200):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        word = self.word2index['<begin>']

        song = []

        for i in range(num_out):
            input_word_id = np.array([[word]])
            feed_dict = self.create_generate_feed_dict(
                data=input_word_id,
                temperature=0.7,
                state=state)
            probs, state = sess.run(
                [self.generate_predictions, self.final_state], feed_dict=feed_dict)

            probs = probs[0].reshape(-1)

            generated_word_id = np.random.choice(
                np.arange(len(probs)), p=probs)
            word = generated_word_id

            generated_word = self.index2word[generated_word_id]

            if generated_word == '<br>':
                generated_word = '\n'
            elif generated_word == '<par>':
                generated_word = '\n\n'

            if generated_word == '<end>':
                break

            song.append(generated_word)

        return song

    def build_graph(self):
        with tf.name_scope('placeholders'):
            self.add_placeholders_op()

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
            train_l2_loss = self.add_l2_regularizer_op(self.train_loss)
            self.train_op = self.add_train_op(train_l2_loss)

        with tf.name_scope('validation'):
            validation_logits = self.add_logits_op(validation_data, validation_size, reuse=True)
            self.validation_loss = self.add_loss_op(
                validation_logits, validation_labels, validation_size)

        with tf.name_scope('generate'):
            generate_size = np.array([1])
            generate_logits = self.add_logits_op(
                self.data_placeholder, generate_size, reuse=True)

            temperature_logits = tf.div(generate_logits, self.temperature_placeholder)
            self.generate_predictions = tf.nn.softmax(temperature_logits)
