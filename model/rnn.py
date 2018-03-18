import tensorflow as tf


class RecurrentConfig:

    def __init__(self, num_layers, num_units,
                 vocab_size, embedding_size, min_val, max_val,
                 lstm_output_drop):

        self.num_layers = num_layers
        self.num_units = num_units
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.min_val = min_val
        self.max_val = max_val
        self.lstm_output_drop = lstm_output_drop


class RecurrentModel:

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def get_embeddings(self, data_batch):
        with tf.name_scope('embeddings'):
            embeddings = tf.get_variable(
                'embeddings',
                initializer=tf.random_uniform_initializer(
                    minval=self.config.min_val, maxval=self.config.max_val),
                shape=(self.config.vocab_size, self.config.embedding_size),
                dtype=tf.float32
            )

            inputs = tf.nn.embedding_lookup(
                embeddings, data_batch)

        return inputs

    def get_logits(self, data_batch, labels_batch, size_batch, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            data_embeddings = self.get_embeddings(data_batch)

            with tf.name_scope('recurrent layer'):
                cell = tf.nn.rnn_cell.LSTMCell(self.config.num_units)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=self.config.lstm_output_drop)

                lstm_layers = tf.nn.rnn_cell.MultiRNNCell(
                    [drop_cell] * self.config.num_layers)

                outputs, _ = tf.nn.dynamic_rnn(
                    lstm_layers,
                    data_embeddings,
                    sequence_length=size_batch,
                    dtype=tf.float32
                )

            with tf.name_scope('logits'):
                flat_outputs = tf.reshape(outputs, [-1, self.config.num_units])

                weights = tf.get_variable(
                    'weights',
                    initializer=tf.random_uniform_initializer(
                        minval=self.config.min_val, maxval=self.config.max_val),
                    shape=(self.config.num_units, self.config.vocab_size),
                    dtype=tf.float32)
                bias = tf.get_variable(
                    'bias',
                    initializer=tf.ones_initializer(),
                    shape=(self.config.vocab_size),
                    dtype=tf.float32)

                flat_logits = tf.matmul(flat_outputs, weights) + bias

                batch_size = tf.shape(labels_batch)[0]
                max_len = tf.shape(labels_batch)[1]

                logits = tf.reshape(
                    flat_logits, [batch_size, max_len, self.config.vocab_size])

            return logits

    def build_graph(self):

        with tf.name_scope('iterator'):
            train_iterator = self.train_iterator
            validation_iterator = self.dataset.validation_iterator
            test_iterator = self.dataset.test_iterator

        with tf.names_scope('train data'):
            data_batch, labels_batch, size_batch = train_iterator.get_next()

        with tf.name_scope('train'):
            train_logits = self.get_logitst(
                data_batch, labels_batch, size_batch)
