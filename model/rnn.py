import tensorflow as tf


from model.song_model import ModelConfig, SongLyricsModel


class RecurrentConfig(ModelConfig):

    def __init__(self, model_params):
        super().__init__(model_params)

        self.num_layers = model_params['num_layers']
        self.num_units = model_params['num_units']
        self.lstm_output_dropout = model_params['lstm_output_dropout']
        self.min_val = model_params['min_val']
        self.max_val = model_params['max_val']


class RecurrentModel(SongLyricsModel):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

    def add_embeddings_op(self, data_batch):
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

    def add_logits_op(self, data_batch, size_batch, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            data_embeddings = self.add_embeddings_op(data_batch)

            with tf.name_scope('recurrent_layer'):
                cell = tf.nn.rnn_cell.LSTMCell(self.config.num_units)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(
                    cell, output_keep_prob=self.config.lstm_output_dropout)

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

                batch_size = tf.shape(data_batch)[0]
                max_len = tf.shape(data_batch)[1]

                logits = tf.reshape(
                    flat_logits, [batch_size, max_len, self.config.vocab_size])

            return logits

    def add_loss_op(self, logits, labels_batch, size_batch):
        with tf.name_scope('loss'):
            weights = tf.sequence_mask(size_batch, dtype=tf.float32)

            seq_loss = tf.contrib.seq2seq.sequence_loss(
                logits=logits,
                targets=labels_batch,
                weights=weights
            )

            loss = tf.reduce_sum(seq_loss)

        return loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        optimizer_op = optimizer.minimize(loss)

        return optimizer_op
