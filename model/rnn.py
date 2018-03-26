import tensorflow as tf


from model.song_model import ModelConfig, SongLyricsModel


class RecurrentConfig(ModelConfig):

    def __init__(self, model_params):
        super().__init__(model_params)

        self.num_layers = model_params['num_layers']
        self.num_units = model_params['num_units']
        self.embedding_dropout = model_params['embedding_dropout']
        self.lstm_state_dropout = model_params['lstm_state_dropout']
        self.lstm_output_dropout = model_params['lstm_output_dropout']
        self.weight_decay = model_params['weight_decay']
        self.min_val = model_params['min_val']
        self.max_val = model_params['max_val']


class RecurrentModel(SongLyricsModel):

    def add_placeholders_op(self):
        self.embedding_dropout_placeholder = tf.placeholder(
            tf.float32, name='embedding_dropout')
        self.lstm_state_dropout_placeholder = tf.placeholder(
            tf.float32, name='lstm_state_dropout')
        self.lstm_output_dropout_placeholder = tf.placeholder(
            tf.float32, name='lstm_output_dropout')

        self.data_placeholder = tf.placeholder(tf.int32, [None, 1])
        self.temperature_placeholder = tf.placeholder(tf.float32)

    def create_train_feed_dict(self):
        feed_dict = {
            self.embedding_dropout_placeholder: self.config.embedding_dropout,
            self.lstm_state_dropout_placeholder: self.config.lstm_state_dropout,
            self.lstm_output_dropout_placeholder: self.config.lstm_output_dropout,
        }

        return feed_dict

    def create_validation_feed_dict(self):
        feed_dict = {
            self.embedding_dropout_placeholder: 1.0,
            self.lstm_state_dropout_placeholder: 1.0,
            self.lstm_output_dropout_placeholder: 1.0,
        }

        return feed_dict

    def create_generate_feed_dict(self, data, temperature, state):
        feed_dict = self.create_validation_feed_dict()

        feed_dict[self.data_placeholder] = data
        feed_dict[self.temperature_placeholder] = temperature
        feed_dict[self.initial_state] = state

        return feed_dict

    def add_embeddings_op(self, data_batch):
        with tf.name_scope('embeddings'):
            self.embeddings = tf.get_variable(
                'embeddings',
                initializer=tf.random_uniform_initializer(
                    minval=self.config.min_val, maxval=self.config.max_val),
                shape=(self.config.vocab_size, self.config.embedding_size),
                dtype=tf.float32
            )

            self.embeddings_dropout = tf.nn.dropout(
                self.embeddings, keep_prob=self.embedding_dropout_placeholder)

            inputs = tf.nn.embedding_lookup(
                self.embeddings_dropout, data_batch)

        return inputs

    def add_logits_op(self, data_batch, size_batch, reuse=False):
        variational_recurrent = False if reuse else True

        with tf.variable_scope('logits', reuse=reuse):
            data_embeddings = self.add_embeddings_op(data_batch)

            with tf.name_scope('recurrent_layer'):
                def make_cell():
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        self.config.num_units)
                    drop_cell = tf.nn.rnn_cell.DropoutWrapper(
                        lstm_cell,
                        state_keep_prob=self.lstm_state_dropout_placeholder,
                        output_keep_prob=self.lstm_output_dropout_placeholder,
                        variational_recurrent=variational_recurrent,
                        input_size=self.config.num_units,
                        dtype=tf.float32)

                    return drop_cell

                self.cell = tf.nn.rnn_cell.MultiRNNCell(
                    [make_cell() for _ in range(self.config.num_layers)])

                self.initial_state = self.cell.zero_state(
                    tf.shape(data_batch)[0], tf.float32)

                outputs, self.final_state = tf.nn.dynamic_rnn(
                    self.cell,
                    data_embeddings,
                    sequence_length=size_batch,
                    initial_state=self.initial_state,
                    dtype=tf.float32
                )

            with tf.name_scope('logits'):
                flat_outputs = tf.reshape(outputs, [-1, self.config.num_units])

                bias = tf.get_variable(
                    'bias',
                    initializer=tf.ones_initializer(),
                    shape=(self.config.vocab_size),
                    dtype=tf.float32)

                flat_logits = tf.matmul(
                    flat_outputs, tf.transpose(self.embeddings_dropout)) + bias

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

    def add_l2_regularizer_op(self, loss):
        l2_loss = self.config.weight_decay * tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        return loss + l2_loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        optimizer_op = optimizer.minimize(loss)

        return optimizer_op
