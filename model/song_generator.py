import numpy as np
import tensorflow as tf


class GreedySongGenerator:

    def __init__(self, model):
        self.model = model

    def parse_song(self, song_list):
        parsed_song = []
        is_mc = False

        for index in range(len(song_list)):
            curr_word = song_list[index]

            if curr_word[0].isupper():
                parsed_song.append('\n')

            if is_mc:
                parsed_song.append('Neural')
                is_mc = False
            else:
                parsed_song.append(curr_word)

            if curr_word.lower() == 'mc':
                is_mc = True

        print(' '.join(parsed_song))

    def weighted_pick(self, weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        return(int(np.searchsorted(t, np.random.rand(1)*s)))

    def create_initial_state(self, sess):
        state = sess.run(self.model.cell.zero_state(1, tf.float32))
        word = self.model.word2index['<begin>']

        return state, word

    def create_prime_state(self, sess, prime_words, temperature):
        state = sess.run(self.model.cell.zero_state(1, tf.float32))

        for word in prime_words:
            id_word = self.model.word2index.get(word, -1)

            if id_word == -1:
                continue

            probs, state = self.model.predict(sess, state, id_word, temperature)

        while True:
            generated_word_id = self.weighted_pick(probs)

            if generated_word_id != 1:
                break

        return state, generated_word_id

    def generate(self, sess, prime_words=None, temperature=0.7, num_out=200):
        song = []
        current_word = "<UNK>"
        repetition_counter = 0

        sequences = []
        sequence = ""
        restart = False

        if prime_words:
            state, word = self.create_prime_state(sess, prime_words, temperature)
            song.extend(prime_words)
        else:
            state, word = self.create_initial_state(sess)

        for i in range(num_out):
            probs, state = self.model.predict(sess, state, word, temperature)
            probs = probs[0].reshape(-1)

            while True:
                generated_word_id = self.weighted_pick(probs)
                generated_word = self.model.index2word.get(generated_word_id, 1)

                if generated_word == '<UNK>':
                    continue

                if generated_word == '<end>' and len(song) < 100:
                    continue

                if generated_word.lower() != current_word.lower():
                    current_word = generated_word
                    repetition_counter = 0
                elif current_word != '<UNK>':
                    repetition_counter += 1

                if repetition_counter >= 5:
                    continue

                if generated_word != '<UNK>':
                    break

            word = generated_word_id

            if generated_word[0].isupper():

                if sequences.count(sequence) >= 3:
                    state, word = self.create_initial_state(sess)
                    restart = True

                sequences.append(sequence)
                sequence = generated_word

                if restart:
                    restart = False
                    continue

            else:
                sequence += generated_word

            if generated_word == '<end>':
                break

            song.append(str(generated_word))

        return self.parse_song(song)
