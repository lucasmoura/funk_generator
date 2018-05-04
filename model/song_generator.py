import numpy as np
import tensorflow as tf


class GreedySongGenerator:

    def __init__(self, model):
        self.model = model

    def generate(self, sess, num_out=200):
        state = sess.run(self.model.cell.zero_state(1, tf.float32))
        word = self.model.word2index['<begin>']

        song = []
        current_word = None

        for i in range(num_out):
            probs, state = self.model.predict(sess, state, word)

            probs = probs[0].reshape(-1)
            repetition_counter = 0

            while True:
                generated_word_id = np.random.choice(
                    np.arange(len(probs)), p=probs)
                generated_word = self.model.index2word.get(generated_word_id, 1)

                if generated_word == '<end>' and len(song) < 100:
                    continue

                if generated_word != current_word:
                    current_word = generated_word
                    repetition_counter = 0
                else:
                    repetition_counter += 1

                if repetition_counter >= 5:
                    continue

                if generated_word != '<UNK>':
                    break

            word = generated_word_id

            if generated_word[0].isupper():
                generated_word = '\n' + generated_word
            elif generated_word == '<end>':
                break

            song.append(str(generated_word))

        print(' '.join(song))


class BeamSearchSongGenerator:

    def __init__(self, model):
        self.model = model

    def predict_samples(self, sess, samples, states):
        probs = []
        next_states = []

        for i in range(len(samples)):
            prob, next_state = self.model.predict(sess, states[i], samples[i][-1])
            probs.append(prob.squeeze())
            next_states.append(next_state)

        return np.array(probs), next_states

    def search(self, sess, num_samples=10, num_out=200):
        state = sess.run(self.model.cell.zero_state(1, tf.float32))
        prime_word = self.model.word2index['<begin>']

        prime_score = 0
        probs, prime_state = self.model.predict(sess, state, prime_word)
        probs = probs.squeeze()[:, None].T

        dead_num = 0  # samples that reached eos
        dead_samples = []
        dead_scores = []
        dead_states = []

        live_num = 1  # samples that did not yet reached eos
        live_samples = [[prime_word]]
        live_scores = [prime_score]
        live_states = [prime_state]

        while live_num and dead_num < num_samples:
            # total score for every sample is sum of -log of word prb
            cand_scores = np.array(live_scores)[:, None] - np.log(probs)

            cand_scores[:, self.model.word2index['<UNK>']] = 1e20
            cand_flat = cand_scores.flatten()

            # find the best (lowest) scores we have from all possible samples and new words
            ranks_flat = cand_flat.argsort()[:(num_samples - dead_num)]
            live_scores = cand_flat[ranks_flat]

            # append the new words to their appropriate live sample
            voc_size = probs.shape[1]
            live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_flat]
            live_states = [live_states[r // voc_size] for r in ranks_flat]

            # live samples that should be dead are...
            zombie = [
                s[-1] == self.model.word2index['<end>'] or len(s) >= num_out
                for s in live_samples
            ]

            # add zombies to the dead
            dead_samples += [s for s, z in zip(live_samples, zombie) if z]
            dead_scores += [s for s, z in zip(live_scores, zombie) if z]
            dead_states += [s for s, z in zip(live_states, zombie) if z]
            dead_num = len(dead_samples)
            # remove zombies from the living
            live_samples = [s for s, z in zip(live_samples, zombie) if not z]
            live_scores = [s for s, z in zip(live_scores, zombie) if not z]
            live_states = [s for s, z in zip(live_states, zombie) if not z]
            live_num = len(live_samples)

            # Finally, compute the next-step probabilities and states.
            probs, live_states = self.predict_samples(sess, live_samples, live_states)

        return dead_samples + live_samples, dead_scores + live_scores

    def generate(self, sess, num_samples=4, num_out=200):
        samples, scores = self.search(sess, num_samples=4, num_out=200)
        song = samples[np.argmin(scores)]
        print(scores[np.argmin(scores)])

        print(' '.join([self.model.index2word[index] for index in song]))
