import pickle

from tensorflow.contrib import learn


def get_vocabulary(text_array, min_frequency):
    def tokenizer_fn(iterator):
        return (x.split(' ') for x in iterator)

    max_size = max([len(review) for review in text_array])
    text_array = [' '.join(text) for text in text_array]

    vocabulary_processor = learn.preprocessing.VocabularyProcessor(
        max_size, tokenizer_fn=tokenizer_fn, min_frequency=min_frequency)

    vocabulary_processor.fit(text_array)

    vocab = vocabulary_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    return sorted_vocab


def create_word_dictionaties(vocab):
    word2index = {word: index + 1 for (word, index) in vocab}
    index2word = {index: word for word, index in word2index.items()}

    return word2index, index2word


def save(data, data_path):
    with data_path.open('wb') as data_file:
        pickle.dump(data, data_file)


def replace_unk_words(dataset, word2index):
    for text in dataset:
        for index_word, word in enumerate(text[:]):
            if word not in word2index:
                text[index_word] = '<UNK>'


def replace_words_with_ids(dataset, word2index):
    word_id_dataset = []

    for data in dataset:
        word_id = [word2index[word] for word in data]
        word_id_dataset.append(word_id)

    return word_id_dataset


def get_sizes_list(dataset):
    return [len(data) for data in dataset]


def create_chunks(dataset, chunk_max_size):
    chunks_dataset = []

    for song in dataset:
        chunks = [song[x:x + chunk_max_size]
                  for x in range(0, len(song), chunk_max_size)]
        chunks_dataset.extend(chunks)

    return chunks_dataset


def create_labels(dataset):
    new_dataset = []
    labels = []

    for data in dataset:
        new_dataset.append(data[0:-1])
        labels.append(data[1:])

    return new_dataset, labels
