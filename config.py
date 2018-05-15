from collections import defaultdict


default_args = {
    'use_checkpoint': True,
    'embedding_size': 300,
    'num_layers': 3,
    'num_units': 728
}

all_args = {
    'checkpoint_path': 'checkpoint',
    'index2word_path': 'data/song_dataset/index2word.pkl',
    'word2index_path': 'data/song_dataset/word2index.pkl',
    'vocab_size': 12551,
}
all_args = {**default_args, **all_args}
all_args = defaultdict(int, all_args)

kondzilla_args = {
    'checkpoint_path': 'kondzilla_checkpoint',
    'index2word_path': 'kondzilla/song_dataset/index2word.pkl',
    'word2index_path': 'kondzilla/song_dataset/word2index.pkl',
    'vocab_size': 2192,
}
kondzilla_args = {**default_args, **kondzilla_args}
kondzilla_args = defaultdict(int, kondzilla_args)

proibidao_args = {
    'checkpoint_path': 'proibidao_checkpoint',
    'index2word_path': 'proibidao/song_dataset/index2word.pkl',
    'word2index_path': 'proibidao/song_dataset/word2index.pkl',
    'vocab_size': 1445,
}
proibidao_args = {**default_args, **proibidao_args}
proibidao_args = defaultdict(int, proibidao_args)

ostentacao_args = {
    'checkpoint_path': 'ostentacao_checkpoint',
    'index2word_path': 'ostentacao/song_dataset/index2word.pkl',
    'word2index_path': 'ostentacao/song_dataset/word2index.pkl',
    'vocab_size': 2035,
}
ostentacao_args = {**default_args, **ostentacao_args}
ostentacao_args = defaultdict(int, ostentacao_args)
