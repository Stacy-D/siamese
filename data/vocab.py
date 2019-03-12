import os
from collections import defaultdict

import pandas as pd
import numpy as np
from bpemb import BPEmb
from utils.config_helpers import MainConfig

PAD = "<PAD>"
UNK = "<UNK>"
COL_NAMES = ['is_duplicate', 'question1', 'question2', 'id', 'fake']
SEQ_LEN_THRESHOLD = 0.999


class Vocabulary(object):
    def __init__(self):
        self.vocab = defaultdict(self.next_value)
        self.vocab[PAD] = 0
        self.vocab[UNK] = 1
        self.next = 1
        self.seq_lengths = []

    def next_value(self):
        self.next += 1
        return self.next

    @property
    def size(self):
        return len(self.vocab)

    def process_sentence(self, sentence):
        self.seq_lengths.append(len(sentence))
        for word in sentence:
            # dummy call to initialise defaultdict key for word
            self.vocab[word]

    @property
    def max_seq_length(self):
        if len(self.seq_lengths) == 1:
            return self.seq_lengths[0]
        sort_lengths = sorted(self.seq_lengths)
        num_sents = len(sort_lengths)
        return sort_lengths[round(SEQ_LEN_THRESHOLD * num_sents) - 1]


def check_file_exists(directory, filename):
    return os.path.exists(os.path.join(directory, filename))


def prepare_file(df, vocab, bpemb, join_vocab=True):
    for index, row in df.iterrows():
        q1 = bpemb.encode(row[1])
        q2 = bpemb.encode(row[2])
        if isinstance(q1[0], list):
            q1 = [subword for word in q1 for subword in word]
        if isinstance(q2[0], list):
            q2 = [subword for word in q1 for subword in word]
        if join_vocab:
            vocab.process_sentence(q1)
            vocab.process_sentence(q2)
        # update question fields with bpe version
        df.at[index, 'question1'] = ' '.join(q1)
        df.at[index, 'question2'] = ' '.join(q2)


def create_vocab(main_cfg, bpemb):
    vocab = Vocabulary()

    # load files with non-bpe data
    train_data = pd.read_csv(os.path.join(main_cfg.data_dir, main_cfg.train_file), header=None, sep='\t',
                             names=COL_NAMES).dropna()
    print('TRAIN DATA WAS LOADED')
    train_data.fake = train_data.fake.astype(int)
    dev_data = pd.read_csv(os.path.join(main_cfg.data_dir, main_cfg.dev_file), header=None, sep='\t',
                           names=COL_NAMES).dropna()
    dev_data.fake = dev_data.fake.astype(int)
    test_data = pd.read_csv(os.path.join(main_cfg.data_dir, main_cfg.test_file), header=None, sep='\t',
                            names=COL_NAMES).dropna()
    test_data.fake = test_data.fake.astype(int)

    # apply bpe, add to vocab, update data frames
    prepare_file(train_data, vocab, bpemb)
    prepare_file(dev_data, vocab, bpemb)
    prepare_file(test_data, vocab, bpemb, join_vocab=False)

    # save files with bpe data
    train_data.to_csv(os.path.join(main_cfg.data_dir, main_cfg.train_bpe_file), header=False, sep='\t', index=False)
    dev_data.to_csv(os.path.join(main_cfg.data_dir, main_cfg.dev_bpe_file), header=False, sep='\t', index=False)
    test_data.to_csv(os.path.join(main_cfg.data_dir, main_cfg.test_bpe_file), header=False, sep='\t', index=False)
    return vocab


def write_vocab(main_cfg, vocab):
    with open(os.path.join(main_cfg.data_dir, main_cfg.vocab_file), "w") as outfile:
        with open(os.path.join(main_cfg.data_dir, "max_seq_length.txt"), "w") as outfile2:
            outfile2.write("{}\n".format(vocab.max_seq_length))
            for word in vocab.vocab.keys():
                outfile.write("{}\n".format(word))


def load_vocab(main_cfg):
    vocab = Vocabulary()
    with open(os.path.join(main_cfg.data_dir, main_cfg.vocab_file), "r") as infile:
        with open(os.path.join(main_cfg.data_dir, "max_seq_length.txt"), "r") as infile2:
            for line in infile.readlines():
                vocab.process_sentence(line.rstrip().split())
            vocab.seq_lengths = [infile2.readline().rstrip()]
    return vocab


def create_embeddings(main_cfg, vocab):
    # initialise random embeddings (will stay for UNK and vocab entries not in pretrained model)
    embeddings = 1 * np.random.randn(vocab.size, main_cfg.embedding_size).astype(np.float32)
    for word, index in vocab.vocab.items():
        if word == PAD:
            embeddings[index] = np.zeros(main_cfg.embedding_size)
        else:
            try:
                embeddings[index] = np.float32(bpemb.embed(word).flatten())
            except:
                pass
    np.save(os.path.join(main_cfg.embeddings[:-4]), embeddings)


def get_vocab(main_config, args, logger):
    main_cfg = MainConfig(main_config, args)
    vocab_loaded = False
    # if we don't have vocab or bpe files, create everything from scratch
    if check_file_exists(main_cfg.data_dir, main_cfg.vocab_file) and \
            check_file_exists(main_cfg.data_dir, main_cfg.train_bpe_file) and \
            check_file_exists(main_cfg.data_dir, main_cfg.dev_bpe_file) and \
            check_file_exists(main_cfg.data_dir, main_cfg.test_bpe_file):
        logger.info('"{}" and bpe files found in data folder, loading stats.'.format(main_cfg.vocab_file))
        vocab = load_vocab(main_cfg)
        vocab_loaded = True
    else:
        logger.info('No "{}" or bpe files found in data folder, creating new vocab.'.format(main_cfg.vocab_file))
        bpemb = BPEmb(lang="en", dim=main_cfg.embedding_size, vs=main_cfg.emb_vocab_size)
        logger.info('Bemb loaded')
        vocab = create_vocab(main_cfg, bpemb)
        write_vocab(main_cfg, vocab)

    logger.info('Max sequence length of {} covers {}% of sentences'.format(vocab.max_seq_length, SEQ_LEN_THRESHOLD))

    # if vocab is created new we should probably also create emebeddings again
    if not vocab_loaded or not check_file_exists("", main_cfg.embeddings):
        logger.info('No embedding matrix found, loading embeddings.')
        create_embeddings(main_cfg, vocab)
        logger.info('Saved embedding matrix to "{}" in data folder.'.format(main_cfg.embeddings))
    else:
        logger.info('Embedding matrix found in data folder.')

    return int(vocab.max_seq_length), int(vocab.size)
