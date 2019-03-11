import tensorflow as tf
import os
import pandas as pd


def load_quora_dataset(csv_path, vocab, max_len, batch_size, weight, buffer, repeat=1):
    dataset = tf.contrib.data.CsvDataset(
        csv_path,
        [tf.int64,  # Required field, use dtype or empty tensor
         tf.string,  # Optional field, default to 0.0
         tf.string,
         tf.int32,  # Required field, use dtype or empty tensor
         tf.float32  # fake or not
         ],
        field_delim='\t',
        select_cols=[0, 1, 2, 3, 4]
    )
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=buffer).repeat(repeat)
    dataset = dataset.map(lambda label, s1, s2, id, fake: (label,
                                                           vocab.lookup(tf.string_split([s1]).values[:max_len]),
                                                           vocab.lookup(tf.string_split([s2]).values[:max_len]),
                                                           id,
                                                           (1.0 - fake * (1.0 - weight)),  # build weight vector
                                                           s1,
                                                           s2), num_parallel_calls=4)
    dataset = dataset.map(lambda label, s1, s2, id, fake, sent1, sent2: (label,
                                                                         s1,
                                                                         s2,
                                                                         id,
                                                                         fake,
                                                                         sent1,
                                                                         sent2,
                                                                         tf.size(s1),
                                                                         tf.size(s2)))
    # Convert to a batch of size 32. Padded batch appends 0 for shorter sentences.
    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(tf.TensorShape([]),
                                                                         tf.TensorShape([max_len]),
                                                                         tf.TensorShape([max_len]),
                                                                         tf.TensorShape([]),  # id
                                                                         tf.TensorShape([]),  # fake
                                                                         tf.TensorShape([]),  # sent 1
                                                                         tf.TensorShape([]),  # sent 2
                                                                         tf.TensorShape([]),  # sent 1
                                                                         tf.TensorShape([])  # sent 2
                                                                         ))
    dataset = dataset.map(lambda label, s1, s2, id, fake, sent_1, sent_2, l1, l2: (label,  # label
                                                                                   s1,  # s1 in ids
                                                                                   s2,  # s2 in ids
                                                                                   id,  # cur_ids
                                                                                   fake,  # fake
                                                                                   sent_1,  # s1 str
                                                                                   sent_2,  # s2 str
                                                                                   l1,
                                                                                   l2
                                                                                   ), num_parallel_calls=4)
    dataset = dataset.prefetch(50)
    return dataset


def get_help_dataset(main_config, batch_size, vocab):
    help_labels = tf.placeholder(shape=[None], dtype=tf.int64, name='help_labels')
    help_sent1 = tf.placeholder(shape=[None], dtype=tf.string, name='help_sent1')
    help_sent2 = tf.placeholder(shape=[None], dtype=tf.string, name='help_sent2')
    help_ids = tf.placeholder(shape=[None], dtype=tf.int32, name='help_ids')
    cool_dataset = tf.data.Dataset.from_tensor_slices((help_labels, help_sent1, help_sent2, help_ids))
    cool_dataset = cool_dataset.map(lambda label,
                                           s1,
                                           s2,
                                           id:
                                    (label,
                                     vocab.lookup(tf.string_split([s1]).values[:main_config.max_sequence_len]),
                                     vocab.lookup(tf.string_split([s2]).values[:main_config.max_sequence_len]),
                                     id,
                                     s1,
                                     s2))
    cool_dataset = cool_dataset.map(lambda label, s1, s2, id, sent_1, sent_2: (label,  # label
                                                                               s1,  # s1 in ids
                                                                               s2,  # s2 in ids
                                                                               id,  # cur_ids
                                                                               main_config.syn_weight,  # fake
                                                                               sent_1,  # s1 str
                                                                               sent_2,  # s2 str
                                                                               tf.size(s1),
                                                                               tf.size(s2)))
    cool_dataset = cool_dataset.padded_batch(batch_size=batch_size, padded_shapes=(tf.TensorShape([]),
                                                                                   tf.TensorShape(
                                                                                       [main_config.max_sequence_len]),
                                                                                   tf.TensorShape(
                                                                                       [main_config.max_sequence_len]),
                                                                                   tf.TensorShape([]),  # id
                                                                                   tf.TensorShape([]),  # fake
                                                                                   tf.TensorShape([]),  # sent 1
                                                                                   tf.TensorShape([]),  # sent 2
                                                                                   tf.TensorShape([]),  # sent 1
                                                                                   tf.TensorShape([])  # sent 2
                                                                                   ))
    cool_dataset = cool_dataset.shuffle(10000)
    return cool_dataset.make_initializable_iterator()


def get_help_batch(cur_ids, losses, args, help_dataset):
    pairs = sorted(zip(cur_ids, losses), key=lambda x: x[1], reverse=True)
    df_ = pd.DataFrame(columns=['is_duplicate', 'question1', 'question2', 'id', 'fake'])
    for pair in pairs:
        help = help_dataset[help_dataset['id'] == pair[0]]
        if len(help):
            # concat dataset, get max 5 examples on each
            df_ = pd.concat([df_, help.sample(frac=1).reset_index(drop=True)[:5]], axis=0)
            # no need to collect more then batch size
            if len(df_) > args.batch_size:
                break
    return df_


def get_quora_datasets(main_config, batch_size, vocab):
    return load_quora_dataset(os.path.join(main_config.data_dir, main_config.train_bpe_file),
                              vocab,
                              main_config.max_sequence_len,
                              batch_size,
                              weight=main_config.syn_weight,
                              buffer=1000000).make_initializable_iterator(), \
           load_quora_dataset(os.path.join(main_config.data_dir, main_config.dev_bpe_file),
                              vocab,
                              main_config.max_sequence_len,
                              batch_size,
                              weight=main_config.syn_weight,
                              buffer=10000).make_initializable_iterator(), \
           load_quora_dataset(os.path.join(main_config.data_dir, main_config.test_bpe_file),
                              vocab,
                              main_config.max_sequence_len,
                              batch_size,
                              weight=main_config.syn_weight,
                              buffer=10000).make_initializable_iterator(), \
           load_quora_dataset(os.path.join(main_config.data_dir, main_config.dev_bpe_file),
                              vocab,
                              main_config.max_sequence_len,
                              batch_size,
                              weight=main_config.syn_weight,
                              buffer=10000,
                              repeat=None).make_initializable_iterator()
    # get_help_dataset(main_config.max_sequence_len, vocab).make_initializable_iterator()
