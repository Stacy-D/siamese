from argparse import ArgumentParser
import tensorflow as tf
import os
from data.quora_dataset import load_quora_dataset
from data.vocab import get_vocab
from models.bilstm import BiLSTMSiamese
from utils.other_utils import timer, set_visible_gpu, init_config
from utils.config_helpers import MainConfig
from utils.best_saver import BestCheckpointSaver, get_best_checkpoint
import time
import csv

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging


def score_data(main_config, args):
    main_cfg = MainConfig(main_config, args)
    main_cfg.max_sequence_len = args.max_seq_length
    main_cfg.syn_weight = args.syn_weight
    main_cfg.vocab_size = args.vocab_size

    vocab = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(main_cfg.data_dir,
                                                                                 main_cfg.vocab_file),
                                                    num_oov_buckets=0,
                                                    default_value=1)
    score_iter = load_quora_dataset(os.path.join(args.data_dir, args.file_name),
                                    vocab,
                                    args.max_seq_length,
                                    args.batch_size,
                                    weight=args.syn_weight,
                                    buffer=1000000).make_initializable_iterator()
    score_handle = score_iter.string_handle()
    # Switcher handle placeholder, iterator
    handle = tf.placeholder(tf.string, shape=[], name='handle')
    quora_iter = tf.data.Iterator.from_string_handle(handle,
                                                     score_iter.output_types,
                                                     score_iter.output_shapes)
    quora_example = quora_iter.get_next()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    model = BiLSTMSiamese(quora_example, args, main_config)
    saver = tf.train.Saver()
    start_time = time.time()
    with tf.Session(config=config) as session:
        session.run(tf.tables_initializer())
        saver.restore(session, get_best_checkpoint(args.model_dir, select_maximum_value=True))
        model.set_session(session)
        score_handle = session.run(score_handle)
        session.run(score_iter.initializer)
        with open(os.path.join(args.data_dir, args.output_file), 'w', encoding='UTF-8') as output:
            csv_writer = csv.writer(output, delimiter='\t')
            while True:
                try:
                        sent_1, sent_2, labels, current_ids, losses, _ = model.run_score_batch(score_handle)
                        print(losses.shape)
                        print(losses)
                        process_batch(sent_1, sent_2, labels, current_ids, losses, csv_writer)
                except tf.errors.OutOfRangeError:
                    break

    logger.info('Finished scoring. Took {}'.format(timer(start_time, time.time())))


def score_model(main_config, args):
    main_cfg = MainConfig(main_config, args)
    main_cfg.max_sequence_len = args.max_seq_length
    main_cfg.syn_weight = args.syn_weight
    main_cfg.vocab_size = args.vocab_size

    vocab = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(main_cfg.data_dir,
                                                                                 main_cfg.vocab_file),
                                                    num_oov_buckets=0,
                                                    default_value=1)
    score_iter = load_quora_dataset(os.path.join(args.data_dir, args.file_name),
                                    vocab,
                                    args.max_seq_length,
                                    args.batch_size,
                                    weight=args.syn_weight,
                                    buffer=10000).make_initializable_iterator()
    score_handle = score_iter.string_handle()
    # Switcher handle placeholder, iterator
    handle = tf.placeholder(tf.string, shape=[], name='handle')
    quora_iter = tf.data.Iterator.from_string_handle(handle,
                                                     score_iter.output_types,
                                                     score_iter.output_shapes)
    quora_example = quora_iter.get_next()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    model = BiLSTMSiamese(quora_example, args, main_config)
    saver = tf.train.Saver()
    start_time = time.time()
    with tf.Session(config=config) as session:
        session.run(tf.tables_initializer())
        saver.restore(session, get_best_checkpoint(args.model_dir, select_maximum_value=True))
        model.set_session(session)
        score_handle = session.run(score_handle)
        session.run(score_iter.initializer)
        session.run(model.metrics_init_op)
        while True:
            try:
                model.count_stats(score_handle)
            except tf.errors.OutOfRangeError:
                TP, FP, FN, TN = model.get_stats()
                calc_stats(TP, FP, FN, TN, logger, args.output_file)

                break

    logger.info('Finished scoring. Took {}'.format(timer(start_time, time.time())))


def calc_stats(TP, FP, FN, TN, logger, file):
    logger.info('TP - {},\tFP - {},\tFN - {},\tTN - {}'.format(TP, FP, FN, TN))
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(TP + FN + TN + FP)
    specificity = TN/(TN + FP)
    balanced_acc = (TP/(TP+FN) + TN/(TN+FP))/2
    logger.info('-----------EVALUATION STATS------------')
    logger.info('Precision: {:.5}'.format(precision*100))
    logger.info('Recall: {:.5}'.format(recall * 100))
    logger.info('F1: {:.5}'.format(f1 * 100))
    logger.info('Accuracy: {:.5}'.format(accuracy * 100))
    logger.info('Specificity: {:.5}'.format(specificity * 100))
    logger.info('Balanced accuracy: {:.5}'.format(balanced_acc * 100))
    with open(file, 'w', encoding='UTF-8') as output:
        csv_writer = csv.writer(output, delimiter='\t')
        csv_writer.writerow(['Prec', 'Recall', "F1", "ACC", 'SP', "BACC"])
        csv_writer.writerow([precision, recall, f1, accuracy, specificity, balanced_acc])



def process_batch(sents_1, sents_2, labels, current_ids, losses, writer):
    for s1, s2, label, cur_id, loss in zip(sents_1, sents_2, labels, current_ids, losses):
        writer.writerow([label,
                         s1.decode('UTF-8'),
                         s2.decode('UTF-8'),
                         cur_id,
                         loss])


def main():
    parser = ArgumentParser()

    parser.add_argument('--model',
                        default='cnn',
                        choices=['rnn', 'cnn', 'multihead'],
                        help='model to be used')

    parser.add_argument('--data-dir',
                        default='./corpora',
                        help='Path to original quora split')

    parser.add_argument('--file-name',
                        default='./fake.tsv',
                        help='Path to original quora split')

    parser.add_argument('--model-dir',
                        default='./model_dir',
                        help='Path to save the trained model')
    parser.add_argument('--gpu',
                        default='0',
                        help='index of GPU to be used (default: %(default))')

    parser.add_argument('--embeddings',
                        choices=['no', 'fixed', 'tunable'],
                        default='no',
                        type=str,
                        help='embeddings')
    parser.add_argument('--heads',
                        choices=[4, 5, 10, 20],
                        default=None,
                        type=int,
                        help='num of multi heads')
    parser.add_argument('--batch-size',
                        default=128,
                        type=int,
                        help='batch size')
    parser.add_argument('--syn-weight',
                        default=1.0,
                        type=float,
                        help='Weight for loss function')
    parser.add_argument('--output-file',
                        default='scored.tsv',
                        help='Output file with scores')
    parser.add_argument('--mode',
                        default='scoring',
                        help='mode to produce model scores')

    args = parser.parse_args()
    if args.embeddings == 'no':
        args.use_embed = False
        args.tune = False
    else:
        args.use_embed = True
        args.tune = False if args.embeddings == 'fixed' else True

    set_visible_gpu(args.gpu)

    main_config = init_config()


    args.max_seq_length, args.vocab_size = get_vocab(main_config, args, logger)
    logger.info(args)
    if args.mode == 'scoring':
        score_model(main_config, args)
    else:
        score_data(main_config, args)


if __name__ == '__main__':
    main()
