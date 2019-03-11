from argparse import ArgumentParser
import tensorflow as tf
import os
from data.quora_dataset import get_quora_datasets, get_help_dataset
from data.vocab import get_vocab
from utils.config_helpers import MainConfig
from utils.log_saver import LogSaver
from utils.eval import EvalHelper
from utils.model_saver import SaveAtEnd
from utils.other_utils import timer, set_visible_gpu, init_config
from utils.best_saver import BestCheckpointSaver, get_best_checkpoint
from models.bilstm import BiLSTMSiamese
import time

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging


def train(main_config, args):
    main_cfg = MainConfig(main_config, args)
    main_cfg.max_sequence_len = args.max_seq_length
    main_cfg.syn_weight = args.syn_weight
    main_cfg.vocab_size = args.vocab_size
    batch_size = tf.placeholder(tf.int64)
    vocab = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(main_cfg.data_dir,
                                                                                 main_cfg.vocab_file),
                                                    num_oov_buckets=0,
                                                    default_value=1)
    with tf.device('/cpu:0'):
        train_iter, dev_iter, test_iter, dev_run_iter = get_quora_datasets(main_cfg, batch_size, vocab)
    train_handle, test_handle, dev_handle, dev_run_handle = train_iter.string_handle(), \
                                                            test_iter.string_handle(), \
                                                            dev_iter.string_handle(), dev_run_iter.string_handle()

    model_name = 'bilstm_{}'.format(main_config['PARAMS']['embedding_size'])

    # Switcher handle placeholder, iterator
    handle = tf.placeholder(tf.string, shape=[], name='handle')
    quora_iter = tf.data.Iterator.from_string_handle(handle,
                                                     train_iter.output_types,
                                                     train_iter.output_shapes)
    quora_example = quora_iter.get_next()
    # obtaining finished
    step = tf.train.get_or_create_global_step()
    main_config['DATA']['emb_path'] = os.path.join(args.data_dir, main_config['DATA']['embeddings'])
    model = BiLSTMSiamese(quora_example, args, main_config)
    best_loss = float("inf")
    best_ckpt_saver = BestCheckpointSaver(
        save_dir=args.model_dir,
        num_to_keep=main_cfg.checkpoints_to_keep,
        maximize=True
    )
    best_accuracy = best_ckpt_saver.get_best_accuracy()

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    saver = tf.train.Saver()
    with tf.Session(config=config) as session:
        session.run(tf.tables_initializer())
        saver.restore(session, get_best_checkpoint(args.model_dir, select_maximum_value=True))
        model.set_session(session)
        model_evaluator = EvalHelper(model, session, logger)
        session.run(dev_iter.initializer, feed_dict={batch_size: main_cfg.eval_batch_size})
        dev_handle = session.run(dev_handle)
        all_dev_acc, all_dev_loss = model_evaluator.evaluate_dev(dev_handle, 1, logger)
        test_handle = session.run(test_handle)
        session.run(test_iter.initializer, feed_dict={batch_size: main_cfg.eval_batch_size})
        test_acc, test_loss = model_evaluator.evaluate_test(test_handle, logger)
        model_evaluator.save_test_evaluation(args.model_dir)


def main():
    parser = ArgumentParser()

    parser.add_argument('--data-dir',
                        default='./corpora',
                        help='Path to original quora split')

    parser.add_argument('--model-dir',
                        default='./model_dir',
                        help='Path to save the trained model')

    parser.add_argument('--use-help',
                        choices=[True, False],
                        default=False,
                        type=bool,
                        help='should model use help on difficult examples')

    parser.add_argument('--gpu',
                        default='0',
                        help='index of GPU to be used (default: %(default))')

    parser.add_argument('--embeddings',
                        choices=['no', 'fixed', 'tunable'],
                        default='no',
                        type=str,
                        help='embeddings')
    parser.add_argument('--batch-size',
                        choices=[4, 128, 256, 512],
                        default=128,
                        type=int,
                        help='batch size')
    parser.add_argument('--syn-weight',
                        default=1,
                        type=float,
                        help='Weight for loss function')

    args = parser.parse_args()
    logger.info(args)
    if args.embeddings == 'no':
        args.use_embed = False
        args.tune = False
    else:
        args.use_embed = True
        args.tune = False if args.embeddings == 'fixed' else True

    set_visible_gpu(args.gpu)
    args.model_dir = '{}_bilstm_{}_{}_{}'.format(args.model_dir,
                                                 args.embeddings,
                                                 args.batch_size,
                                                 args.syn_weight
                                                 )

    main_config = init_config()

    args.max_seq_length, args.vocab_size = get_vocab(main_config, args, logger)
    logger.info(args)
    train(main_config, args)


if __name__ == '__main__':
    main()
