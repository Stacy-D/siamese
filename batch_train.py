from argparse import ArgumentParser
import tensorflow as tf
import os
from data.quora_dataset import get_quora_datasets, get_help_dataset, get_help_batch
from data.vocab import get_vocab
from utils.config_helpers import MainConfig
from utils.log_saver import LogSaver
from utils.eval import EvalHelper
from utils.model_saver import SaveAtEnd
from utils.other_utils import timer, set_visible_gpu, init_config
from utils.best_saver import BestCheckpointSaver, get_best_checkpoint
from models.bilstm import BiLSTMSiamese
import time
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging


def train(main_config, args):
    def use_sublearning(cur_ids, losses, bilstm):
        df_ = get_help_batch(cur_ids, losses, args, help_dataset)
        bilstm.session.run(help_iter.initializer,
                           feed_dict={'help_labels:0': df_['is_duplicate'].values,
                                      'help_sent1:0': df_['question1'].values,
                                      'help_sent2:0': df_['question2'].values,
                                      'help_ids:0': df_['id'].values,
                                      batch_size: args.batch_size
                                      }
                           )
        # run help step
        loss, _, current_quora_ids, _ = model.run_train_batch(help_handle)
        return

    main_cfg = MainConfig(main_config, args)
    main_cfg.max_sequence_len = args.max_seq_length
    main_cfg.syn_weight = args.syn_weight
    main_cfg.vocab_size = args.vocab_size
    batch_size = tf.placeholder(tf.int64)
    help_dataset = pd.read_csv(os.path.join(args.data_dir, args.fake_data),
                               header=None,
                               sep='\t',
                               names=['is_duplicate', 'question1', 'question2', 'id', 'fake'])
    vocab = tf.contrib.lookup.index_table_from_file(vocabulary_file=os.path.join(main_cfg.data_dir,
                                                                                 main_cfg.vocab_file),
                                                    num_oov_buckets=0,
                                                    default_value=1)
    with tf.device('/cpu:0'):
        train_iter, dev_iter, test_iter, dev_run_iter = get_quora_datasets(main_cfg, batch_size, vocab)
        help_iter = get_help_dataset(main_cfg, batch_size, vocab)
    train_handle, test_handle, dev_handle, dev_run_handle, help_handle = train_iter.string_handle(), \
                                                                         test_iter.string_handle(), \
                                                                         dev_iter.string_handle(), dev_run_iter.string_handle(), \
                                                                         help_iter.string_handle()

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
    with tf.train.MonitoredTrainingSession(checkpoint_dir=main_cfg.model_dir,
                                           save_checkpoint_steps=main_cfg.save_every,
                                           config=config,
                                           save_summaries_steps=0, save_summaries_secs=None,
                                           log_step_count_steps=0,
                                           hooks=[
                                               SaveAtEnd(args.model_dir, model_name,
                                                         main_cfg.checkpoints_to_keep)]) as session:
        model.set_session(session)
        log_saver = LogSaver(args.model_dir, 'summaries', session.graph)
        model_evaluator = EvalHelper(model, session, logger)
        start_time = time.time()
        patience_left = main_cfg.patience
        train_handle, dev_handle, dev_run_handle = session.run([train_handle, dev_handle, dev_run_handle])
        help_handle = session.run(help_handle)
        session.run(train_iter.initializer, feed_dict={batch_size: main_cfg.batch_size})
        session.run(dev_run_iter.initializer, feed_dict={batch_size: main_cfg.eval_batch_size})
        logger.info('Starting training...')
        warm_up=2
        warmed= False
        while patience_left:
            try:
                # run train batch
                loss, _, current_quora_ids, _ = model.run_train_batch(train_handle)
                global_step = session.run(step)
                # use sublearning
                use_sublearning(current_quora_ids, loss, model)
                global_step = session.run(step)
                if global_step % main_cfg.eval_every == 0:
                    model.evaluation_stats(train_handle, dev_run_handle, global_step, logger, log_saver)

            except tf.errors.OutOfRangeError:
                session.run(dev_iter.initializer, feed_dict={batch_size: main_cfg.eval_batch_size})
                all_dev_acc, all_dev_loss = model_evaluator.evaluate_dev(dev_handle, global_step, logger)
                session.run(model.inc_gstep)  # decay LR
                warm_up-=1
                warmed=(warm_up<0)
                if all_dev_acc > best_accuracy:
                    best_accuracy = all_dev_acc
                    best_ckpt_saver.handle(all_dev_acc, session, step)
                    patience_left = main_cfg.patience
                else:
                    patience_left -= 1
                    logger.info('Patience left {}'.format(patience_left))
                if best_loss > all_dev_loss:
                    best_loss = all_dev_loss

                session.run(dev_iter.initializer, feed_dict={batch_size: main_cfg.eval_batch_size})
                session.run(train_iter.initializer, feed_dict={batch_size: main_cfg.batch_size})

        logger.info('No improvement observed over {} epochs. Initiating early stopping'.format(main_cfg.patience))
        end_time = time.time()
        total_time = timer(start_time, end_time)
        logger.info('Training took {}, best accuracy is {}'.format(total_time, best_accuracy))
        # model_evaluator.save_dev_evaluation(args.model_dir, total_time)
    with open(os.path.join(args.model_dir, 'run_config.ini'), 'w') as configfile:  # save
        main_config.write(configfile)

    saver = tf.train.Saver()
    with tf.Session(config=config) as session:
        session.run(tf.tables_initializer())
        saver.restore(session, get_best_checkpoint(args.model_dir, select_maximum_value=True))
        model.set_session(session)
        model_evaluator = EvalHelper(model, session, logger)
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
                        default=1.0,
                        type=float,
                        help='Weight for loss function')
    parser.add_argument('--fake-data',
                        default='fake.bpe.tsv',
                        type=str,
                        help='name of fake data file')
    parser.add_argument('--examples',
                        default=5,
                        type=int,
                        help='number of examples per hard pair')
    args = parser.parse_args()
    logger.info(args)
    if args.embeddings == 'no':
        args.use_embed = False
        args.tune = False
    else:
        args.use_embed = True
        args.tune = False if args.embeddings == 'fixed' else True

    set_visible_gpu(args.gpu)
    args.model_dir = '{}_batch_bilstm_{}_{}_{}'.format(args.model_dir,
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
