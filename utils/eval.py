import configparser
import os
import tensorflow as tf
import numpy as np


class EvalHelper:

    def __init__(self, model, session, logger):
        self._model = model
        self._session = session
        self.cur_loss = float("inf")
        self.dev_accuracies = []
        self.dev_loss = []
        self.test_accuracies = []
        self.test_loss = []
        self.logger = logger

    def _evaluate(self, ev_handle):
        # reset metrtrics
        self._session.run(self._model.metrics_init_op)
        losses = []
        step = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        while True:
            try:
                _, _, _, _, loss = self._model.count_stats(ev_handle)
                losses.append(loss)
                if step % 10 == 0:
                    TP, FP, FN, TN = self._model.get_stats()
                    self.logger.info('EVALUATION {}, Accuracy {:.3f}, Loss {:.3f}'.format(step,
                                                                                          (TP / (TP + FN) + TN / (
                                                                                                      TN + FP)) / 2,
                                                                                          loss))
                step += 1
            except tf.errors.OutOfRangeError:
                TP, FP, FN, TN = self._model.get_stats()
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2 * precision * recall / (precision + recall)
                accuracy_run = (TP / (TP + FN) + TN / (TN + FP)) / 2
                self.logger.info('TP - {},\tFP - {},\tFN - {},\tTN - {}'.format(TP, FP, FN, TN))
                self.logger.info(
                    'Evaluation finished with prec {:.3f} recall {:.3f} f1 {:.3f}, accuracy {:.3f}'.format(precision,
                                                                                                           recall,
                                                                                                           f1,
                                                                                                           accuracy_run))
                loss_run = np.mean(losses)
                break
        return accuracy_run, loss_run

    def get_mean_dev_stats(self):
        return np.mean(self.dev_accuracies), np.mean(self.dev_loss)

    def get_mean_test_stats(self):
        return np.mean(self.test_accuracies), np.mean(self.test_loss)

    def evaluate_dev(self, handle, global_step, logger):
        dev_accuracy, dev_loss = self._evaluate(handle)
        self.dev_accuracies.append(dev_accuracy)
        self.dev_loss.append(dev_loss)
        logger.info('Step {}, running full validation with accuracy {:.5f}, loss {:.5f}'.format(global_step,
                                                                                                dev_accuracy,
                                                                                                dev_loss))
        return dev_accuracy, dev_loss

    def evaluate_test(self, handle, logger):
        test_accuracy, test_loss = self._evaluate(handle)
        self.test_accuracies.append(test_accuracy)
        self.test_loss.append(test_loss)
        logger.info('Evaluating test set {:.5f} Loss {:.3f}'.format(test_accuracy, test_loss))
        return test_accuracy, test_loss

    def save_dev_evaluation(self, model_path, epoch_time):
        mean_dev_acc = np.mean(self.dev_accuracies)
        mean_dev_loss = np.mean(self.dev_loss)
        last_dev_acc = self.dev_accuracies[-1]

        config = configparser.ConfigParser()
        config.add_section('EVALUATION')
        config.set('EVALUATION', 'MEAN_DEV_ACC', str(mean_dev_acc))
        config.set('EVALUATION', 'MEAN_DEV_LOSS', str(mean_dev_loss))
        config.set('EVALUATION', 'LAST_DEV_ACC', str(last_dev_acc))
        config.set('EVALUATION', 'LAST_DEV_LOSS', str(self.dev_loss[-1]))
        config.set('EVALUATION', 'TRAINING_TIME', str(epoch_time))

        with open(os.path.join(model_path, 'eval_dev.ini'), 'w') as configfile:  # save
            config.write(configfile)

    def save_test_evaluation(self, model_path):
        test_acc = self.test_accuracies[-1]

        config = configparser.ConfigParser()
        config.add_section('EVALUATION')
        config.set('EVALUATION', 'TEST_ACC', str(test_acc))
        config.set('EVALUATION', 'TEST_LOSS', str(self.test_loss[-1]))

        with open(os.path.join(model_path, 'eval_test.ini'), 'w') as configfile:  # save
            config.write(configfile)
