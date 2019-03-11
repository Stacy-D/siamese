import configparser
import os

import tensorflow as tf


logger = tf.logging


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def set_visible_gpu(gpu_number: str):
    logger.info('Setting visible GPU to {}'.format(gpu_number))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


def init_config(specific=None):
    config = configparser.ConfigParser()
    if specific is None:  # return main config
        logger.info('Reading main configuration.')
        config.read('config/main.ini')
    else:
        config.read('config/model/{}.ini'.format(specific))
        logger.info('Reading configuration for {} model.'.format(specific))
    return config



