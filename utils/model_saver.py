import os

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs


class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final variables'''

    def __init__(self, model_dir, model_name, checkpoints_to_keep=10):
        '''hook constructor

        Args:
            filename: where the model will be saved
            variables: the variables that will be saved'''

        self.checkpoints_to_keep = checkpoints_to_keep
        self.model_path = os.path.join(model_dir, model_name)
        os.makedirs(self.model_path, exist_ok=True)

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(max_to_keep=self.checkpoints_to_keep)

    def end(self, session):
        '''this will be run at session closing'''

        self._saver.save(session, self.model_path)