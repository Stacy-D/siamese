import tensorflow as tf
import numpy as np

max_pool = tf.contrib.keras.layers.GlobalMaxPool1D()


class BiLSTMSiamese():

    def __init__(self, example, args, main_cfg):
        self.x1, self.x2, self.labels, \
        self.current_ids, self.fake, \
        self.sent_1, self.sent_2, self.len1, self.len2 = example[1], \
                                                         example[2], \
                                                         example[0], \
                                                         example[3], \
                                                         example[4], \
                                                         example[5], \
                                                         example[6], \
                                                         example[7], \
                                                         example[8]
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.step = tf.train.get_global_step()
        self.embedding_size = main_cfg['PARAMS'].getint('embedding_size')
        self.dropout = float(main_cfg['TRAINING']['dropout'])
        gstep = tf.Variable(0, trainable=False, name='decay_step')
        self.inc_gstep = tf.assign(gstep, gstep + 1)
        self.hidden_size = main_cfg['PARAMS'].getint('hidden_size')
        self.learning_rate = tf.train.exponential_decay(
            main_cfg['TRAINING'].getfloat('learning_rate'),  # Base learning rate.
            global_step=gstep,
            decay_steps=1,
            decay_rate=0.95,  # Decay rate.
            staircase=True)
        with tf.variable_scope("train_step") as scope:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        with tf.variable_scope('embeddings'):
            # are we using pretrained embeddings
            if args.use_embed:
                word_embeddings = tf.get_variable('word_embeddings',
                                                  initializer=np.load(main_cfg['DATA'].get('emb_path')),
                                                  dtype=tf.float32,
                                                  trainable=args.tune)
            else:
                word_embeddings = tf.get_variable('word_embeddings',
                                                  shape=[args.vocab_size, self.embedding_size],
                                                  dtype=tf.float32,
                                                  trainable=True)

            self.embedded_x1 = tf.nn.embedding_lookup(word_embeddings, self.x1)
            self.embedded_x2 = tf.nn.embedding_lookup(word_embeddings, self.x2)
            we1 = tf.layers.dropout(self.embedded_x1, rate=self.dropout,
                                    training=tf.convert_to_tensor(self.is_training))
            we2 = tf.layers.dropout(self.embedded_x2, rate=self.dropout,
                                    training=tf.convert_to_tensor(self.is_training))

        self.siamese(we1, we2)

        ### Loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.fc2,
                                                                       weights=self.fake)
        self.loss_per_example_flat = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.fc2,
                                                                       weights=self.fake,
                                                                        reduction=tf.losses.Reduction.NONE)

        # self.flat_losses = tf.squeeze(self.loss_per_example, axis=1)
        self.train_step_inf = optimizer.minimize(self.loss, global_step=self.step)

        ### Evaluation
        with tf.variable_scope('metrics'):
            self.temp_sim = tf.argmax(self.fc2, 1)
            self.correct_predictions = tf.equal(tf.argmax(self.fc2, 1), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))
        with tf.variable_scope('scoring') as scope:
            self.FN, self.fn_update = tf.metrics.false_negatives(labels=self.labels,
                                                                 predictions=self.temp_sim)
            self.TN, self.tn_update = tf.metrics.true_negatives(labels=self.labels,
                                                                predictions=self.temp_sim)
            self.TP, self.tp_update = tf.metrics.true_positives(labels=self.labels,
                                                                predictions=self.temp_sim)
            self.FP, self.fp_update = tf.metrics.false_positives(labels=self.labels,
                                                                 predictions=self.temp_sim)
            self.running_vars = tf.contrib.framework.get_variables(collection=tf.GraphKeys.LOCAL_VARIABLES, scope=scope)
            self.metrics_init_op = tf.variables_initializer(self.running_vars)

    def bilstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        cell_bw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, seq, sequence_length=seq_len,
                                                                        dtype=tf.float32)
        output = tf.concat([output_fw, output_bw], axis=-1)
        return output, state

    def lstm(self, seq, seq_len):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        output, state = tf.nn.dynamic_rnn(cell_fw, seq, sequence_length=seq_len, dtype=tf.float32)
        return output, state

    def activation(self, x):
        return tf.nn.relu(x)

    def siamese(self, we1, we2):
        with tf.variable_scope("bilstm") as scope:
            lstm1, state1 = self.bilstm(we1, self.len1)
            scope.reuse_variables()
            lstm2, state2 = self.bilstm(we2, self.len2)
            scope.reuse_variables()
        ### Max pooling
        lstm1_pool = max_pool(lstm1)
        lstm2_pool = max_pool(lstm2)

        ### Features
        flat1 = tf.contrib.layers.flatten(lstm1_pool)
        flat2 = tf.contrib.layers.flatten(lstm2_pool)
        mult = tf.multiply(flat1, flat2)
        diff = tf.abs(tf.subtract(flat1, flat2))

        concat = tf.concat([mult, diff], axis=-1)

        ### FC layers
        concat_size = int(concat.get_shape()[1])
        intermediary_size = 2 + (concat_size - 2) // 2
        # intermediary_size = 512

        with tf.variable_scope("fc1") as scope:
            W1 = tf.Variable(tf.random_normal([concat_size, intermediary_size], stddev=1e-3), name="w_fc")
            b1 = tf.Variable(tf.zeros([intermediary_size]), name="b_fc")

            z1 = tf.matmul(concat, W1) + b1

            epsilon = 1e-3
            batch_mean1, batch_var1 = tf.nn.moments(z1, [0])
            scale1, beta1 = tf.Variable(tf.ones([intermediary_size])), tf.Variable(tf.zeros([intermediary_size]))
            z1 = tf.nn.batch_normalization(z1, batch_mean1, batch_var1, beta1, scale1, epsilon)

            fc1 = tf.layers.dropout(self.activation(z1), rate=self.dropout,
                                    training=tf.convert_to_tensor(self.is_training))

        with tf.variable_scope("fc2") as scope:
            W2 = tf.Variable(tf.random_normal([intermediary_size, 2], stddev=1e-3), name="w_fc")
            b2 = tf.Variable(tf.zeros([2]), name="b_fc")

            z2 = tf.matmul(fc1, W2) + b2

            epsilon = 1e-3
            batch_mean2, batch_var2 = tf.nn.moments(z2, [0])
            scale2, beta2 = tf.Variable(tf.ones([2])), tf.Variable(tf.zeros([2]))
            z2 = tf.nn.batch_normalization(z2, batch_mean2, batch_var2, beta2, scale2, epsilon)

            self.fc2 = z2

    def set_session(self, sess):
        self.session = sess

    def run_dev_batch(self, train_handle):
        return self.session.run([self.accuracy, self.loss], feed_dict={'handle:0': train_handle,
                                                                       self.is_training: False,
                                                                       })

    def run_train_batch(self, train_handle):
        return self.session.run([self.loss_per_example_flat,
                                 self.train_step_inf,
                                 self.current_ids,
                                 self.accuracy], feed_dict={'handle:0': train_handle,
                                                            self.is_training: True})

    def run_score_batch(self, train_handle):
        return self.session.run([self.sent_1,
                                 self.sent_2,
                                 self.labels,
                                 self.current_ids,
                                 self.loss_per_example_flat,
                                 self.fc2],
                                feed_dict={'handle:0': train_handle,
                                           self.is_training: False})

    def evaluation_stats(self, train_handle, dev_handle, global_step, logger, log_saver):
        train_accuracy, train_cur_loss = self.run_dev_batch(train_handle)
        dev_accuracy, dev_cur_loss = self.run_dev_batch(dev_handle)
        logger.info(
            "Global step {} Train_acc: {:.2f}, Train loss {:.2f} Dev_acc: {:.3f} Dev_loss {:.3f}".format(
                global_step,
                float(train_accuracy),
                float(train_cur_loss),
                float(dev_accuracy),
                float(dev_cur_loss)
            ))

    def count_stats(self, train_handle):
        return self.session.run([
            self.fn_update,
            self.tn_update,
            self.tp_update,
            self.fp_update,
            self.loss
        ],
            feed_dict={'handle:0': train_handle,
                       self.is_training: False})

    def get_stats(self):
        return self.session.run([self.TP,
                                 self.FP,
                                 self.FN,
                                 self.TN])
