"""
CRNN with Feature Enhancement (CRNN-FE)

Pipeline: CNN → BiLSTM → Feature Enhancement (HRNN + Attention) → CTC

Step 3: CNN feature extraction
Step 4: BiLSTM sequence learning
Step 5: HRNN + Attention for feature-level enhancement (stroke, alignment, spacing)
Step 6: CTC decoding

Uses tf.nn.ctc_loss (no warpctc) for better portability.
"""

from base.base_model import BaseModel
from utils.tf_compat import tf, dense_to_sparse
import tensorflow_addons as tfa
from .hrnn_attention import feature_enhancement_block


class Model(BaseModel):
    def __init__(self, data_loader, config):
        super(Model, self).__init__(config)
        self.rnn_num_hidden = getattr(config, 'rnn_num_hidden', 256)
        self.rnn_num_layers = getattr(config, 'rnn_num_layers', 5)
        self.rnn_dropout = getattr(config, 'rnn_dropout', 0.5)
        self.conv_patch_sizes = [3] * 5
        self.conv_depths = [16, 32, 48, 64, 80]
        self.conv_dropouts = [0, 0, 0.2, 0.2, 0.2]
        self.linear_dropout = getattr(config, 'linear_dropout', 0.5)
        self.reduce_factor = 8

        # Feature Enhancement (HRNN + Attention)
        self.hrnn_units = getattr(config, 'hrnn_units', 256)
        self.attention_units = getattr(config, 'attention_units', 128)
        self.attention_dropout = getattr(config, 'attention_dropout', 0.1)
        self.hrnn_dropout = getattr(config, 'hrnn_dropout', 0.2)

        self.data_loader = data_loader
        self.x, self.y, self.length, self.lab_length = None, None, None, None
        self.is_training = None
        self.prediction = None
        self.loss = None
        self.ler = None
        self.optimizer = None
        self.train_step = None

        self.build_model()
        self.init_saver()

    @staticmethod
    def calc_cer(predicted, targets):
        return tf.edit_distance(tf.cast(predicted, tf.int32), targets, normalize=True)

    def _cnn_blocks(self, x, batch_size, initializer):
        with tf.name_scope('CNN_Block_1'):
            c = tf.layers.dropout(x, self.conv_dropouts[0],
                                  noise_shape=tf.concat([tf.reshape(batch_size, [-1]), [1, 1, 1]], 0),
                                  training=self.is_training)
            c = tf.layers.conv2d(c, self.conv_depths[0], self.conv_patch_sizes[0], padding='same',
                                 activation=None, kernel_initializer=initializer)
            c = tf.layers.batch_normalization(c)
            c = tf.nn.leaky_relu(c)
            c = tf.layers.max_pooling2d(c, 2, 2, padding='same')

        with tf.name_scope('CNN_Block_2'):
            c = tf.layers.dropout(c, self.conv_dropouts[1],
                                  noise_shape=tf.concat([tf.reshape(batch_size, [-1]), [1, 1, self.conv_depths[0]]], 0),
                                  training=self.is_training)
            c = tf.layers.conv2d(c, self.conv_depths[1], self.conv_patch_sizes[1], padding='same',
                                 activation=None, kernel_initializer=initializer)
            c = tf.layers.batch_normalization(c)
            c = tf.nn.leaky_relu(c)
            c = tf.layers.max_pooling2d(c, 2, 2, padding='same')

        with tf.name_scope('CNN_Block_3'):
            c = tf.layers.dropout(c, self.conv_dropouts[2],
                                  noise_shape=tf.concat([tf.reshape(batch_size, [-1]), [1, 1, self.conv_depths[1]]], 0),
                                  training=self.is_training)
            c = tf.layers.conv2d(c, self.conv_depths[2], self.conv_patch_sizes[2], padding='same',
                                 activation=None, kernel_initializer=initializer)
            c = tf.layers.batch_normalization(c)
            c = tf.nn.leaky_relu(c)
            c = tf.layers.max_pooling2d(c, 2, 2, padding='same')

        with tf.name_scope('CNN_Block_4'):
            c = tf.layers.dropout(c, self.conv_dropouts[3],
                                  noise_shape=tf.concat([tf.reshape(batch_size, [-1]), [1, 1, self.conv_depths[2]]], 0),
                                  training=self.is_training)
            c = tf.layers.conv2d(c, self.conv_depths[3], self.conv_patch_sizes[3], padding='same',
                                 activation=None, kernel_initializer=initializer)
            c = tf.layers.batch_normalization(c)
            c = tf.nn.leaky_relu(c)

        with tf.name_scope('CNN_Block_5'):
            c = tf.layers.dropout(c, self.conv_dropouts[4],
                                  noise_shape=tf.concat([tf.reshape(batch_size, [-1]), [1, 1, self.conv_depths[3]]], 0),
                                  training=self.is_training)
            c = tf.layers.conv2d(c, self.conv_depths[4], self.conv_patch_sizes[4], padding='same',
                                 activation=None, kernel_initializer=initializer)
            c = tf.layers.batch_normalization(c)
            c = tf.nn.leaky_relu(c)
        return c

    def build_model(self):
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)

        with tf.variable_scope('inputs'):
            self.x, y, self.length, self.lab_length = self.data_loader.get_input()
            self.y = dense_to_sparse(y, eos_token=-1)
            self.x = tf.expand_dims(self.x, 3)
            x_shift = (tf.shape(self.x)[2] - self.length) / tf.constant(2)
            y_shift = tf.zeros_like(x_shift)
            translation_vector = tf.cast(tf.stack([x_shift, y_shift], axis=1), tf.float32)
            self.x = tfa.image.translate(self.x, translation_vector)
            self.length = tf.cast(tf.math.ceil(tf.math.divide(self.length, tf.constant(self.reduce_factor))), tf.int32)
            batch_size = tf.shape(self.x)[0]
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.length)
        tf.add_to_collection('inputs', self.lab_length)
        tf.add_to_collection('inputs', y)
        tf.add_to_collection('inputs', self.is_training)

        initializer = tf.glorot_uniform_initializer()

        # CNN
        cnn_out = self._cnn_blocks(self.x, batch_size, initializer)

        # (H,W,B,C) -> (W,B,H*C)
        output = tf.transpose(cnn_out, [2, 0, 1, 3])
        output = tf.reshape(output, [-1, batch_size, (self.config.im_height // self.reduce_factor) * self.conv_depths[4]])
        self.length = tf.tile(tf.expand_dims(tf.shape(output)[0], axis=0), [batch_size])

        # BiLSTM (replaces tf.contrib.cudnn_rnn.CudnnLSTM, removed in TF2)
        with tf.variable_scope('MultiRNN', reuse=tf.AUTO_REUSE):
            for i in range(self.rnn_num_layers):
                output = tf.layers.dropout(output, self.rnn_dropout, training=self.is_training)
                fw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_num_hidden)
                bw_cell = tf.nn.rnn_cell.LSTMCell(self.rnn_num_hidden)
                outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, output, time_major=True, dtype=tf.float32
                )
                output = tf.concat(outputs, 2)

        # Feature Enhancement: HRNN + Attention
        output = feature_enhancement_block(
            output,
            seq_length=self.length,
            num_units=self.hrnn_units,
            is_training=self.is_training,
            attention_units=self.attention_units,
            attention_dropout=self.attention_dropout,
            hrnn_dropout=self.hrnn_dropout,
            scope='feature_enhancement',
        )
        # output: (T, B, hrnn_units)

        # Dense: (T, B, hrnn_units) -> (T, B, num_classes)
        with tf.variable_scope('Dense'):
            output = tf.layers.dropout(output, self.linear_dropout, training=self.is_training)
            out_W = tf.get_variable('out_W', [self.hrnn_units, self.data_loader.num_classes],
                                    initializer=tf.glorot_uniform_initializer())
            out_b = tf.get_variable('out_b', [self.data_loader.num_classes],
                                    initializer=tf.zeros_initializer())
            output = tf.reshape(output, [-1, self.hrnn_units])
            logits = tf.matmul(output, out_W) + out_b
            self.logits = tf.reshape(logits, [-1, batch_size, self.data_loader.num_classes])

        # CTC loss (built-in, no warpctc)
        with tf.variable_scope('loss-acc'):
            self.loss = tf.nn.ctc_loss(
                labels=self.y,
                inputs=self.logits,
                sequence_length=self.length,
                time_major=True,
                ignore_longer_outputs_than_inputs=True,
            )
            self.cost = tf.reduce_mean(self.loss)
            self.prediction = tf.nn.ctc_beam_search_decoder(
                self.logits, sequence_length=self.length, merge_repeated=False
            )
            self.cer = self.calc_cer(self.prediction[0][0], self.y)

        with tf.variable_scope('train_step'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.RMSPropOptimizer(
                    learning_rate=self.config.learning_rate,
                    decay=self.config.learning_rate_decay,
                ).minimize(self.cost, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.cost)
        tf.add_to_collection('train', self.cer)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep_cpt, save_relative_paths=True)
        self.best_saver = tf.train.Saver(max_to_keep=self.config.max_to_keep_best, save_relative_paths=True)
