import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import os
import scipy
import tensorflow as tf
from tensorflow.contrib import rnn
from multiprocessing import Pool

import Display.localizationPlot as lp
from Math.NadarayaWatson import *


class SoftNetwork:

    def __init__(self, configuration):
        print('>> Construct Graph...')
        os.environ["CUDA_VISIBLE_DEVICES"] = configuration['GPU_Device']

        self.config = configuration

        # Loss placeholders
        self.loss_window = {'direct': np.zeros([25]), 'indirect': np.zeros([25]), 'loss': np.zeros([25])}
        self.list_loss = {'direct': [], 'indirect': [], 'loss': []}

        # Iteration count
        self.iter = 0

        if self.config['weight_indirect'] is None:
            self.current_balancing = 0
        else:
            self.current_balancing = self.config['weight_indirect']

        # Gaussian filter
        signal = np.zeros([151])  # 21
        signal[len(signal) // 2] = 1
        filt = scipy.ndimage.filters.gaussian_filter(signal, self.config['smoothing_width'])

        # Output layer weights and biases
        hidden_weights = tf.Variable(tf.random_normal([self.config['num_units'], self.config['hidden_size']]))
        hidden_bias = tf.Variable(tf.zeros([self.config['hidden_size']]))
        out_weights = tf.Variable(tf.random_normal([self.config['hidden_size'], self.config['n_channel']]))
        out_bias = tf.Variable(
            np.log((1 / np.power(0.8, 2 / self.config['time_steps'])) - 1) * tf.ones(
                [self.config['n_channel']]))  # P(x==0) = 0.8

        # Placeholder
        self.x = tf.placeholder("float", [None, self.config['time_steps'], self.config['n_filter']])
        self.y = tf.placeholder("float", [None, self.config['n_channel'], self.config['max_occurence']])
        self.y_series = tf.placeholder("float", [None, self.config['n_channel'], self.config['time_steps']])
        self.balancing = tf.placeholder("float", ())

        gaussFilter = tf.expand_dims(tf.expand_dims(tf.constant(filt, tf.float32), 1), 1)

        # Downsampling video
        x_conv = tf.expand_dims(self.x, dim=3)  # l x 144
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][0], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 72
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][1], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 36
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][2], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 18
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][3], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 2], strides=[1, 2])  # l x 9
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][4], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 3], strides=[1, 3])  # l x 3
        x_conv = tf.contrib.layers.conv2d(x_conv, self.config['n_filters'][5], kernel_size=[3, 4])
        x_conv = tf.layers.max_pooling2d(x_conv, pool_size=[1, 3], strides=[1, 3])  # l x 1

        # processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
        x_conv = tf.reshape(x_conv, [self.config['batch_size'], self.config['time_steps'], self.config['n_filters'][5]])

        def length(sequence):
            used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
            length = tf.reduce_sum(used, 1)
            length = tf.cast(length, tf.int32)
            return length

        # Reccuring unit
        length_sequence = length(self.x) - 2
        lstm_layer = rnn.BasicLSTMCell(self.config['num_units'], forget_bias=1)
        output_rnn, _ = tf.nn.dynamic_rnn(lstm_layer, x_conv, dtype=tf.float32, sequence_length=length_sequence)

        def last_relevant(output, length):
            batch_size = tf.shape(output)[0]
            max_length = tf.shape(output)[1]
            out_size = int(output.get_shape()[2])
            index = tf.range(0, batch_size) * max_length + (length - 1)
            flat = tf.reshape(output, [-1, out_size])
            relevant = tf.gather(flat, index)
            return relevant

        prediction = tf.one_hot(tf.zeros(self.config['batch_size'] * self.config['n_channel'], dtype=tf.int32),
                                self.config['max_occurence'])  # initial prediction
        output_rnn_stack = tf.unstack(output_rnn, self.config['time_steps'], 1)

        outputs = []
        pp_list = []
        for output in output_rnn_stack:
            hidden_layer = tf.sigmoid(tf.matmul(output, hidden_weights) + hidden_bias)
            increment = tf.sigmoid(tf.matmul(hidden_layer, out_weights) + out_bias)
            increment = tf.reshape(increment, [self.config['batch_size'] * self.config['n_channel'], 1])

            # todo: use a convolution instead of this mess with [1-alpha, alpha] as filter
            # stayed + moved
            prediction = tf.multiply(
                tf.concat((tf.tile(1 - increment, [1, self.config['max_occurence'] - 1]),
                           tf.ones([self.config['batch_size'] * self.config['n_channel'], 1])), axis=1),
                prediction) + \
                         tf.multiply(tf.tile(increment, [1, self.config['max_occurence']]),
                                     tf.slice(tf.concat(
                                         (tf.zeros([self.config['batch_size'] * self.config['n_channel'], 1]),
                                          prediction),
                                         axis=1), [0, 0],
                                         [self.config['batch_size'] * self.config['n_channel'],
                                          self.config['max_occurence']]))

            outputs.append(prediction)
            pp_list.append(increment)

        # Indirect Loss
        prediction = last_relevant(tf.stack(outputs, 1),
                                   tf.reshape(
                                       tf.tile(tf.expand_dims(length_sequence, 1), [1, self.config['n_channel']]),
                                       [self.config['batch_size'] * self.config['n_channel']]))
        y_reshaped = tf.identity(self.y)
        y_reshaped = tf.reshape(y_reshaped,
                                [self.config['batch_size'] * self.config['n_channel'], self.config['max_occurence']])
        self.loss_indirect = tf.reduce_mean(
            -tf.reduce_sum(y_reshaped * tf.log(prediction + 1e-9), reduction_indices=[1]))  # does not work

        # Direct Loss
        if self.config['Direct']:
            self.filtered = tf.nn.conv1d(tf.stack(pp_list, 1), gaussFilter, stride=1, padding='SAME')
            self.filtered_y = tf.nn.conv1d(
                tf.expand_dims(
                    tf.reshape(self.y_series,
                               [self.config['batch_size'] * self.config['n_channel'], self.config['time_steps']]),
                    axis=2),
                gaussFilter, stride=1, padding='SAME')
            self.loss_direct = tf.reduce_sum(tf.square(tf.subtract(self.filtered_y, self.filtered))) / self.config[
                'n_channel']

            # Aggregate
            self.factor = tf.placeholder(tf.float32, shape=())
            self.loss = (1-self.factor) * self.loss_direct + self.factor * self.loss_indirect
        else:
            self.loss = self.loss_indirect
            self.loss_direct = tf.ones([])

        # optimization
        if self.config['bool_gradient_clipping']:
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate'])
            grads = tf.gradients(self.loss, tf.trainable_variables())
            self.norm = tf.global_norm(grads)
            grads, _ = tf.clip_by_global_norm(grads, self.config['clipping_ratio'])  # gradient clipping
            grads_and_vars = list(zip(grads, tf.trainable_variables()))
            self.opt = optimizer.apply_gradients(grads_and_vars)
        else:
            self.opt = tf.train.AdamOptimizer(learning_rate=self.config['learning_rate']).minimize(self.loss)

        # model evaluation
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_reshaped, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.stacked_prediction = tf.stack(outputs, axis=1)

        # Model saver
        self.saver = tf.train.Saver()

        # initialize variables
        self.init = tf.global_variables_initializer()

    def reset(self):
        tf.reset_default_graph()

    # One step optimization
    def optimize(self, session, batch_x, batch_y, batch_y_series):

        self.current_factor = max(min((self.iter - self.config['start_processing']) / 100000, 0.9),self.config['weight_indirect'])

        # Run optimization
        ll, ll_dir, ll_ind, _ = session.run([self.loss, self.loss_direct, self.loss_indirect, self.opt],
                                            feed_dict={self.x: batch_x.astype(np.float),
                                                       self.y: batch_y.astype(np.float),
                                                       self.y_series: batch_y_series.astype(np.float),
                                                       self.balancing: self.current_balancing,
                                                       self.factor: self.current_factor})

        self.iter += 1

        # update balancing
        if self.config['weight_indirect'] is None:
            if self.iter > 3000 and self.current_balancing < 1:
                self.current_balancing += 0.0001

        if self.iter % 10 == 0:
            # Save window loss
            self.loss_window['loss'][1:] = self.loss_window['loss'][:-1]
            self.loss_window['loss'][0] = ll
            self.list_loss['loss'].append(np.median(self.loss_window['loss'][self.loss_window['loss'] != 0]))

            if self.config['Direct']:
                self.loss_window['direct'][1:] = self.loss_window['direct'][:-1]
                self.loss_window['direct'][0] = ll_dir
                self.list_loss['direct'].append(np.median(self.loss_window['direct'][self.loss_window['direct'] != 0]))

            self.loss_window['indirect'][1:] = self.loss_window['indirect'][:-1]
            self.loss_window['indirect'][0] = ll_ind
            self.list_loss['indirect'].append(
                np.median(self.loss_window['indirect'][self.loss_window['indirect'] != 0]))

        if self.iter % self.config['save_frequency'] == 0:
            self.save(session)

        if self.iter % self.config['save_frequency'] == self.config['save_frequency'] // 2:
            self.save(session, version="_mid")


    # initialize variables
    def initialize(self, session):
        session.run(self.init)

    def save(self, session, version=""):
        self.saver.save(session, "models/model" + self.config['extension'] + version + ".ckpt")
        print("Model saved")

    def restore(self, session):
        self.saver.restore(session, "models/model" + self.config['extension'] + ".ckpt")
        # print("Model Restored!")

    def predict(self, session, xx):
        pp = np.zeros([xx.shape[0], self.config['n_channel'], self.config['time_steps']])

        for ii in range(xx.shape[0] // self.config['batch_size']):
            pred_ts = session.run(self.stacked_prediction, feed_dict={
                self.x: xx[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size']]})
            predictions = np.array([pred_ts])

            pp[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size'], :, :-1] = (
                    1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config[
                'trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel'], self.config['time_steps'] - 1])
            pp[ii * self.config['batch_size']:(ii + 1) * self.config['batch_size'], :, 0] = (
                    1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel']])

        if xx.shape[0] % self.config['batch_size'] > 0:
            pred_ts = session.run(self.stacked_prediction, feed_dict={self.x: xx[-self.config['batch_size']:]})
            predictions = np.array([pred_ts])

            pp[-self.config['batch_size']:, :, :-1] = (
                    1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config[
                'trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel'], self.config['time_steps'] - 1])
            pp[-self.config['batch_size']:, :, 0] = (
                    1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
                [self.config['batch_size'], self.config['n_channel']])

        return pp

    def evaluate(self, session, x, y):
        pp = self.predict(session, x)

        fig, stats = lp.localizationPlot(
            pp, y, n_samples=20, dist_threshold=self.config['tolerence'], factor=1, bias=self.config['temporal_bias'])

        return fig, stats

    def infer(self, session, batch_x, batch_y, batch_y_series):
        a, b, c, d, pred_ts = session.run(
            [self.accuracy, self.loss, self.loss_direct, self.loss_indirect, self.stacked_prediction],
            feed_dict={self.x: batch_x, self.y: batch_y, self.y_series: batch_y_series.astype(np.float),
                       self.balancing: self.current_balancing, self.factor: self.current_factor})

        predictions = np.array([pred_ts])
        pp = np.zeros([batch_y.shape[0], batch_y_series.shape[1], batch_y_series.shape[2]])
        pp[:, :, :-1] = (
                1 - predictions[:, :, 1:, 0] / predictions[:, :, :-1, 0] > self.config['trigger_threshold']).reshape(
            [batch_y.shape[0], batch_y_series.shape[1], batch_y_series.shape[2] - 1])
        pp[:, :, 0] = (1 - predictions[:, :, 0, 0] > self.config['trigger_threshold']).reshape(
            [batch_y.shape[0], batch_y_series.shape[1]])

        return a, b, c, d, pp

    def smoothPlot(self, session, batch_x, batch_y, batch_y_series):

        smoothed_pred, smoothed_y = session.run([self.filtered, self.filtered_y],
                                                feed_dict={self.x: batch_x, self.y: batch_y,
                                                           self.y_series: batch_y_series.astype(np.float)})

        smoothed_pred = smoothed_pred.reshape(
            [self.config['batch_size'], self.config['n_channel'], self.config['time_steps']])
        smoothed_y = smoothed_y.reshape(
            [self.config['batch_size'], self.config['n_channel'], self.config['time_steps']])

        height = np.max(smoothed_y[:12, :, :]) * 1.2
        plt.figure(figsize=(15, 10))
        for ii in range(0, 12):
            plt.subplot(4, 3, ii + 1)
            for kk in range(self.config['n_channel']):
                plt.plot(smoothed_pred[ii, kk, :] + height * kk, 'b', linewidth=5)
                plt.plot(smoothed_y[ii, kk, :] + height * kk, 'r')
            plt.ylim([-0.01, 3 * height])
        plt.savefig('plt/smoothed_' + self.config['extension'])

    def performancePlot(self, stats_history):

        f = plt.figure(figsize=(15, 7))
        ax = plt.subplot(1, 4, 1)
        if self.config['Direct']:
            plt.plot(10 * np.arange(1, 1 + len(self.list_loss['direct'])),
                     np.log(np.array(self.list_loss['direct'])), label='Direct')

        plt.plot(10 * np.arange(1, 1 + len(self.list_loss['indirect'])),
                 np.log(np.array(self.list_loss['indirect'])), label='Indirect')

        plt.plot(10 * np.arange(1, 1 + len(self.list_loss['loss'])),
                 np.log(np.array(self.list_loss['loss'])), label='Loss')

        plt.plot(10 * np.arange(1, 1 + len(self.list_loss['loss'])),
                 GaussKernel(np.arange(1, 1 + len(self.list_loss['loss'])),
                             np.arange(1, 1 + len(self.list_loss['loss'])),
                             np.log(np.array(self.list_loss['loss'])), self.config['show_frequency']))

        plt.legend()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.grid(True)

        if self.config['Direct']:
            plt.ylim([min(np.min(np.log(np.array(self.list_loss['direct']))),
                          np.min(np.log(np.array(self.list_loss['indirect'])))) - 0.02, 0 + self.config['Direct']])
        else:
            plt.ylim([np.min(np.log(np.array(self.list_loss['indirect']))) - 0.02, 0 + self.config['Direct']])

        # Train set subplot
        ax = plt.subplot(1, 4, 2)
        standardPerformanceSubplot(ax, self.config['show_frequency'], stats_history['f1'],
                                   stats_history['precision'], stats_history['recall'])

        # Validation set subplot
        ax = plt.subplot(1, 4, 3)
        standardPerformanceSubplot(ax, self.config['show_frequency'], stats_history['f1_val'],
                                   stats_history['precision_val'], stats_history['recall_val'])

        # Test set subplot
        ax = plt.subplot(1, 4, 4)
        standardPerformanceSubplot(ax, self.config['show_frequency'], stats_history['f1_out'],
                                   stats_history['precision_out'], stats_history['recall_out'])

        if stats_history['f1_ens'] is not []:
            plt.plot(self.config['show_frequency'] * np.arange(1, 1 + len(stats_history['f1_ens'])),
                     GaussKernel(np.arange(1, 1 + len(stats_history['f1_ens'])),
                                 np.arange(1, 1 + len(stats_history['f1_ens'])),
                                 np.array(stats_history['f1_ens']), 5), 'r')

        if stats_history['f1_ens'] is not []:
            plt.text(self.config['show_frequency'] * len(stats_history['f1_ens']),
                     GaussKernel(np.arange(1, 1 + len(stats_history['f1_ens'])),
                                 np.arange(1, 1 + len(stats_history['f1_ens'])),
                                 np.array(stats_history['f1_ens']), 5)[-1],
                     str(np.round(100 * GaussKernel(np.arange(1, 1 + len(stats_history['f1_ens'])),
                                                    np.arange(1, 1 + len(stats_history['f1_ens'])),
                                                    np.array(stats_history['f1_ens']), 5)[-1], 1)) + '%', color='r')

        plt.savefig('plt/loss_history_' + self.config['extension'])
        plt.close()


    def FastEnsembling(self, pp, factor, mass_threshold=0.2, suppression_field=14):
        """
        pp: [fold, samples, channel, time_steps]
        :param pp:
        :return:
        """
        p = Pool(8)
        simulated_fast = p.map(singleEnsemble,
                               [(pp, factor, x, mass_threshold, suppression_field) for x in range(pp.shape[1])])
        p.close()

        pp_ensembling = np.concatenate([x[0][np.newaxis, :, :] for x in simulated_fast], axis=0)
        pp_idx = np.array([x[1] for x in simulated_fast])

        return pp_ensembling[pp_idx]


def standardPerformanceSubplot(ax, frequency, f1, precision, recall):
    plt.plot(frequency * np.arange(1, 1 + len(f1)), f1)
    plt.plot(frequency * np.arange(1, 1 + len(precision)), precision)
    plt.plot(frequency * np.arange(1, 1 + len(recall)), recall)
    plt.plot(frequency * np.arange(1, 1 + len(f1)),
             GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5), 'k')

    plt.text(frequency * len(f1), f1[-1], str(np.round(100 * f1[-1], 1)) + '%')

    plt.text(frequency * len(f1),
             GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5)[-1],
             str(np.round(100 * GaussKernel(np.arange(1, 1 + len(f1)), np.arange(1, 1 + len(f1)), np.array(f1), 5)[-1],
                          1)) + '%')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.grid(True)
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.ylim([0, 1])


def singleEnsemble(input):
    pp = input[0]
    factor = input[1]
    sample_idx = input[2]
    mass_threshold = input[3]
    suppression_field = input[4]

    pp_ensembling = np.zeros([pp.shape[2], pp.shape[3]])
    for note_idx in range(pp.shape[2]):

        detected_events = []
        flag = 0
        iteration = 0
        while flag == 0 and iteration < 1000:

            # plt.figure()
            full = 0
            for kk in range(pp.shape[0]):
                series = pp[kk, sample_idx, note_idx, :]
                series_conv = GaussKernel(np.arange(pp.shape[3]),
                                          np.arange(pp.shape[3]) * factor[sample_idx * pp.shape[0] + kk],
                                          series, 5)
                full += series_conv

            if np.max(full) > mass_threshold:
                idx_max = np.where(full == np.max(full))[0][0]

                # delete max mass around that point
                for kk in range(pp.shape[0]):
                    local_idx_max = int(idx_max / factor[sample_idx * pp.shape[0] + kk])

                    pp[kk, sample_idx, note_idx,
                    max(local_idx_max - suppression_field, 0):local_idx_max + suppression_field] = 0

                detected_events.append(idx_max)
                pp_ensembling[note_idx, idx_max] += 1
            else:
                flag = 1

            iteration += 1

    return pp_ensembling, sample_idx
