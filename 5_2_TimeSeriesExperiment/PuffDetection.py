import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os
import sys
import scipy

softness_parameter = 3

if len(sys.argv)>1:
    loss_function = ['SoftLoc', 'CE', 'Classical'][int(sys.argv[1])]
else:
    loss_function = ['SoftLoc', 'CE', 'Classical'][0]


print("-------------------")
print(loss_function)


if len(sys.argv)>2:
    user_out = [10, 11, 12, 14, 15, 16][int(sys.argv[2])]
    noise = float(sys.argv[3])
    noise_type = ["normal", "binary", "skewed"][int(sys.argv[4])]
else:
    user_out = 10
    noise = 0
    noise_type = ["normal", "binary", "skewed"][0]


# Loss placeholders
loss_window = {'soft': np.zeros([25]), 'count': np.zeros([25]), 'loss': np.zeros([25])}
list_loss = {'soft': [], 'count': [], 'loss': []}
list_f1, list_recall, list_precision = [], [], []


###### Dataset Creation Functions
def parseData():
    count = []
    length = 0
    user = []
    features = np.zeros([0, 39])
    predictions = np.zeros([0])
    for kk in [10, 11, 12, 14, 15, 16]:
        pred = np.genfromtxt("data/predictions_" + str(kk) + ".csv", delimiter=',', dtype=None, names=True)
        feat = np.genfromtxt("data/features_" + str(kk) + ".csv", delimiter=',', dtype=None, names=True)

        sequence = np.array([xx[3] for xx in pred])
        features_tmp = np.concatenate([np.array([elem for elem in xx])[np.newaxis, :] for xx in feat], axis=0)

        count.append(len(np.where(sequence[1:] - sequence[:-1] == 1)[0]))
        length += len(sequence)

        user += features_tmp.shape[0] * [kk]
        features = np.concatenate([features, features_tmp], axis=0)
        predictions = np.concatenate([predictions, sequence], axis=0)

    return features, predictions, np.array(user)


def createDataset(length=50, step=1, leave_one_out=10, noise_level=0, type="binary", loo_condition="session"):
    raw_features, raw_predictions, raw_user = parseData()

    idx_session = np.cumsum(np.concatenate([[0], np.diff(raw_features[:, 0]) > 5 * np.power(10, 6)]))

    # Generate full dataset
    final_feature = np.zeros([100000, length, raw_features.shape[1] - 2])
    final_labels = np.zeros([100000, length])
    final_labels_clean = np.zeros([100000, length])
    final_user = np.zeros([100000])
    final_session = np.zeros([100000])
    full_session_features = np.zeros([30, 700, raw_features.shape[1] - 2])
    full_session_labels = np.zeros([30, 700])
    session_user = []

    X_RoyAdam = []
    T_RoyAdam = []
    Z_RoyAdam = []
    Y_RoyAdam = []

    idx = 0
    for kk in np.unique(idx_session):
        subset = idx_session == kk
        # subset = raw_user == kk
        label_tmp = raw_predictions[subset]

        full_session_features[kk, :label_tmp.shape[0], :] = raw_features[subset][:, 2:]
        full_session_labels[kk, :label_tmp.shape[0]] = raw_predictions[subset]

        if type == "binary":
            noisy_idx = np.minimum(np.maximum(
                np.where(label_tmp)[0] + np.random.choice([-1, 1], size=len(np.where(label_tmp)[0])) * noise_level, 0),
                len(label_tmp) - 1)
        elif type == "normal":
            noisy_idx = np.minimum(np.maximum(
                np.round(np.where(label_tmp)[0] + np.random.normal(size=len(np.where(label_tmp)[0])) * noise_level), 0),
                len(label_tmp) - 1)

        elif type == "skewed":
            def randn_skew_fast(N=1, alpha=0.0, loc=0.0, scale=1.0):
                sigma = alpha / np.sqrt(1.0 + alpha**2)
                u0 = np.random.randn(N)
                v = np.random.randn(N)
                u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
                u1[u0 < 0] *= -1
                u1 = u1 + loc
                return u1

            noise = randn_skew_fast(alpha=-2, loc=0, scale=noise_level, N=len(np.where(label_tmp)[0]))
            noisy_idx = np.minimum(np.maximum(
                            np.round(np.where(label_tmp)[0] + noise), 0),
                            len(label_tmp) - 1)

        label_noisy = np.zeros(label_tmp.shape)

        if loss_function == "CE":
            label_noisy[noisy_idx.astype(np.int)] = 1
        else:
            label_noisy[noisy_idx.astype(np.int)] += 1

        if noise_level == 0:
            assert (np.sum(label_noisy - label_tmp) == 0)

        session_user.append(raw_user[subset][0])
        for tt in range(int(np.ceil((raw_features[subset].shape[0]) / step))):
            final_feature[idx, :raw_features[subset][step * tt:step * tt + length, 2:].shape[0], :] \
                = raw_features[subset][step * tt:step * tt + length, 2:]
            final_labels[idx, :raw_features[subset][step * tt:step * tt + length, 2:].shape[0]] \
                = label_noisy[step * tt:step * tt + length]
            final_labels_clean[idx, :raw_features[subset][step * tt:step * tt + length, 2:].shape[0]] \
                = label_tmp[step * tt:step * tt + length]
            final_user[idx] = raw_user[subset][step * tt]
            final_session[idx] = kk

            idx += 1

        X_RoyAdam.append(raw_features[subset, 2:])
        T_RoyAdam.append(raw_features[subset, 0])
        Z_RoyAdam.append(np.where(label_noisy))
        Y_RoyAdam.append(label_tmp)

    np.save("RoyAdams/code/X.npy", X_RoyAdam)
    np.save("RoyAdams/code/T.npy", T_RoyAdam)
    np.save("RoyAdams/code/Z.npy", Z_RoyAdam)
    np.save("RoyAdams/code/Y.npy", Y_RoyAdam)

    final_feature, final_labels, final_labels_clean, final_user, final_session \
        = final_feature[:idx], final_labels[:idx], final_labels_clean[:idx], final_user[:idx], final_session[:idx]
    if loo_condition == "session":
        session_out = np.random.choice(np.unique(idx_session), 3, False)
        loo_idx = np.array([xx in session_out for xx in final_session]).astype(np.bool)
        los_idx_full = np.array([xx in session_out for xx in np.unique(idx_session)]).astype(np.bool)
    else:
        loo_idx = final_user == leave_one_out
        los_idx_full = np.array(session_user) == leave_one_out


    return final_feature[np.logical_not(loo_idx)], \
           final_labels[np.logical_not(loo_idx)], \
           final_feature[loo_idx], \
           final_labels_clean[loo_idx], \
           full_session_features[los_idx_full], full_session_labels[los_idx_full]


###### Learning Pipeline
train_features, train_label, test_features, test_label, full_features, full_labels \
    = createDataset(leave_one_out=user_out, loo_condition="", noise_level=noise, type=noise_type)
if loss_function == "CE":
    train_label = np.concatenate([np.logical_not(train_label[:, :, np.newaxis]), train_label[:, :, np.newaxis]], axis=2)
else:
    train_label = train_label[:, :, np.newaxis]

# Define Simple Network
if loss_function == "CE":
    config = {'niter': 10000, 'batch_size': 32, 'num_units': 14, 'learning_rate': 0.001,'bool_gradient_clipping': False,'clipping_ratio': 5}
else:
    config = {'niter': 10000, 'batch_size': 32, 'num_units': 14, 'learning_rate': 0.001, 'bool_gradient_clipping': True,'clipping_ratio': 5}

if loss_function=='Classical':
    config['learning_rate'] = 0.0002

x = tf.placeholder("float", [None, None, train_features.shape[2]])
if loss_function == "CE":
    y = tf.placeholder("float", [None, None, 2])
else:
    y = tf.placeholder("float", [None, None, 1])
current_keep_prob = tf.placeholder("float", [])
alphaFactor = tf.placeholder("float", [])

x_fully = tf.contrib.layers.fully_connected(x, 14)
dropped = tf.nn.dropout(x_fully, keep_prob=current_keep_prob)
forward_layer = tf.contrib.rnn.LSTMBlockCell(config['num_units'])
output_rnn, _ = tf.nn.dynamic_rnn(forward_layer, dropped, dtype=tf.float32)

if loss_function == "CE":
    logit = tf.contrib.layers.fully_connected(output_rnn, 2, activation_fn=None)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logit))
    loss_soft, loss_count = tf.ones([]), tf.ones([])

elif loss_function=='Classical':

    # Gaussian filter
    smoothness = softness_parameter
    import scipy

    signal = np.zeros([50])  # 21
    signal[len(signal) // 2] = 1
    filt = scipy.ndimage.filters.gaussian_filter(signal, smoothness)

    gaussFilter = tf.expand_dims(tf.expand_dims(tf.constant(filt, tf.float32), 1), 1)


    def smooth(input):
        shape_in = tf.shape(input)
        input = tf.reshape(tf.transpose(input, [0, 2, 1]), [shape_in[0] * shape_in[2], shape_in[1], 1])
        input = tf.nn.conv1d(input, gaussFilter, stride=1, padding='SAME')
        input = tf.transpose(tf.reshape(input, [shape_in[0], shape_in[2], shape_in[1]]), [0, 2, 1])
        return input

    logit = tf.contrib.layers.fully_connected(output_rnn, 1, activation_fn=tf.nn.sigmoid)
    loss = tf.reduce_sum(tf.square(tf.subtract(smooth(y), logit)))
    loss_soft, loss_count = tf.ones([]), tf.ones([])

else:
    # SoftLoc
    def customLoss(yTrue, yPred, smoothness=1, alpha=0.2, SoftLoc=True):

        # Gaussian filter
        signal = np.zeros([50])  # 21
        signal[len(signal) // 2] = 1
        filt = scipy.ndimage.filters.gaussian_filter(signal, smoothness)

        gaussFilter = tf.expand_dims(tf.expand_dims(tf.constant(filt, tf.float32), 1), 1)

        def smooth(input):
            shape_in = tf.shape(input)
            input = tf.reshape(tf.transpose(input, [0, 2, 1]), [shape_in[0] * shape_in[2], shape_in[1], 1])
            input = tf.nn.conv1d(input, gaussFilter, stride=1, padding='SAME')
            input = tf.transpose(tf.reshape(input, [shape_in[0], shape_in[2], shape_in[1]]), [0, 2, 1])
            return input

        if SoftLoc:
            max_occurrence = 50

            shape_in = tf.shape(yPred)

            loss_smooth = tf.reduce_sum(tf.square(tf.subtract(smooth(yTrue), smooth(yPred))))

            # Counting Loss
            count_prediction = tf.one_hot(tf.zeros([shape_in[0], shape_in[2]], dtype=tf.int32),
                                          max_occurrence)
            #mass = tf.unstack(yPred, axis=1)
            partitions = tf.range(shape_in[1])
            num_partitions = 60
            mass = tf.dynamic_partition(tf.transpose(yPred, [1,0,2]), partitions, num_partitions, name='dynamic_unstack')

            for output in mass:
                def f1(): return tf.expand_dims(tf.reshape(output,[shape_in[0],shape_in[2]]), 2)
                def f2(): return tf.zeros([shape_in[0], shape_in[2],1])
                increment = tf.cond(tf.less(0,tf.shape(output)[0]),f1,f2)

                count_prediction = tf.multiply(tf.concat((tf.tile(1 - increment, [1, 1, max_occurrence - 1]),
                                                          tf.ones([shape_in[0], shape_in[2], 1])), axis=2),
                                               count_prediction) \
                                   + tf.multiply(tf.tile(increment, [1, 1, max_occurrence]),
                                                 tf.slice(
                                                     tf.concat(
                                                         (tf.zeros([shape_in[0], shape_in[2], 1]), count_prediction),
                                                         axis=2), [0, 0, 0],
                                                     [shape_in[0], shape_in[2], max_occurrence]))

            # Count regularizer Loss
            loss_count = tf.reduce_mean(
                -tf.reduce_sum(tf.one_hot(tf.cast(tf.reduce_sum(yTrue, axis=1), tf.int32), max_occurrence) * tf.log(
                    count_prediction + 1e-9), reduction_indices=[1]))  # does not wor
            custom_loss = (1 - alpha) * loss_smooth + alpha * loss_count
        else:
            custom_loss = tf.reduce_sum(tf.square(tf.subtract(smooth(yTrue), smooth(yPred))))

        return custom_loss, loss_smooth, loss_count

    logit = tf.contrib.layers.fully_connected(output_rnn, 1, activation_fn=tf.nn.sigmoid)
    loss, loss_soft, loss_count = customLoss(y, logit, smoothness=softness_parameter, alpha=alphaFactor)

# optimization
if config['bool_gradient_clipping']:
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config['learning_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate'])
    grads = tf.gradients(loss, tf.trainable_variables())
    norm = tf.global_norm(grads)
    grads, _ = tf.clip_by_global_norm(grads, config['clipping_ratio'])  # gradient clipping
    grads_and_vars = list(zip(grads, tf.trainable_variables()))
    opt = optimizer.apply_gradients(grads_and_vars)
else:
    opt = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(loss)

init = tf.global_variables_initializer()

iter = 0
with tf.Session() as session:
    session.run(init)
    while iter <= config['niter']:
        current_factor = 0.95

        idx_batch = np.random.randint(0, len(train_features), config['batch_size'])
        batch_features, batch_y = train_features[idx_batch, :, :], train_label[idx_batch, :, :]

        # Run optimization
        ll, ll_soft, ll_count, _ = session.run([loss, loss_soft, loss_count, opt],
                                               feed_dict={x: batch_features.astype(np.float),
                                                          y: batch_y.astype(np.float),
                                                          current_keep_prob: 0.5,
                                                          alphaFactor: current_factor})

        if iter % 2 == 0:
            # Save window loss
            loss_window['loss'][1:] = loss_window['loss'][:-1]
            loss_window['loss'][0] = ll
            list_loss['loss'].append(np.median(loss_window['loss'][loss_window['loss'] != 0]))

            loss_window['soft'][1:] = loss_window['soft'][:-1]
            loss_window['soft'][0] = ll_soft
            list_loss['soft'].append(np.median(loss_window['soft'][loss_window['soft'] != 0]))

            loss_window['count'][1:] = loss_window['count'][:-1]
            loss_window['count'][0] = ll_count
            list_loss['count'].append(np.median(loss_window['count'][loss_window['count'] != 0]))

        if iter % 500 == 0 and iter > 0:
            print(iter, ll, ll_soft, ll_count, "factor:", current_factor)

            predictions = session.run(logit, feed_dict={x: full_features.astype(np.float)[:, :, :],current_keep_prob: 1})

            if loss_function == "CE":
                list_f1.append(f1_score(full_labels.flatten(), np.argmax(predictions, axis=2).flatten()))
                list_precision.append(precision_score(full_labels.flatten(), np.argmax(predictions, axis=2).flatten()))
                list_recall.append(recall_score(full_labels.flatten(), np.argmax(predictions, axis=2).flatten()))

            elif loss_function == 'Classical':

                # Peak-picking for one-sided smoothing
                predictions /= np.max(filt)

                final_prediction = np.zeros([predictions.shape[0], predictions.shape[1]])
                for idx in range(predictions.shape[0]):
                    while np.max(predictions[idx, :, 0] - scipy.ndimage.filters.gaussian_filter(final_prediction[idx, :],
                                                 softness_parameter) / np.max(filt)) >= 0.3:

                        series = predictions[idx, :, 0] - scipy.ndimage.filters.gaussian_filter(
                            final_prediction[idx, :], softness_parameter) / np.max(filt)
                        xx_max = np.argmax(series)
                        if final_prediction[idx, xx_max]==1:
                        # Already processed
                            predictions[idx, xx_max, 0]=0                      
                        else:
                            n_points = max(int(np.round(series[xx_max]*1.2)),1)

                            range_min = xx_max
                            range_max = xx_max

                            for pp in range(n_points - 1):
                                if series[range_min - 1] > series[range_max + 1]:
                                    range_min -= 1
                                else:
                                    range_max += 1

                            final_prediction[idx, range_min:range_max + 1] = 1

                list_f1.append(f1_score(full_labels.flatten(), final_prediction.flatten()))
                list_precision.append(precision_score(full_labels.flatten(), final_prediction.flatten()))
                list_recall.append(recall_score(full_labels.flatten(), final_prediction.flatten()))
            else:
                threshold = 0.2
                list_f1.append(f1_score(full_labels.flatten(), predictions.flatten() > threshold))
                list_precision.append(precision_score(full_labels.flatten(), predictions.flatten() > threshold))
                list_recall.append(recall_score(full_labels.flatten(), predictions.flatten() > threshold))

        iter += 1


with open('results.txt', 'a') as f:
    f.write("%s," % str(loss_function))
    f.write("%s," % str(noise_type))
    f.write("%s," % str(noise))
    f.write("%s," % str(user_out))
    f.write("%s," % str(list_f1[-1]))
    f.write("%s," % str(list_precision[-1]))
    f.write("%s," % str(list_recall[-1]))
    f.write("\n")

