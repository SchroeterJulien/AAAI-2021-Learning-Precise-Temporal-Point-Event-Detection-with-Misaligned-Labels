import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from multi_config import load_configurations
import Display.localizationPlot as lp
from new_createDataset_v2 import *
from SoftNetworkModel import SoftNetwork

# Load configurations
config = load_configurations()
os.environ["CUDA_VISIBLE_DEVICES"] = config['GPU_Device']
print("---", config['extension'], "---")

if not os.path.exists('plt'):
    os.mkdir('plt')

if not os.path.exists('models'):
    os.mkdir('models')

# Open tensorflow session
with tf.Session() as sess:

    # Initialize model
    softModel = SoftNetwork(config)
    softModel.initialize(sess)
    #softModel.restore(sess)

    print('>> Load Dataset...')
    train_files, validation_file , test_files = loadData(config['extension'], config['seed'], "random", "Real")

    print(len(train_files), len(validation_file), len(test_files))

    # Train set
    print(">> create train set")
    X_data, Y_label, Y_data_raw, _, _ = generateSplitDataset(train_files, config, labelNoise = True)

    # Test set
    print(">> create test set")
    x_out, y_out_label, y_out_raw, stretch_factor_out, _ = generateSplitDataset(test_files, config, labelNoise = False)

    # Validation set
    print(">> create validation set")
    x_val, y_val_label, y_val_raw, _ , _ = generateSplitDataset(validation_file, config, labelNoise = False)

    print(Y_label.shape, x_val.shape, x_out.shape)

    # Training
    print('>>> Training:')
    stats_history = {'f1': [], 'precision': [], 'recall': [],
                     'f1_out': [], 'precision_out': [], 'recall_out': [],
                     'f1_val': [], 'precision_val': [], 'recall_val': [],
                     'f1_ens': [], 'precision_ens': [], 'recall_ens': []}

    iter = 1
    while iter <= config['niter']:
        idx_batch = np.random.randint(0, len(Y_label), config['batch_size'])
        batch_x, batch_y, batch_y_series = \
            X_data[idx_batch, :, :], Y_label[idx_batch, :, :], Y_data_raw[idx_batch, :, :]

        # Run optimization
        softModel.optimize(sess, batch_x, batch_y, batch_y_series)


        if iter % config['show_frequency'] == 0:
            acc, los, loss_dir, loss_ind, pp = softModel.infer(sess, batch_x, batch_y, batch_y_series)

            print("For iter ", iter)
            print("Accuracy ", acc)
            if config['Direct']:
                print("Loss ", np.round(los, 3), np.round(loss_dir, 3), np.round(loss_ind, 3))
            else:
                print("Loss ", np.round(los, 3))
            print("__________________")

            # Display (train) localization
            fig, stats = lp.localizationPlot(
                pp,
                batch_y_series, n_samples=20, dist_threshold=config['tolerence'], factor=1,
                bias=config['temporal_bias'])
            plt.savefig('plt/localization_in_' + config['extension'])
            plt.close()

            stats_history['f1'].append(stats['f1'])
            stats_history['precision'].append(stats['precision'])
            stats_history['recall'].append(stats['recall'])

            # Display (validation) localization
            pp = softModel.predict(sess, x_val)
            fig, stats_out = lp.localizationPlot(pp, y_val_raw, n_samples=20, dist_threshold=config['tolerence'],
                                                 factor=1, bias=config['temporal_bias'])
            plt.savefig('plt/localization_out_' + config['extension'])
            plt.close()

            stats_history['f1_val'].append(stats_out['f1'])
            stats_history['precision_val'].append(stats_out['precision'])
            stats_history['recall_val'].append(stats_out['recall'])


            # Display (test) localization
            pp = softModel.predict(sess, x_out)
            fig, stats_out = lp.localizationPlot(pp, y_out_raw, n_samples=20, dist_threshold=config['tolerence'],
                                                 factor=1, bias=config['temporal_bias'])
            plt.savefig('plt/localization_out_' + config['extension'])
            plt.close()

            stats_history['f1_out'].append(stats_out['f1'])
            stats_history['precision_out'].append(stats_out['precision'])
            stats_history['recall_out'].append(stats_out['recall'])

            # pp_trans = pp.reshape([5, pp.shape[0] // 5, pp.shape[1], pp.shape[2]])
            pp_trans = np.transpose(pp.reshape([pp.shape[0] // config['augmentation_factor'], config['augmentation_factor'], pp.shape[1], pp.shape[2]]), [1, 0, 2, 3])
            pp_ensemble = softModel.FastEnsembling(pp_trans, stretch_factor_out, config['ensembling_factor'])
            fig, stats_ensemble = lp.localizationPlot(pp_ensemble, y_out_raw[::config['augmentation_factor'], :, :], n_samples=20,
                                                      dist_threshold=config['tolerence'], factor=1,
                                                      bias=config['temporal_bias'])
            plt.savefig('plt/localization_ens_' + config['extension'])
            plt.close()

            stats_history['f1_ens'].append(stats_ensemble['f1'])
            stats_history['precision_ens'].append(stats_ensemble['precision'])
            stats_history['recall_ens'].append(stats_ensemble['recall'])

            # Display Loss & Performance
            softModel.performancePlot(stats_history)

            # Smoothed plot
            print("__________________")

            if config['Direct']:
                softModel.smoothPlot(sess, batch_x, batch_y, batch_y_series)

        iter += 1

from addSweep import *
addSweep(config['extension'])S