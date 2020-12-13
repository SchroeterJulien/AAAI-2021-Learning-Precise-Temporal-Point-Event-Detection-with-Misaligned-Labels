# Script performing the inference and evaluation of smt-models

import matplotlib
matplotlib.use('Agg')

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import mir_eval
import pretty_midi

from config import load_configurations
import Display.localizationPlot as lp
from new_createDataset_v2 import *
from SoftNetworkModel import SoftNetwork



if not os.path.exists('results'):
    os.mkdir('results')

def addSweep(extension):
    # Load configuration

    if os.path.exists('results/sweep_data.npz'):
        tmp = np.load('results/sweep_data.npz')
        current_extension = tmp['arr_0']
    else:
        current_extension = []

    if extension in current_extension:
        print("< Extension already in archive")
    else:
        config = load_configurations(extension)
        assert(extension==config['extension'])


        config['temporal_bias'] = 0
        config['augmentation_factor'] = 7

        ensembling_factor = 0.25
        suppression_field = 9

        # Load out-of-sample data
        print('>> Load Dataset...')
        test_files = np.load('models/' + config['extension'] + '_test_files.npy')
        x_out_raw, _, y_out_raw, stretch_factor_out, file_list_out_raw = generateSplitDataset(test_files, config, infer=True, labelNoise=False)
        print("---", len(test_files), "---")


        print("---", config['extension'], "---")

        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        new_results = np.zeros([3,3])
        with tf.Session() as sess:
            # Restore model
            softModel = SoftNetwork(config)
            softModel.restore(sess)

            for selection_channel in range(0,3):

                x_out = np.copy(x_out_raw)
                y_out = np.copy(y_out_raw)
                file_list_out = np.copy(file_list_out_raw)


                # Select channel
                idx_channel = [0, 1, 2]
                idx_channel.remove(selection_channel)
                y_out[:, idx_channel, :] = 0

                # Single extract score
                pp = softModel.predict(sess, x_out)
                pp[:, idx_channel, :] = 0
                _, _ = lp.localizationPlot(pp, y_out, n_samples=20, dist_threshold=config['tolerence'], factor=1,
                                           bias=config['temporal_bias'], decimals=7)
                plt.close()

                # Ensembling score
                pp_trans = np.transpose(pp.reshape([pp.shape[0] // config['augmentation_factor'], config['augmentation_factor'], pp.shape[1], pp.shape[2]]), [1, 0, 2, 3])
                pp_ensemble = softModel.FastEnsembling(pp_trans, stretch_factor_out, ensembling_factor, suppression_field)

                plt.figure()
                _, _ = lp.localizationPlot(pp_ensemble, y_out[::config['augmentation_factor'], :, :], n_samples=10, dist_threshold=config['tolerence'],
                                           factor=1, bias=config['temporal_bias'], decimals=7)
                plt.close()

                _start_extract = 16
                y_ensemble = y_out[::config['augmentation_factor'], :, :]
                file_list_out = file_list_out[::config['augmentation_factor']]

                y_pasted = np.zeros([len(test_files), pp_ensemble.shape[1], 30000])
                pp_pasted = np.zeros([len(test_files), pp_ensemble.shape[1], 30000])
                ww = np.zeros([len(test_files), pp_ensemble.shape[1], 30000])
                file_out_unique = []
                previous_source = ""
                idx_source = -1
                for ii in range(len(file_list_out)):
                    if file_list_out[ii] == previous_source:
                        idx_start += int(config['split_step'] * 200)
                    else:
                        idx_start = 0
                        idx_source += 1
                        previous_source = file_list_out[ii]
                        file_out_unique.append(previous_source)

                    y_pasted[idx_source, :, idx_start:idx_start + y_ensemble[ii, :, _start_extract:].shape[1]] += y_ensemble[ii, :, _start_extract:]
                    pp_pasted[idx_source, :, idx_start:idx_start + pp_ensemble[ii, :, _start_extract:].shape[1]] += pp_ensemble[ii, :, _start_extract:]
                    ww[idx_source, :, idx_start:idx_start + pp_ensemble[ii, :, _start_extract:int(config['split_length'] * 200)+_start_extract].shape[1]] += 1

                # Normalize
                pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
                y_final = y_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] > 0

                # Load labels from file
                yy = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
                yy_list = []
                for jj in range(yy.shape[0]):
                    label_raw = np.array(parseXML(file_out_unique[jj].replace('audio', 'annotation_xml').replace('wav', 'xml')))
                    for kk in range(label_raw.shape[0]):
                        yy[jj, int(label_raw[kk, 1]), int(label_raw[kk, 0] * 200)] += 1

                    yy_list.append(label_raw[np.logical_not([x in idx_channel for x in label_raw[:,1]]),:])

                yy[:, idx_channel, :] = 0


                # Check alignment
                plt.figure()
                plt.plot(yy[0, 0, :] - y_final[0, 0, :])
                plt.close('all')

                # Final prediction cleaning
                pp_final = pp_pasted[:, :, np.sum(ww, axis=(0, 1)) > 0] / ww[:, :, np.sum(ww, axis=(0, 1)) > 0]
                pp_final_cleaning = np.zeros([pp_final.shape[0], pp_final.shape[1], pp_final.shape[2]])
                for ii in range(pp_final_cleaning.shape[0]):
                    for jj in range(pp_final_cleaning.shape[1]):
                        for tt in range(pp_final_cleaning.shape[2]):
                            if pp_final[ii, jj, tt] > 0:
                                if np.sum(pp_final[ii, jj, tt:tt + suppression_field]) >= 0.50:
                                    pp_final_cleaning[ii, jj, tt] = 1
                                    pp_final[ii, jj, tt:tt + suppression_field] = 0

                # Final score computation
                plt.figure()
                fig, _ = lp.localizationPlot(pp_final_cleaning[:, :, :], yy[:, :, :], n_samples=pp_final_cleaning.shape[0],
                                             dist_threshold=config['tolerence'],
                                             factor=1, bias=config['temporal_bias'], decimals=7)
                plt.close()


                pp_list = []
                for ii in range(pp_final.shape[0]):
                    triggers = np.zeros([0,2])
                    for jj in range(pp_final.shape[1]):
                        list_hits = np.where(pp_final_cleaning[ii,jj])[0]/200
                        triggers = np.concatenate([triggers, np.concatenate([list_hits[:,np.newaxis],np.array([jj]*len(list_hits))[:,np.newaxis]],axis=1)])
                    pp_list.append(triggers)

                fig, _ = lp.localizationPlotList(pp_list, yy_list, decimals=7, bias= 0.000,  n_samples = 20, dist_threshold=0.050)
                plt.savefig('plt/inference/' + config['extension']+ "_" + str(selection_channel))
                plt.close()

                f1_list = []
                pre_list = []
                rec_list = []
                for kk in range(0,len(yy_list)):
                    pre, rec, f1, _ = (
                        mir_eval.transcription.precision_recall_f1_overlap(
                            np.concatenate([np.array([max(x,0) for x in yy_list[kk][:, 0]])[:,np.newaxis], yy_list[kk][:, 0:1] + 1], axis=1),
                            pretty_midi.note_number_to_hz(yy_list[kk][:, 1]),
                            np.concatenate([np.array([max(x,0) for x in pp_list[kk][:, 0]])[:,np.newaxis], pp_list[kk][:, 0:1] + 1], axis=1),
                            pretty_midi.note_number_to_hz(pp_list[kk][:, 1]),
                            offset_ratio=None))
                    f1_list.append(f1)
                    pre_list.append(pre)
                    rec_list.append(rec)

                print(np.mean(f1_list), np.mean(pre_list), np.mean(rec_list))

                new_results[selection_channel,:] = np.array([np.mean(f1_list), np.mean(pre_list), np.mean(rec_list)])

                print("---", config['extension'], "---", selection_channel, "---")

        softModel.reset()

        # Reload in case other update occurred in the mean-time
        if os.path.exists('results/sweep_data.npz'):
            tmp = np.load('results/sweep_data.npz')
            current_extension = tmp['arr_0']
            current_results = tmp['arr_1']
            current = current_extension.tolist()
            current.append(extension)
            np.savez('results/sweep_data.npz', current, np.concatenate([current_results,new_results[np.newaxis, :, :]],axis=0))
        else:
            np.savez('results/sweep_data.npz', current_extension.tolist(), new_results[np.newaxis, :, :])

    # create updated image
    import sweepVisualization

if __name__ == '__main__':
    # Map command line arguments to function arguments.
    addSweep(*sys.argv[1:])


