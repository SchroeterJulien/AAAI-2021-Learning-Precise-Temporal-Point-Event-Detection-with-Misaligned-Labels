# File containing all functions for the SMT-Drums dataset creation

import librosa
from multiprocessing import Pool
import numpy as np
import os
import os.path
import soundfile as sf
import xml.etree.ElementTree as ET

from AudioProcessing.mel_spectrogram import *

# Channel names
drum_list = ['HH', 'KD', 'SD']
cores = 40

# Split specification (train, validation, split)
spilt_size = [.7, .15, .15]

# General settings
settings = {}
settings['sample_rate'] = 44100
settings['nchannels'] = len(drum_list)  # todo: pass this through config
settings['time_steps'] = 370  # todo: pass this through config

# Spectrogram setting # todo: pass this through config
spectrum_settings = {}

spectrum_settings['frame_size'] = 0.025
spectrum_settings['frame_stride'] = 0.005  # 0.01
spectrum_settings['number_filter'] = 72
spectrum_settings['NFFT'] = 4096


def splitDataset(seed=42, type='random', subset_criterion=None):
    """
    Split the extracts into train, validation and test set
    :param seed: random seed in case of 'random' sampling
    :param type: either 'random' or 'subset' indicating which kind of sampling to use
    :param subset_criterion: substring in the file name that characterizes a test file (for subset sampling only)
    :return: train, validation, test (file path)
    """

    # Find all audio files in subdirectories
    np.random.seed(seed)
    audio_files = []
    for dirpath, dirnames, filenames in os.walk("."):
        for filename in [f for f in filenames if f.endswith(".wav")]:
            audio_files.append(os.path.join(dirpath, filename))

    # Discard #train extracts
    audio_files = np.array(audio_files)[np.logical_not(['train' in x for x in audio_files])]

    # Random sampling
    if type == 'random':
        shuffled_list = np.random.permutation(np.unique([file.split("#")[0] for file in audio_files]))
        train_list = shuffled_list[:int(spilt_size[0] * len(shuffled_list))]
        validation_list = shuffled_list[
                          int(spilt_size[0] * len(shuffled_list)):int(
                              (spilt_size[0] + spilt_size[2]) * len(shuffled_list))]
        test_list = shuffled_list[int((spilt_size[0] + spilt_size[2]) * len(shuffled_list)):]

    # Sampling based on criterion
    elif type == "subset":
        if subset_criterion is None:
            raise ValueError('No subset criterion entered...')
        else:
            name = np.unique([file.split("#")[0] for file in audio_files])
            idx_test = [subset_criterion in x for x in name]

            # train sample
            train_list = name[np.logical_not(idx_test)]

            # Out-of-sample
            out_samples = np.random.permutation(name[idx_test])
            validation_list = out_samples[:len(out_samples) // 2]
            test_list = out_samples[len(out_samples) // 2:]

    else:
        print("Type not supported")

    # Save the splits
    test_set = np.array([file + "#MIX.wav" for file in test_list])
    validation_set = np.array([file + "#MIX.wav" for file in validation_list])
    train_set = np.array([file + "#MIX.wav" for file in train_list])

    return train_set, validation_set, test_set


def parseXML(file):
    """
    Parse the annotation xml files
    :param file: path to the xml file
    :return: drum hits annotations
    """

    # Initialize
    root = ET.parse(file).getroot()

    # Iterate over events
    idx_hit = 0
    labels = np.zeros([1000, 2])
    for event in root[1]:
        labels[idx_hit, 0] = event[1].text
        labels[idx_hit, 1] = [i for i, x in enumerate(['HH', 'KD', 'SD']) if x == event[3].text][0]

        idx_hit += 1

    return labels[:idx_hit, :]


def loadData(name="", seed=42, type="random", subset_criterion=""):
    train, validation, test = splitDataset(seed, type=type, subset_criterion=subset_criterion)

    np.save('models/' + name + '_train_files.npy', train)
    np.save('models/' + name + '_test_files.npy', test)

    return train, validation, test


def processSplitSample(input):

    file = input[0]
    infer = input[1]
    labelNoise = input[2]
    config = input[3]

    print(file, infer)

    # Load audio
    signal, sr = sf.read(file)
    assert (sr == settings['sample_rate'])

    if '#MIX' in file:
        label_raw = parseXML(file.replace('audio', 'annotation_xml').replace('wav', 'xml'))
    else:
        label_raw = parseXML(file.replace('audio', 'annotation_xml').split('#')[0] + "#MIX.xml")

    if labelNoise:
        label_raw[:, 0] += config['label_noise'] * np.random.randn(len(label_raw[:, 0]))


    final_data = np.zeros(
        [int(np.ceil(signal.shape[0] / (sr * config['split_step'])) * config['augmentation_factor']),
         settings['time_steps'],
         2 * spectrum_settings['number_filter']])

    labels = np.zeros([int(np.ceil(signal.shape[0] / (sr * config['split_step'])) * config['augmentation_factor']),
                       settings['time_steps'],
                       settings['nchannels']], np.int8)

    factor = np.zeros([int(np.ceil(signal.shape[0] / (sr * config['split_step'])) * config['augmentation_factor'])])
    idx_sample = 0
    for idx_split in range(int(np.ceil(signal.shape[0] / (sr * config['split_step'])))):
        for idx_augmentation in range(config['augmentation_factor']):

            signal_window = signal[int(idx_split * sr * config['split_step']):int(
                sr * (idx_split * config['split_step'] + config['split_length']))]
            flag_process = True
            if idx_augmentation == 0:
                stretching_factor = 1

            elif idx_augmentation >= 4 and labelNoise and config['stack_augmentation']:
                print('stack')

                # Combine two extracts
                stretching_factor = 1

                idx_tmp = np.random.randint(0, int(np.ceil(signal.shape[0] / (sr * config['split_step']))) - 1)
                signal_new = signal[int(idx_tmp * sr * config['split_step']):int(
                    sr * (idx_tmp * config['split_step'] + config['split_length']))]

                if signal_window.shape[0] == signal_new.shape[0]:
                    signal_window += signal_new
                else:
                    flag_process = False
            else:
                stretching_factor = np.random.rand() / 4 + 7 / 8

            factor[idx_sample] = stretching_factor

            extended_signal = np.concatenate([1e-5 * np.random.randn(int(config['pad'] * sr)), signal_window])

            spectrum = melSpectrogram(extended_signal + np.ones(extended_signal.shape[0]) * config['noise'],
                                          sr * stretching_factor, spectrum_settings['frame_size'],
                                          spectrum_settings['frame_stride'],
                                          spectrum_settings['number_filter'], spectrum_settings['NFFT'],
                                          normalized=True)

            # spectrum = np.exp(spectrum/100) #for bass
            final_data[idx_sample, :spectrum.shape[0], :spectrum_settings['number_filter']] = spectrum
            # First order-derivative
            final_data[idx_sample, 1:spectrum.shape[0], spectrum_settings['number_filter']:] = np.diff(spectrum,
                                                                                                           n=1,
                                                                                                           axis=0)

            for ii in range(label_raw.shape[0]):
                if np.round(label_raw[ii, 0], 2) - idx_split * config['split_step'] >= 0 and label_raw[
                        ii, 0] - idx_split * config['split_step'] <= config['split_length'] and (
                            config['pad'] > 0 or np.round(label_raw[ii, 0] - idx_split * config['split_step'], 2) > 0):
                    labels[idx_sample, signal2spectrumTime((np.round(label_raw[ii, 0] - idx_split * config['split_step'], 3) + config[
                                'pad']) / stretching_factor * settings['sample_rate']), int(label_raw[ii, 1])] += 1

            if idx_augmentation >= 4 and labelNoise and config['stack_augmentation'] and flag_process:
                print('stack')
                for ii in range(label_raw.shape[0]):
                    if np.round(label_raw[ii, 0], 2) - idx_tmp * config['split_step'] >= 0 and label_raw[
                            ii, 0] - idx_tmp * config['split_step'] <= config['split_length'] and (
                                config['pad'] > 0 or np.round(label_raw[ii, 0] - idx_tmp * config['split_step'], 2) > 0):
                        labels[idx_sample, signal2spectrumTime(
                                (np.round(label_raw[ii, 0] - idx_tmp * config['split_step'], 3) + config[
                                    'pad']) / stretching_factor * settings['sample_rate']), int(label_raw[ii, 1])] += 1

            # Single instrument: select only
            for idx_drum in range(len(drum_list)):
                if '#' + drum_list[idx_drum] in file:
                    selector = [x for x in range(len(drum_list)) if x != idx_drum]
                    labels[idx_sample, :, selector] = 0

            if idx_augmentation >= 4 and labelNoise and config['stack_augmentation']:
                if np.sum(labels[idx_sample, :,:])>= config['max_occurence']-2:
                    labels[idx_sample, :,:] = 0
                    final_data[idx_sample, : ,:] = 0
                    factor[idx_sample] = 0
            else:
                idx_sample += 1

    labels[labels > 1] = 1
    return final_data[:idx_sample,:,:], labels[:idx_sample,:,:], factor[:idx_sample], np.array([file] * int(idx_sample))


def generateSplitDataset(file_list, config, infer=False, labelNoise=False):
    p = Pool(cores)
    data_simulated = p.map(processSplitSample, [(x, infer, labelNoise, config) for x in file_list])
    p.close()

    x_data = np.concatenate([x[0][:, :, :] for x in data_simulated], axis=0)
    y_data = np.concatenate([x[1][:, :, :] for x in data_simulated], axis=0)
    factor_list = np.concatenate([x[2] for x in data_simulated], axis=0)
    file_list = np.concatenate([x[3] for x in data_simulated], axis=0)

    y_data = y_data.transpose([0, 2, 1])
    y_data_transformed = y_data.reshape([y_data.shape[0] * y_data.shape[1], y_data.shape[2]])

    Y_label = np.zeros([len(np.sum(y_data_transformed, axis=1)), config['max_occurence']])
    Y_label[np.arange(len(np.sum(y_data_transformed, axis=1))), np.sum(y_data_transformed, axis=1)] = 1
    Y_label = Y_label.reshape([y_data.shape[0], y_data.shape[1], config['max_occurence']])

    return x_data, Y_label, y_data, factor_list, file_list


# Utility functions
def signal2spectrumTime(time):
    """
    Converts signal time (seconds) into spectrogram bin location
    :param time: in seconds
    :return: corresponding spectrogram bin location
    """
    if time <= settings['sample_rate'] * spectrum_settings['frame_size']:
        return int(0)
    else:
        time -= settings['sample_rate'] * spectrum_settings['frame_size']
        return int(1 + time // (settings['sample_rate'] * spectrum_settings['frame_stride']))


def spectrum2signalTime(time):
    """
    Converts spectrogram bin location into signal time (seconds)
    :param time: spectrogram bin location
    :return: corresponding signal time (seconds)
    """
    if time == 0:
        return int(0)
    else:
        time -= 1
        return int(
            settings['sample_rate'] * spectrum_settings['frame_size'] + time * settings['sample_rate'] *
            spectrum_settings['frame_stride'])
