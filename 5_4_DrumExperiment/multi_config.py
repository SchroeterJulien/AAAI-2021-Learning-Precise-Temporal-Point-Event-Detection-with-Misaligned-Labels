# Configuration file
import numpy as np
import os.path


def load_configurations(extension=""):
    config = {}

    # Extension Name
    config['smoothing_width'] = np.random.uniform(0,30) 
    config['seed'] = np.random.randint(1,10000)
    config['label_noise'] = np.random.uniform(0,100) / 1000 

    config['extension'] = 'sweep2_' + str(config['seed']) + "_"\
                          + str(int(config['smoothing_width'])) + "_" \
                          + str(int(1000 * config['label_noise']))

    print(config['extension'])

    # Augmentation
    config['augmentation_factor'] = 10
    config['stack_augmentation'] = False

    if os.path.isfile('models/' + extension + '.npy'):
        print('Load from backup')
        config = np.load('models/' + extension + '.npy').item()

    else:
        config['Direct'] = True
        config['GPU_Device'] = "0"

        config['tolerence'] = 10
        config['temporal_bias'] = 0

        config['ensembling_factor'] = 0.25


        # Direct settings
        config['start_processing'] = 30000
        config['weight_indirect'] = 0.2

        #
        config['trigger_threshold'] = 0.4

        # -------------------------------


        config['niter'] = 150000

        # Network settings
        config['num_units'] = 24
        config['hidden_size'] = 16
        config['n_filters'] = [8, 16, 16, 16, 16, 16]

        # -------------------------------
        # Save and display settings
        config['show_frequency'] = 5000  # 200
        config['save_frequency'] = 5000

        # Learning settings
        config['clipping_ratio'] = 10  # 5
        config['learning_rate'] = 0.0001  # 0.0002
        config['batch_size'] = 32  # 28
        config['bool_gradient_clipping'] = True

        # -------------------------------

        # Dataset settings
        config['dataset_size'] = 8000  # 4000
        config['dataset_update_size'] = 500
        config['dataset_update_frequency'] = 500000  # 1000
        config['update_start'] = 0

        # Dataset constants
        config['time_steps'] = 370 # 370 (bk), 740
        config['n_filter'] = 144
        config['max_occurence'] = 23 # 15 (bk), 35
        config['n_channel'] = 3

        # Data settings
        config['split_length'] = 1.5 # 1.5 (bk), 3.0
        config['split_step'] = 0.5
        config['pad'] = 0.1
        config['noise'] = 0

        # Save settings
        np.save('models/' + config['extension'] + '.npy', config)

    return config
