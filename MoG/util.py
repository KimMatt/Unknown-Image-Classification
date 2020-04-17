import numpy as np


def LoadData(fname):
    """ Loads data """

    npzfile = np.load(fname)

    inputs_train = npzfile['inputs_train'].T / 255.0
    inputs_valid = npzfile['inputs_valid'].T / 255.0
    inputs_test = npzfile['inputs_test'].T / 255.0
    target_train = npzfile['target_train']
    target_valid = npzfile['target_valid']
    target_test = npzfile['target_test']

    return inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test


def LoadDataQ4(fname):
    """ Loads data """
    npzfile = np.load(fname)

    inputs_train = npzfile['training_inputs'].T / 255.0
    inputs_valid = npzfile['valid_inputs'].T / 255.0
    inputs_test = npzfile['test_inputs'].T / 255.0
    target_train = npzfile['training_labels']
    target_valid = npzfile['valid_labels']
    target_test = npzfile['test_labels']

    # Flatten input
    train_shape = inputs_train.shape
    valid_shape = inputs_valid.shape
    test_shape = inputs_test.shape

    inputs_train = np.reshape(inputs_train, (train_shape[0] * train_shape[1], train_shape[2]))
    inputs_valid = np.reshape(inputs_valid, (valid_shape[0] * valid_shape[1], valid_shape[2]))
    inputs_test  = np.reshape(inputs_test, (test_shape[0] * test_shape[1], test_shape[2]))

    data = {
        # image
        'x_train_orange': inputs_train[:, target_train == 0],
        'x_valid_orange': inputs_valid[:, target_valid == 0],
        'x_test_orange': inputs_test[:, target_test == 0],
        'x_train_apple': inputs_train[:, target_train== 1],
        'x_valid_apple': inputs_train[:, target_valid==1],
        'x_test_apple': inputs_test[:, target_test == 1],
        'x_train_banana': inputs_train[:, target_train == 2],
        'x_valid_banana': inputs_valid[:, target_valid == 2],
        'x_test_banana': inputs_test[:, target_test == 2],
        # label
        'y_train_orange': np.zeros_like(target_train[target_train == 0]),
        'y_valid_orange': np.zeros_like(target_valid[target_valid == 0]),
        'y_test_orange': np.zeros_like(target_test[target_test == 0]),
        'y_train_apple': np.ones_like(target_train[target_train == 1]),
        'y_valid_apple': np.ones_like(target_valid[target_valid == 1]),
        'y_test_apple': np.ones_like(target_test[target_test == 1]),
        'y_train_banana': np.ones_like(target_train[target_train == 2]) * 2,
        'y_valid_banana': np.ones_like(target_valid[target_valid == 2]) * 2,
        'y_test_banana': np.ones_like(target_test[target_test == 2]) * 2
    }

    return data
