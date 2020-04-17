"""
"""

from __future__ import division
from __future__ import print_function

from util import LoadData, Load, Save, DisplayPlot
from conv2d import conv2d as Conv2D
from nn import Affine, ReLU, AffineBackward, ReLUBackward, Softmax, CheckGrad, Train, Evaluate

import numpy as np
import copy



def InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_filters_3, num_hiddens,
            num_outputs):
    """Initializes CNN parameters.

    Args:
        num_channels:  Number of input channels.
        filter_size:   Filter size.
        num_filters_1: Number of filters for the first convolutional layer.
        num_filters_2: Number of filters for the second convolutional layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    W1 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_channels,  num_filters_1)
    W2 = 0.1 * np.random.randn(filter_size, filter_size,
                               num_filters_1, num_filters_2)
    W3 = 0.1 * np.random.randn(filter_size, filter_size,
                                num_filters_2, num_filters_3)
    W4 = 0.01 * np.random.randn(num_filters_3 * 16, num_hiddens)
    W5 = 0.01 * np.random.randn(num_hiddens, num_outputs)

    M1 = np.zeros((filter_size, filter_size,
                               num_channels,  num_filters_1))
    M2 = np.zeros((filter_size, filter_size,
                               num_filters_1, num_filters_2))
    M3 = np.zeros((filter_size, filter_size,
                                num_filters_2, num_filters_3))
    M4 = np.zeros((num_filters_3*16, num_hiddens))
    M5 = np.zeros((num_hiddens, num_outputs))
    b1 = np.zeros((num_filters_1))
    b2 = np.zeros((num_filters_2))
    b3 = np.zeros((num_filters_3))
    b4 = np.zeros((num_hiddens))
    b5 = np.zeros((num_outputs))
    Mb1 = np.zeros((num_filters_1))
    Mb2 = np.zeros((num_filters_2))
    Mb3 = np.zeros((num_filters_3))
    Mb4 = np.zeros((num_hiddens))
    Mb5 = np.zeros((num_outputs))
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'W4': W4,
        'W5': W5,
        'b1': b1,
        'b2': b2,
        'b3': b3,
        'b4': b4,
        'b5': b5,
        'M1': M1,
        'M2': M2,
        'M3': M3,
        'M4': M4,
        'M5': M5,
        'Mb1': Mb1,
        'Mb2': Mb2,
        'Mb3': Mb3,
        'Mb4': Mb4,
        'Mb5': Mb5
    }
    return model


def MaxPool(x, ratio):
    """Computes non-overlapping max-pooling layer.

    Args:
        x:     Input values.
        ratio: Pooling ratio.

    Returns:
        y:     Output values.
    """
    xs = x.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y = np.max(np.max(h, axis=4), axis=2)
    return y


def MaxPoolBackward(grad_y, x, y, ratio):
    """Computes gradients of the max-pooling layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
    """
    dy = grad_y
    xs = x.shape
    ys = y.shape
    h = x.reshape([xs[0], int(xs[1] / ratio), ratio,
                   int(xs[2] / ratio), ratio, xs[3]])
    y_ = np.expand_dims(np.expand_dims(y, 2), 4)
    dy_ = np.expand_dims(np.expand_dims(dy, 2), 4)
    dy_ = np.tile(dy_, [1, 1, ratio, 1, ratio, 1])
    dx = dy_ * (y_ == h).astype('float')
    dx = dx.reshape([ys[0], ys[1] * ratio, ys[2] * ratio, ys[3]])
    return dx


def Conv2DBackward(grad_y, x, y, w):
    """Computes gradients of the convolutional layer.

    Args:
        grad_y: Gradients wrt. the inputs.
        x:      Input values.
        y:      Output values.

    Returns:
        grad_x: Gradients wrt. the inputs.
        grad_w: Gradients wrt. the weights.
    """
    wshape = w.shape
    I = wshape[0]
    J = wshape[1]
    
    xcp = np.transpose(x,axes=[3,1,2,0])

    grad_ycp = np.transpose(grad_y,axes=[1,2,0,3])

    grad_w = Conv2D(xcp,grad_ycp,pad=(I-1,J-1))    

    # wcp = np.rot90(np.rot90(copy.deepcopy(w)))
    # flip k and c

    # transpose f
    wcp = np.zeros(wshape)
    for i in xrange(0,wshape[0]):
        for j in xrange(0,wshape[1]):
            wcp[i,j,:,:] = copy.deepcopy(w[I-i-1,J-j-1,:,:])

    wcp = np.transpose(wcp, axes=[0,1,3,2])

    
    grad_x = Conv2D(copy.deepcopy(grad_y),wcp, pad=(I-1,J-1))
    return grad_x, np.transpose(grad_w, axes=[1,2,0,3])




def CNNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    x = x.reshape([-1, 32, 32, 3])
    h1c = Conv2D(x, model['W1']) + model['b1']
    h1r = ReLU(h1c)
    h1p = MaxPool(h1r, 2)
    h2c = Conv2D(h1p, model['W2']) + model['b2']
    h2r = ReLU(h2c)
    h2p = MaxPool(h2r, 2)
    h3c = Conv2D(h2p, model['W3']) + model['b3']
    h3r = ReLU(h3c)
    h3p = MaxPool(h3r, 2)
    h3p_ = np.reshape(h3p, [x.shape[0], -1])
    h4 = Affine(h3p_, model['W4'], model['b4'])
    h5r = ReLU(h4)
    y = Affine(h5r, model['W5'], model['b5'])
    var = {
        'x': x,
        'h1c': h1c,
        'h1r': h1r,
        'h1p': h1p,
        'h2c': h2c,
        'h2r': h2r,
        'h2p': h2p,
        'h3c': h3c,
        'h3r': h3r,
        'h3p': h3p,
        'h3p_': h3p_,
        'h4': h4,
        'h5r' :h5r,
        'y': y
    }
    return var


def CNNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh4r, dE_dW5, dE_db5 = AffineBackward(err, var['h5r'], model['W5'])
    dE_dh4c = ReLUBackward(dE_dh4r, var['h4'], var['h5r'])
    dE_dh3p_, dE_dW4, dE_db4 = AffineBackward(dE_dh4c, var['h3p_'], model['W4'])
    dE_dh3p = np.reshape(dE_dh3p_, var['h3p'].shape)
    dE_dh3r = MaxPoolBackward(dE_dh3p, var['h3r'], var['h3p'], 2)
    dE_dh3c = ReLUBackward(dE_dh3r, var['h3c'], var['h3r'])
    dE_dh2p, dE_dW3 = Conv2DBackward(
        dE_dh3c, var['h2p'], var['h3c'], model['W3'])
    dE_db3 = dE_dh3c.sum(axis=2).sum(axis=1).sum(axis=0)
    dE_dh2r = MaxPoolBackward(dE_dh2p, var['h2r'], var['h2p'], 2)
    dE_dh2c = ReLUBackward(dE_dh2r, var['h2c'], var['h2r'])
    dE_dh1p, dE_dW2 = Conv2DBackward(
        dE_dh2c, var['h1p'], var['h2c'], model['W2'])
    dE_db2 = dE_dh2c.sum(axis=2).sum(axis=1).sum(axis=0)
    dE_dh1r = MaxPoolBackward(dE_dh1p, var['h1r'], var['h1p'], 2)
    dE_dh1c = ReLUBackward(dE_dh1r, var['h1c'], var['h1r'])
    _, dE_dW1 = Conv2DBackward(dE_dh1c, var['x'], var['h1c'], model['W1'])
    dE_db1 = dE_dh1c.sum(axis=2).sum(axis=1).sum(axis=0)
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_dW4'] = dE_dW4
    model['dE_dW5'] = dE_dW5
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    model['dE_db4'] = dE_db4
    model['dE_db5'] = dE_db5
    pass


def CNNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    
    model['W1'] = model['W1'] - model['M1']
    model['W2'] = model['W2'] - model['M2']
    model['W3'] = model['W3'] - model['M3']
    model['W4'] = model['W4'] - model['M4']
    model['W5'] = model['W5'] - model['M5']
    model['b1'] = model['b1'] - model['Mb1']
    model['b2'] = model['b2'] - model['Mb2']
    model['b3'] = model['b3'] - model['Mb3']
    model['b4'] = model['b4'] - model['Mb4']
    model['b5'] = model['b5'] - model['Mb5']
    
    model['M1'] = momentum*model['M1'] + eps * model['dE_dW1']
    model['M2'] = momentum*model['M2'] + eps * model['dE_dW2']
    model['M3'] = momentum*model['M3'] + eps * model['dE_dW3']
    model['M4'] = momentum*model['M4'] + eps * model['dE_dW4']
    model['M5'] = momentum*model['M5'] + eps * model['dE_dW5']
    model['Mb1'] = momentum*model['Mb1'] + eps * model['dE_db1']
    model['Mb2'] = momentum*model['Mb2'] + eps * model['dE_db2']
    model['Mb3'] = momentum*model['Mb3'] + eps * model['dE_db3']
    model['Mb4'] = momentum*model['Mb4'] + eps * model['dE_db4']
    model['Mb5'] = momentum*model['Mb5'] + eps * model['dE_db5'] 


def main():

    '''
    Run on all combinations of hyper parameters
    num_epochs = 1#30
    filter_size = 5
    num_filters_1 = 8
    num_filters_2 = 16
    li_num_filters_1 = [2,50,25,2,25]
    li_num_filters_2 = [25,50,2,2,25]
    eps = [.001,.01,.1,.5,1]
    d_eps = .01
    momentum = [0,.45,.9]
    d_mom = 0.0
    batch_size=[1,20,300,750,1000]
    d_batch_size = 100
    # Input-output dimensions.
    num_channels = 1
    num_outputs = 7
    for i in range(10):
        # Initialize model.
        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        x = np.random.rand(10, 48, 48, 1) * 0.1
        CheckGrad(model, CNNForward, CNNBackward, 'W3', x)
        CheckGrad(model, CNNForward, CNNBackward, 'b3', x)
        CheckGrad(model, CNNForward, CNNBackward, 'W2', x)
        CheckGrad(model, CNNForward, CNNBackward, 'b2', x)
        CheckGrad(model, CNNForward, CNNBackward, 'W1', x)
        CheckGrad(model, CNNForward, CNNBackward, 'b1', x)

    for each in eps:
        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, each,
                  d_mom, num_epochs, d_batch_size)
        Save('results/eps_' + str(each) + '_cnn.npz', stats)

    for each in momentum:
        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, d_eps,
                  each, num_epochs, d_batch_size)
        Save('results/momentum_' + str(each) + '_cnn.npz',stats)

    for each in batch_size:
        model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, d_eps,
                  d_mom, num_epochs, each)
        Save('results/batch_size_' + str(each) + '_cnn.npz',stats)

    for index,each in enumerate(li_num_filters_1):
        model = InitCNN(num_channels, filter_size, each, li_num_filters_2[index],
                        num_outputs)
        model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, d_eps,
                  0.9, num_epochs, d_batch_size)
        Save('results/num_filters_' + str(each) + '_' + str(li_num_filters_2[index]) + '_cnn.npz',stats)


    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2,
                        num_outputs)
    model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, d_eps,
              d_mom, num_epochs, d_batch_size)
    Save('results/normal_cnn.npz',stats)'''

    """Trains a CNN."""
    model_fname = 'exp2/cnn_nonclass_model.npz'
    stats_fname = 'exp2/cnn_nonclass_stats.npz'

    # Hyper-parameters. Modify them if needed.
    eps = 0.05
    momentum = 0.0
    num_epochs = 60
    filter_size = 5
    num_filters_1 = 3
    num_filters_2 = 32
    num_filters_3 = 32
    num_hiddens = 64
    batch_size = 50

    # Input-output dimensions.
    num_channels = 3
    num_outputs = 4

    # Initialize model.
    model = InitCNN(num_channels, filter_size, num_filters_1, num_filters_2, num_filters_3, num_hiddens,
                    num_outputs)

    # Uncomment to reload trained model here.
    # model = Load(model_fname)

    # Check gradient implementation.
    # Uncomment when gradient is implemented
    print('Checking gradients...')
    x = np.random.rand(10, 32, 32, 3) * 0.1
    CheckGrad(model, CNNForward, CNNBackward, 'W3', x)

    # Train model.
    model, stats = Train(model, CNNForward, CNNBackward, CNNUpdate, eps,
                  momentum, num_epochs, batch_size)

    # Uncomment if you wish to save the model.
    # Save(model_fname, model)

    # Uncomment if you wish to save the training statistics.
    Save(stats_fname, stats)
    Save(model_fname, model)

if __name__ == '__main__':
    main()
