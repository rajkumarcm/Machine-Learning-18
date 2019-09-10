import numpy as np
import logging

sigmoid = lambda x: 1/(1 + np.exp(-x))

d_sigmoid = lambda output: output * (1 - output)

def tanh(x):
    return np.tanh(x)

def d_tanh(output):
    logging.warning('formula for tanh gradient needs verification')
    return 1 - output**2

def softmax(x):
    T, N, voc_size = x.shape
    tmp1 = np.exp(x)
    tmp2 = np.sum(tmp1, axis=2).reshape([T, N, 1])
    tmp2 = np.pad(tmp2, pad_width=((0, 0), (0, 0), (0, voc_size-1)), mode='reflect')
    return tmp1/tmp2

def softmax_2d(x):
    N, H_S = x.shape
    tmp1 = np.exp(x)
    tmp2 = np.sum(tmp1, axis=1).reshape([-1, 1])
    tmp2 = np.pad(tmp2, pad_width=((0, 0), (0, H_S-1)), mode='reflect')
    return tmp1/tmp2

def d_softmax(output):
    N, H_S = output.shape
    return output * (np.eye(N, H_S) - output)