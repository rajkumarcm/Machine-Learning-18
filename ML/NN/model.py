import numpy as np
from loss import *
from LSTM import LSTM
from Dense import Dense
from os.path import abspath, join
from matplotlib import pyplot as plt
import time
import cProfile

bias_term = 1
vocab_size = 8000
hidden_size = 1000
n_batches = 1000
tr_path = r"D:\Data\language_modeling\training"
vl_path = r"D:\Data\language_modeling\validation"
batch_size = 30

lstm = LSTM(input_dim=vocab_size+bias_term, output_dim=hidden_size, batch_size=batch_size, T=175)
dense = Dense(input_dim=hidden_size, output_dim=vocab_size)

def infer(X_tr):
    h_state = lstm.forward(X_tr, training=True)  # list of length T, each whose dim: N x H_S
    T = len(h_state)
    output = np.zeros([T, batch_size, vocab_size])
    for t in range(T):
        output[t] = dense.forward(h_state[t])
    return h_state, output

def loss(target, obtained):
    err = cross_entropy(target, obtained)  # returns TxNx1
    return np.mean(err)

def fit(training_data, validation_data):
    X_tr, Y_tr = training_data
    X_vl, Y_vl = validation_data

    # On training data
    T, N, _ = X_tr.shape
    X_tr = np.pad(X_tr, pad_width=((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1)
    result = None
    cProfile.runctx('result = infer(X_tr)', {}, {"X_tr": X_tr, 'infer': infer})
    h_state, output = infer(X_tr)
    tr_err = loss(Y_tr, output)

    # On validation data
    X_vl = np.pad(X_vl, pad_width=((0, 0), (0, 0), (1, 0)), mode='constant', constant_values=1)
    _, vl_output = infer(X_vl)
    vl_err = loss(Y_vl, vl_output)

    inc_grad = output - Y_tr
    dense.update_weight(inc_grad, h_state)
    inc_grad = dense.backward(inc_grad)
    cProfile.runctx("lstm.weight_update(inc_grad, X_tr)", {}, {'inc_grad': inc_grad, 'X_tr': X_tr,
                                                               'lstm': lstm,
                                                               'lstm.weight_update': lstm.weight_update})

    return tr_err, vl_err

# def train(X_tr, Y_tr):
plt.figure()
for epoch in range(1):
    for batch_id in range(1):
        # Load the data
        training_data = np.load(join(tr_path, 'batch_%d.npy' % batch_id), allow_pickle=True)
        validation_data = np.load(join(vl_path, 'batch_%d.npy' % batch_id), allow_pickle=True)
        # X_tr, Y_tr = np.load(r'data\sample.npy', allow_pickle=True)
        tr_err, vl_err = fit(training_data, validation_data)
        plt.scatter(batch_id, tr_err, c='b', marker='+', linewidths=0.7, alpha=0.7)
        plt.scatter(batch_id, vl_err, c='r', marker='*', linewidths=0.7, alpha=0.7)
        plt.pause(0.001)

print("checkpoint...")