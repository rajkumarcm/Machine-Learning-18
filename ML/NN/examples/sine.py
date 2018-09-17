import ML.NN.feedforward as ff
import ML.NN.loss as loss
from ML.NN.activations import *
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    X = np.linspace(0, 200, 400).reshape([-1,1])
    y = np.sin(X)

    X_vl = np.linspace(200, 400, 400).reshape([-1,1])
    y_vl = np.sin(X_vl)

    # Network hyperparameters
    epochs = 1000
    layers = 3
    neurons = [10,10,10]

    tr_loss = np.zeros([epochs])
    vl_loss = np.zeros([epochs])
    activation_f = ([relu]*(layers)).append(tanh)
    plt.figure()

    print("Initialising network...")
    ff = ff.FeedForward(n_features=1, lr=0.01, n_out=1, hidden_layers=layers, \
                        hidden_neurons=neurons, activation_functions=activation_f)

    print("Training initiated\n----------------------------------------------------------------")
    for epoch in range(epochs):
        tr_loss[epoch] = ff.train(x=X, y=y)
        O_vl = ff.forward_pass(data=X_vl)
        vl_loss[epoch] = loss.sse(actual=y_vl, predicted=O_vl)
        print("Epoch %d: training loss: %.3f, validation loss: %.3f"%(epoch, tr_loss[epoch], vl_loss[epoch]))
        plt.scatter(epoch, tr_loss[epoch], c='b')
        plt.scatter(epoch, vl_loss[epoch], c='r')
        plt.pause(0.01)
        if epoch == epochs-1:
            print("Debug...")

    plt.show()