import numpy as np
from .activations.sigmoid import sigmoid
from .loss import *

class FeedForward:

    """
    FeedForward Neural Network API
    Author Name:    Rajkumar Conjeevaram Mohan
    Author Email:   rajkumarcm@yahoo.com
    """

    M                   =   None
    output_dim          =   None
    lr                  =   None
    hidden_layers       =   None
    hidden_neurons      =   None
    activation_functions=   []
    weights             =   []
    #outputs             =   []

    def __init__(self,
                 n_features,
                 n_out,
                 lr = 1e-3,
                 hidden_layers=1,
                 hidden_neurons=[1],
                 activation_functions=None
                 ):
        """---------------------------------------------------------------------------------------------------------
        FeedForward Neural Network
        :param n_features: Input data dimension
        :param n_out: Output data dimension
        :param lr: (Optional) Learning rate can be automatically adjusted depending on the optimization algorithm
        :param hidden_layers: (Optional) Number of hidden layers. By default, this is set to 1
        :param hidden_neurons: (Optional) List in which element refers to the number of neurons in each hidden layer.
        Size of this list must match with that of hidden_layers
        :param activation_function: (Optional) List in which each element refers to the activation function to be
        used by corresponding layer.
        ---------------------------------------------------------------------------------------------------------"""

        self.M                    =   n_features+1 # +1 to account for bias
        self.output_dim           =   n_out
        self.lr                   =   lr
        self.hidden_layers        =   hidden_layers
        self.hidden_neurons       =   hidden_neurons
        # self.outputs              =   [None]*(hidden_layers+1)
        self.output_layer         =   hidden_layers-1+1 # Deliberately coded to avoid confusion
        self.activation_functions =   activation_functions

        if activation_functions is None:
            self.activation_functions = [sigmoid]*(hidden_layers+1)

        for l in range(self.hidden_layers):
            if l == 0:
                self.weights.append(np.random.standard_normal(size=[self.M,hidden_neurons[l]]))
            else:
                self.weights.append(np.random.standard_normal(size=[hidden_neurons[l-1], hidden_neurons[l]]))
        self.weights.append(np.random.standard_normal(size=[hidden_neurons[hidden_layers-1], n_out]))


    def __forward_pass__(self,data):

        outputs = [None]*(self.hidden_layers+1)

        # Compute the output for each hidden layer and finally output layer
        for l in range(self.hidden_layers+1):
            if l == 0:
                outputs[l] = self.activation_functions[l].get(np.dot(data, self.weights[l]))
            else:
                outputs[l] = self.activation_functions[l].get(np.dot(outputs[l-1], self.weights[l]))

        return outputs

    def forward_pass(self,data):
        """
        Computes the output of the network, given data
        :param data: Should be of shape: [N,M], where N = # of samples, and M = # of variables/features
        :return: Tuple consisting of outputs from each layer
        """
        data = np.hstack((np.ones([data.shape[0], 1]), data))
        outputs = self.__forward_pass__(data)
        return outputs[self.output_layer]

    # Give the flexibility to use user-defined loss functions here,
    # and add more optimization options
    def __update_weights__(self,data,actual,outputs,loss_d=sse_derivative):
        """
        Computes error gradient at each layer, and updates the weights using Gradient Descent
        :param data:        Input data with shape [N,M]
        :param actual:      Target data [N]
        :param outputs:     Network outputs from forward_pass
        :param loss_d:      Derivative of loss function. Custom functions can be used.
        :return:            Updated weights, and gradients
        """

        gradients = [None]*(self.hidden_layers+1) # +1 to account for output layer

        for l in np.arange(self.output_layer,-1,-1):

            if l == self.output_layer:
                # (dl/dz)-loss derivative (dz/do)-activation function derivative (do/dw)-input to this layer i.e., the
                # output from previous layer
                gradients[l] = loss_d(actual,outputs[self.output_layer]) * \
                                                          self.activation_functions[l].derivative(outputs[self.output_layer])
            else:
                # (dl/dz dz/do)-previous layer (do/dh)-previous layer (dh/dw)-current layer
                gradients[l] = np.dot(gradients[l+1],self.weights[l+1].T) * \
                                                          self.activation_functions[l].derivative(outputs[l])

            if l-1 >= 0:
                self.weights[l] -= self.lr * (np.dot(outputs[l-1].T,gradients[l]))/outputs[l].shape[0]
            elif l-1 == -1:
                self.weights[l] -= self.lr * (np.dot(data.T, gradients[l]))/outputs[l].shape[0]

        return self.weights,gradients

    def train(self,x,y,loss=sse,loss_d=sse_derivative):
        x = np.hstack((np.ones([x.shape[0],1]),x))
        outputs = self.__forward_pass__(data=x)
        _,_ = self.__update_weights__(data=x,actual=y,outputs=outputs,loss_d=loss_d)
        return loss(y,outputs[self.output_layer])