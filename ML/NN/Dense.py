from activations import *


class Dense:
    W = None

    def __init__(self, input_dim, output_dim):
        """
        Dense / Fully-Connected layer
        :param input_dim: Feature length
        :param output_dim: Number of classes / vocabulary_size
        """

        self.W = np.random.standard_normal(size=[input_dim, output_dim]).astype(np.float16)

    def forward(self, X):
        return softmax_2d(X @ self.W)

    def backward(self, inc_grad):
        """
        Computes the local gradient and returns them
        :param inc_grad: incoming gradient / error signal
        :return: the effect of incoming gradient with respect to
        the input of this dense layer
        """

        # Self note:
        # o: output of this layer
        # incoming grad: d/d act (E)  // with respect to the activation
        # local grad: d/do (act) d/dx (o)
        # using chain rule: dE/dact * d(act)/do * d(o)/dx
        # dE/dact - derivative of cross-entropy

        return np.sum(inc_grad, axis=0) @ self.W.T

    def update_weight(self, inc_grad, X):
        # C x N N x F = C x F tranpose -> F x C
        # self.W -= 1e-3 * (inc_grad.T @ X).T
        grad = None
        T, _, _ = X.shape
        for t in range(T):
            if grad is None:
                grad = X[t].T @ inc_grad[t]
            else:
                grad += X[t].T @ inc_grad[t]

        self.W -= 1e-3 * grad
