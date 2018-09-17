import numpy as np

class relu:

    def get(x):
        N,M = x.shape
        temp = list(map(lambda x: np.max([0,x]), x.ravel()))
        return np.reshape(temp,[N,M])

    def derivative(x):
        indices = x > 0
        grad = np.ones(x.shape)
        return grad * indices