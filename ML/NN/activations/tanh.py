import numpy as np

class tanh:

    def get(x):
        return np.tanh(x)

    def derivative(x):
        return 1 - (x**2)