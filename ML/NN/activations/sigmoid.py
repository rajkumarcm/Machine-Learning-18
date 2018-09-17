import numpy as np

class sigmoid:

    def get(x):
        return (1 + np.exp(-x))** -1

    def derivative(x):
        return x * (1 - x)