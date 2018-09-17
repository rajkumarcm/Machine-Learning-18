import numpy as np

def sse(actual,predicted):
    temp = (actual - predicted) ** 2
    return np.sum(temp)/actual.shape[0]

def sse_derivative(actual,predicted):
    return -2 * (actual - predicted)