import numpy as np


def sigmoid(x):
    return x / (1 + np.exp(-x))


# testing sigmoid with 6

print(sigmoid(6))
