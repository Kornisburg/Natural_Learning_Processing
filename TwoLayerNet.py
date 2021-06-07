import numpy as np


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # initialize weights and biases
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, 0)
        b2 = np.random.randn(0)

        # generated layer
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2),
        ]