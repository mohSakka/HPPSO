"""NumPy-based feedforward neural network for metaheuristic weight training."""

from __future__ import annotations

import numpy as np


class SimpleNeuralNetwork:
    """Two-hidden-layer MLP with sigmoid activations and linear output."""

    def __init__(self, input_size: int, hidden_size1: int, hidden_size2: int, output_size: int):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        h1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        h2 = self.sigmoid(np.dot(h1, self.W2) + self.b2)
        return np.dot(h2, self.W3) + self.b3

    def get_weights_flat(self):
        return np.concatenate(
            [self.W1.flatten(), self.b1.flatten(), self.W2.flatten(), self.b2.flatten(), self.W3.flatten(), self.b3.flatten()]
        )

    def set_weights_flat(self, flat_weights):
        idx = 0
        size = self.input_size * self.hidden_size1
        self.W1 = flat_weights[idx : idx + size].reshape(self.input_size, self.hidden_size1)
        idx += size
        self.b1 = flat_weights[idx : idx + self.hidden_size1].reshape(1, self.hidden_size1)
        idx += self.hidden_size1
        size = self.hidden_size1 * self.hidden_size2
        self.W2 = flat_weights[idx : idx + size].reshape(self.hidden_size1, self.hidden_size2)
        idx += size
        self.b2 = flat_weights[idx : idx + self.hidden_size2].reshape(1, self.hidden_size2)
        idx += self.hidden_size2
        size = self.hidden_size2 * self.output_size
        self.W3 = flat_weights[idx : idx + size].reshape(self.hidden_size2, self.output_size)
        idx += size
        self.b3 = flat_weights[idx : idx + self.output_size].reshape(1, self.output_size)


def mean_squared_error(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def nn_objective_function(flat_weights, nn_model, X, y_true):
    nn_model.set_weights_flat(flat_weights)
    predictions = nn_model.forward(X)
    return mean_squared_error(y_true, predictions)
