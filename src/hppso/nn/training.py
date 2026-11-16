"""Shared full-batch loss, backpropagation, and evaluation accounting."""

import numpy as np
from scipy.special import log_softmax
from sklearn.metrics import accuracy_score, f1_score


class BudgetReached(Exception):
    """The next training evaluation would exceed the budget."""


def loss_and_gradient(network, X, y, classification, *, gradient=False):
    if gradient:
        h1 = network.sigmoid(X @ network.W1 + network.b1)
        h2 = network.sigmoid(h1 @ network.W2 + network.b2)
        prediction = h2 @ network.W3 + network.b3
    else:
        prediction = network.forward(X)
    if classification:
        logs = log_softmax(prediction, axis=1)
        loss = -np.sum(y * logs) / len(X)
        if gradient:
            delta = (np.exp(logs) - y) / len(X)
    else:
        error = prediction - y
        loss = np.mean(error ** 2)
        if gradient:
            delta = 2 * error / y.size
    if not gradient:
        return float(loss)
    delta2 = (delta @ network.W3.T) * h2 * (1 - h2)
    delta1 = (delta2 @ network.W2.T) * h1 * (1 - h1)
    arrays = (X.T @ delta1, delta1.sum(axis=0), h1.T @ delta2, delta2.sum(axis=0),
              h2.T @ delta, delta.sum(axis=0))
    return float(loss), np.concatenate([array.ravel() for array in arrays])


class TrainingOracle:
    def __init__(self, network, X, y, classification, budget):
        self.network = network
        self.X, self.y = X, y
        self.classification = classification
        self.budget = budget
        self.evaluations = self.gradient_evaluations = 0
        self.best_score = float("inf")
        self.best_weights = None
        self.trace = []

    def _evaluate(self, weights, gradient):
        if self.evaluations >= self.budget:
            raise BudgetReached
        self.evaluations += 1
        self.gradient_evaluations += int(gradient)
        self.network.set_weights_flat(np.asarray(weights))
        result = loss_and_gradient(self.network, self.X, self.y, self.classification, gradient=gradient)
        score = result[0] if gradient else result
        if np.isfinite(score) and score < self.best_score:
            self.best_score = score
            self.best_weights = np.asarray(weights).copy()
            self.trace.append((self.evaluations, score))
        return result if gradient or np.isfinite(score) else float("inf")

    def __call__(self, weights):
        return self._evaluate(weights, False)

    def value_and_grad(self, weights):
        return self._evaluate(weights, True)


def adam(oracle, lower, upper, seed):
    weights = np.random.default_rng(seed).uniform(lower, upper)
    m, v = np.zeros_like(weights), np.zeros_like(weights)
    for step in range(1, oracle.budget + 1):
        _, grad = oracle.value_and_grad(weights)
        m = .9 * m + .1 * grad
        v = .999 * v + .001 * grad ** 2
        update = (m / (1 - .9 ** step)) / (np.sqrt(v / (1 - .999 ** step)) + 1e-8)
        weights = np.clip(weights - .001 * update, lower, upper)


def predictive_metrics(network, X, y, target_scaler):
    output = network.forward(X)
    if target_scaler is None:
        labels, predicted = y.argmax(axis=1), output.argmax(axis=1)
        return dict(cross_entropy=float(-np.sum(y * log_softmax(output, axis=1)) / len(X)),
                    accuracy=float(accuracy_score(labels, predicted)),
                    macro_f1=float(f1_score(labels, predicted, labels=np.arange(y.shape[1]),
                                            average="macro", zero_division=0)))
    error = target_scaler.inverse_transform(output) - target_scaler.inverse_transform(y)
    return dict(mse=float(np.mean(error ** 2)), rmse=float(np.sqrt(np.mean(error ** 2))),
                mae=float(np.mean(np.abs(error))))
