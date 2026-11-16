"""Adaptive neural-network MAGD from notebooks/magd_algorithm_demo.ipynb."""

import numpy as np


SETTINGS = {
    "learning_rate": 0.03,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-8,
    "gradient_clip_norm": 1.0,
    "convergence_tolerance": 1e-4,
    "convergence_window": 10,
    "reduction_interval": 10,
}


def run_magd(oracle, lower, upper, population, seed):
    """Start with the shared population size and prune agents as in the NN demo."""
    agents = np.random.RandomState(seed).uniform(lower, upper, size=(population, len(lower)))
    momentum = np.zeros_like(agents)
    variance = np.zeros_like(agents)
    window = SETTINGS["convergence_window"]
    history = np.zeros((population, window))
    beta1, beta2 = SETTINGS["beta1"], SETTINGS["beta2"]

    def evaluate(positions):
        # Cache the gradient of each evaluated candidate for its next update.
        scores, gradients = zip(*(oracle.value_and_grad(agent) for agent in positions))
        return np.asarray(scores), np.stack(gradients)

    _, gradients = evaluate(agents)
    epoch = 0
    while oracle.evaluations < oracle.budget:
        epoch += 1
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        clip = SETTINGS["gradient_clip_norm"]
        gradients *= clip / np.maximum(clip, norms)
        momentum = beta1 * momentum + (1 - beta1) * gradients
        variance = beta2 * variance + (1 - beta2) * gradients**2
        corrected_m = momentum / (1 - beta1**epoch)
        corrected_v = variance / (1 - beta2**epoch)
        step = SETTINGS["learning_rate"] * corrected_m / (np.sqrt(corrected_v) + SETTINGS["epsilon"])
        np.clip(agents - step, lower, upper, out=agents)
        scores, gradients = evaluate(agents)

        keep = np.ones(len(agents), dtype=bool)
        if epoch > window and len(agents) > 1:
            converged = np.all(np.abs(history - scores[:, None]) < SETTINGS["convergence_tolerance"], axis=1)
            keep &= ~(converged & (scores > scores.min()))
        if epoch % SETTINGS["reduction_interval"] == 0 and keep.sum() > 1:
            keep &= ~(scores > scores[keep].mean())

        history = np.roll(history, -1, axis=1)
        history[:, -1] = scores
        if not keep.all():
            agents, momentum, variance = agents[keep], momentum[keep], variance[keep]
            gradients, history = gradients[keep], history[keep]
