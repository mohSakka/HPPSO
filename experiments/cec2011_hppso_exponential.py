"""Damp HPPSO curiosity and aggressiveness by evaluation progress."""
from experiments import cec2011_hppso_epoch_random as driver
from pathlib import Path
import random
import numpy as np
from hppso.algorithms.hppso import HPPSO


TERMINAL_FACTOR = 0.1


class ExponentialHPPSO(HPPSO):
    def _start_epoch(self):
        objective = self.obj_func
        progress = (objective.count - self.n_pop) / max(1, objective.horizon - self.n_pop)
        self.decay = TERMINAL_FACTOR ** min(1.0, max(0.0, progress))
        self.alpha_curio = 0.05 * self.decay

    def _effective_aggressiveness(self):
        return super()._effective_aggressiveness() * self.decay


def run_hppso(objective, dimension, lower, upper, population, iterations, seed):
    random.seed(seed)
    np.random.seed(seed)
    return ExponentialHPPSO(objective, n_pop=population, dimensions=dimension,
                            max_it=iterations, bounds=(lower, upper)).optimize()


def trial(job, output, fingerprint):
    driver.base.DEFAULT_ALGORITHMS['HPPSO'] = run_hppso
    return driver.base.trial(job, output, fingerprint)


if __name__ == '__main__':
    driver.main(dict(title='Exponentially damped HPPSO', folder='cec2011_hppso_exponential_200k', port=8781,
                     sampling=dict(method='exponential decay by evaluation count', terminal_factor=TERMINAL_FACTOR,
                                   alpha_initial=0.05, inertia_floor=0.4, damped_aggressiveness_weight=0.5,
                                   progress='(evaluations - population) / (schedule_horizon - population), clipped to [0,1]'),
                     description='Curiosity alpha and the aggressiveness contribution decay exponentially with evaluation count.',
                     trial=trial, source=Path(__file__)))
