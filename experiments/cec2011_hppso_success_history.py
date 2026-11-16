"""CEC2011 ablation of success-based personality resampling."""

from pathlib import Path
import random

import numpy as np
import pandas as pd

from experiments import cec2011_hppso_epoch_random as driver
from hppso.algorithms.hppso import HPPSO


UNIFORM_FRACTION = 0.2
MEMORY_EPOCHS = 10


class SuccessHistoryHPPSO(HPPSO):
    uniform_fraction = UNIFORM_FRACTION

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_traits = np.empty((0, 4))
        self.history_weights = np.empty(0)
        self.history_epochs = np.empty(0, dtype=int)
        self.epoch = 0

    def _refresh_personalities(self, improved, previous_best, reinit):
        self.epoch += 1
        recent = self.history_epochs > self.epoch - MEMORY_EPOCHS
        comparable = improved & np.isfinite(previous_best) & np.isfinite(self.p_best_fitness)
        gains = previous_best[comparable] - self.p_best_fitness[comparable]
        if gains.size:
            # Give each successful epoch equal total weight.
            gains = gains / gains.max()
            gains /= gains.sum()
        self.history_traits = np.concatenate((self.history_traits[recent], self.personality[comparable]))
        self.history_weights = np.concatenate((self.history_weights[recent], gains))
        self.history_epochs = np.concatenate((self.history_epochs[recent], np.full(gains.size, self.epoch)))

        super()._refresh_personalities(improved, previous_best, reinit)
        if self.history_weights.size and np.any(reinit):
            stagnant = np.flatnonzero(reinit)
            guided = stagnant[np.random.rand(stagnant.size) >= self.uniform_fraction]
            sampled = np.random.choice(self.history_weights.size, size=guided.size,
                                       p=self.history_weights / self.history_weights.sum())
            self.personality[guided] = self.history_traits[sampled]


def run_hppso(objective, dimension, lower, upper, population, iterations, seed, optimizer=SuccessHistoryHPPSO):
    random.seed(seed)
    np.random.seed(seed)
    return optimizer(objective, n_pop=population, dimensions=dimension,
                     max_it=iterations, bounds=(lower, upper)).optimize()


def trial(job, output, fingerprint):
    driver.base.DEFAULT_ALGORITHMS['HPPSO'] = run_hppso
    return driver.base.trial(job, output, fingerprint)


def report(output, baseline, rows, title, description):
    completed = driver.report(output, baseline, rows, title, description)
    results = pd.read_csv(output / 'tables/performance_and_ranks.csv')
    learned = results[results.algorithm.eq('HPPSO')].set_index('problem')['mean']
    fixed = baseline[baseline.algorithm.eq('HPPSO')].set_index('problem')['mean']
    comparison = pd.DataFrame({'success_history_mean': learned, 'default_mean': fixed.reindex(learned.index)})
    difference = comparison.success_history_mean - comparison.default_mean
    tied = np.isclose(comparison.success_history_mean, comparison.default_mean, rtol=1e-12, atol=1e-12)
    comparison['winner'] = np.select([tied, difference < 0], ['Tie', 'Success history'], default='Default')
    comparison.to_csv(output / 'tables/vs_default.csv')
    page = output / 'tables/index.html'
    with page.open('a', encoding='utf-8') as file:
        file.write('<h2>Direct comparison with default HPPSO</h2>'
                   '<p>Scores are means over the three matched seeds.</p>'
                   + comparison.to_html(float_format=lambda x: f'{x:.6g}'))
    return completed


if __name__ == '__main__':
    driver.main(dict(title='Success-history HPPSO', folder='cec2011_hppso_success_history_200k', port=8782,
                     sampling=dict(memory_epochs=MEMORY_EPOCHS, uniform_fraction=UNIFORM_FRACTION,
                                   success='strict personal-best improvement', weighting='gain within epoch'),
                     description='Stagnant particles can inherit traits from recent personal-best improvements.',
                     trial=trial, source=Path(__file__), report=report))
