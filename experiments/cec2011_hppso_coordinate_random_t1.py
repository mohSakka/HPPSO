"""T1 rerun of default HPPSO with per-coordinate attraction draws."""

from pathlib import Path

from experiments import cec2011_hppso_epoch_random as driver
from hppso.experiments.runners import run_hppso


def trial(job, output, fingerprint):
    driver.base.DEFAULT_ALGORITHMS['HPPSO'] = run_hppso
    return driver.base.trial(job, output, fingerprint)


if __name__ == '__main__':
    driver.main(dict(
        title='T1 default HPPSO with per-coordinate attraction draws',
        folder='cec2011_hppso_coordinate_random_t1_200k', port=8790, problems={1},
        sampling=dict(attraction_draws='per particle and coordinate', personality_reinitialization=True),
        description='All four traits use default settings. Each attraction coefficient is drawn per coordinate.',
        trial=trial, source=Path(__file__),
        dependencies=(driver.base.ROOT / 'src/hppso/experiments/runners.py',),
    ))
