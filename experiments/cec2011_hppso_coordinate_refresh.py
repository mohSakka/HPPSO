"""CEC2011 rerun with per-coordinate draws and the requested trait settings."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_success_history as study
from hppso.algorithms.hppso import HPPSO


PARAMETERS = dict(theta=0.2, alpha_mut=0.03, eta=0.1, lam=0.4, stagnation_threshold=5)


class RefreshHPPSO(HPPSO):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs,
            socialism_threshold=PARAMETERS['theta'], alpha_curio=PARAMETERS['alpha_mut'],
            eta=PARAMETERS['eta'], lam=PARAMETERS['lam'],
            stagnation_threshold=PARAMETERS['stagnation_threshold'],
        )


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=RefreshHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='HPPSO with per-coordinate draws and five-iteration personality refreshing',
        folder='cec2011_hppso_coordinate_refresh_200k', port=8795,
        sampling=dict(**PARAMETERS, attraction_draws='per particle and coordinate',
                      personality_reinitialization=True, damping=False),
        description='All four traits are active. Stagnant particles redraw their traits after five iterations.',
        trial=trial, source=Path(__file__), dependencies=(Path(study.__file__),),
    ))
