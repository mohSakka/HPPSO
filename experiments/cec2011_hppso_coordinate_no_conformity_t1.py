"""T1 conformity ablation with per-coordinate draws and standard resets."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_success_history as study
from hppso.algorithms.hppso import HPPSO


class NoConformityHPPSO(HPPSO):
    def _attraction_weights(self):
        return 1.0, 1.0


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=NoConformityHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='T1 HPPSO with per-coordinate draws and no conformity',
        folder='cec2011_hppso_coordinate_no_conformity_t1_200k', port=8792, problems={1},
        sampling=dict(attraction_draws='per particle and coordinate', personality_reinitialization=True,
                      conformity=False, attraction_weights=[1.0, 1.0], damping=False),
        description='Conformity weighting is disabled, and standard personality reinitialization is active.',
        trial=trial, source=Path(__file__), dependencies=(Path(study.__file__),),
    ))
