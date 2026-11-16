"""T1 pilot using the HPPSO update with all four trait effects disabled."""

from functools import partial
from pathlib import Path

import numpy as np

from experiments import cec2011_hppso_no_openness_t1 as control
from experiments import cec2011_hppso_success_history as study


class NoTraitsHPPSO(control.NoOpennessHPPSO):
    def _start_epoch(self):
        self.alpha_curio = 0.0

    def _effective_aggressiveness(self):
        return np.zeros(self.n_pop)


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=NoTraitsHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    curiosity = control.control.control
    dependencies = (control, control.control, curiosity, curiosity.combined,
                    curiosity.combined.damping, curiosity.combined.fixed, study)
    study.driver.main(dict(
        title='T1 HPPSO with all traits disabled',
        folder='cec2011_hppso_no_traits_t1_200k', port=8789, problems={1},
        sampling=dict(personality_reinitialization=False, conformity=False, openness=False,
                      curiosity=False, aggressiveness=False, inertia=0.4, attraction_weights=[1.0, 1.0]),
        description='All personality effects are disabled. The inertia is fixed at the HPPSO floor of 0.4.',
        trial=trial, source=Path(__file__),
        dependencies=tuple(Path(module.__file__) for module in dependencies),
    ))
