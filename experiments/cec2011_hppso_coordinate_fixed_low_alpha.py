"""CEC2011 rerun with fixed personalities and a lower curiosity mutation factor."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_no_reinit_t1 as fixed


ALPHA_MUT = 0.001


class FixedLowAlphaHPPSO(fixed.FixedPersonalityHPPSO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, alpha_curio=ALPHA_MUT)


def trial(job, output, fingerprint):
    study = fixed.study
    study.driver.base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=FixedLowAlphaHPPSO)
    return study.driver.base.trial(job, output, fingerprint)


if __name__ == '__main__':
    fixed.study.driver.main(dict(
        title='HPPSO with fixed personalities and reduced curiosity alpha',
        folder='cec2011_hppso_coordinate_fixed_low_alpha_200k', port=8794,
        sampling=dict(attraction_draws='per particle and coordinate', personality_reinitialization=False,
                      alpha_mut=ALPHA_MUT, damping=False),
        description='All four traits are active. Each particle retains its initial personality, '
                    'and the curiosity mutation factor stays at 0.001.',
        trial=trial, source=Path(__file__),
        dependencies=(Path(fixed.__file__), Path(fixed.study.__file__)),
    ))
