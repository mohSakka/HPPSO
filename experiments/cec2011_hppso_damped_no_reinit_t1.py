"""T1 pilot combining exponential damping with fixed personalities."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_exponential as damping
from experiments import cec2011_hppso_no_reinit_t1 as fixed
from experiments import cec2011_hppso_success_history as study


class DampedFixedPersonalityHPPSO(damping.ExponentialHPPSO, fixed.FixedPersonalityHPPSO):
    """Apply the evaluation-based schedule to fixed initial personalities."""


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=DampedFixedPersonalityHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='T1 HPPSO with fixed personalities and exponential damping',
        folder='cec2011_hppso_damped_no_reinit_t1_200k', port=8785, problems={1},
        sampling=dict(personality_reinitialization=False, terminal_factor=damping.TERMINAL_FACTOR,
                      method='exponential decay by post-initialization evaluation count'),
        description='Curiosity alpha and the aggressiveness contribution decay while initial personalities are retained.',
        trial=trial, source=Path(__file__),
        dependencies=tuple(Path(module.__file__) for module in (damping, fixed, study)),
    ))
