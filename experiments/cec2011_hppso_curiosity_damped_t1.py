"""T1 pilot with fixed personalities and curiosity-only damping."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_damped_no_reinit_t1 as combined
from experiments import cec2011_hppso_success_history as study


class CuriosityDampedHPPSO(combined.DampedFixedPersonalityHPPSO):
    def _effective_aggressiveness(self):
        return self.personality[:, 2]


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=CuriosityDampedHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='T1 HPPSO with fixed personalities and curiosity-only damping',
        folder='cec2011_hppso_curiosity_damped_t1_200k', port=8786, problems={1},
        sampling=dict(personality_reinitialization=False, aggressiveness_damping=False,
                      curiosity_terminal_factor=combined.damping.TERMINAL_FACTOR,
                      method='exponential decay by post-initialization evaluation count'),
        description='Only the curiosity mutation factor decays. Initial personalities and aggressiveness are fixed.',
        trial=trial, source=Path(__file__),
        dependencies=tuple(Path(module.__file__) for module in
                           (combined, combined.damping, combined.fixed, study)),
    ))
