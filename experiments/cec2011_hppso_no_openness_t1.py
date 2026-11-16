"""T1 ablation of openness on the no-conformity configuration."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_no_conformity_t1 as control


class NoOpennessHPPSO(control.NoConformityHPPSO):
    def __init__(self, *args, **kwargs):
        # Openness activates only when the trait exceeds this threshold.
        super().__init__(*args, **kwargs, socialism_threshold=1.0)


def trial(job, output, fingerprint):
    study = control.control.study
    study.driver.base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=NoOpennessHPPSO)
    return study.driver.base.trial(job, output, fingerprint)


if __name__ == '__main__':
    curiosity = control.control
    dependencies = (control, curiosity, curiosity.combined, curiosity.combined.damping,
                    curiosity.combined.fixed, curiosity.study)
    curiosity.study.driver.main(dict(
        title='T1 HPPSO without conformity or openness',
        folder='cec2011_hppso_no_openness_t1_200k', port=8788, problems={1},
        sampling=dict(personality_reinitialization=False, aggressiveness_damping=False,
                      curiosity_terminal_factor=curiosity.combined.damping.TERMINAL_FACTOR,
                      conformity=False, openness=False, theta=1.0, attraction_weights=[1.0, 1.0]),
        description='Fixed personalities and curiosity-only damping, with conformity and openness disabled.',
        trial=trial, source=Path(__file__),
        dependencies=tuple(Path(module.__file__) for module in dependencies),
    ))
