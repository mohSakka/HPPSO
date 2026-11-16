"""T1 conformity ablation with fixed personalities and curiosity damping."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_curiosity_damped_t1 as control


class NoConformityHPPSO(control.CuriosityDampedHPPSO):
    def _attraction_weights(self):
        return 1.0, 1.0


def trial(job, output, fingerprint):
    base = control.study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(control.study.run_hppso, optimizer=NoConformityHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    dependencies = (control, control.combined, control.combined.damping,
                    control.combined.fixed, control.study)
    control.study.driver.main(dict(
        title='T1 HPPSO without conformity',
        folder='cec2011_hppso_no_conformity_t1_200k', port=8787, problems={1},
        sampling=dict(personality_reinitialization=False, aggressiveness_damping=False,
                      curiosity_terminal_factor=control.combined.damping.TERMINAL_FACTOR,
                      conformity=False, attraction_weights=[1.0, 1.0]),
        description='Fixed personalities and curiosity-only damping, with standard PSO attraction coefficients.',
        trial=trial, source=Path(__file__),
        dependencies=tuple(Path(module.__file__) for module in dependencies),
    ))
