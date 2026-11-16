"""T1 pilot with HPPSO personality reinitialization disabled."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_success_history as study


class FixedPersonalityHPPSO(study.HPPSO):
    def _refresh_personalities(self, improved, previous_best, reinit):
        """Keep each particle's initial personality throughout the run."""


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=FixedPersonalityHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='T1 HPPSO with fixed personalities',
        folder='cec2011_hppso_no_reinit_t1_200k', port=8784, problems={1},
        sampling=dict(personality_reinitialization=False),
        description='Each particle keeps its initial personality throughout the run.',
        trial=trial, source=Path(__file__), dependencies=(Path(study.__file__),), report=study.report,
    ))
