"""T1 pilot with 20% success-history personality resets."""

from functools import partial
from pathlib import Path

from experiments import cec2011_hppso_success_history as study


class MostlyFreshHPPSO(study.SuccessHistoryHPPSO):
    uniform_fraction = 0.8


def trial(job, output, fingerprint):
    base = study.driver.base
    base.DEFAULT_ALGORITHMS['HPPSO'] = partial(study.run_hppso, optimizer=MostlyFreshHPPSO)
    return base.trial(job, output, fingerprint)


if __name__ == '__main__':
    study.driver.main(dict(
        title='T1 HPPSO with 20% success-history resets',
        folder='cec2011_hppso_success_history20_t1_200k', port=8783, problems={1},
        sampling=dict(memory_epochs=study.MEMORY_EPOCHS, uniform_fraction=MostlyFreshHPPSO.uniform_fraction,
                      success='strict personal-best improvement', weighting='gain within epoch'),
        description='At stagnation, 20% of resets use success history and 80% draw fresh random traits.',
        trial=trial, source=Path(__file__), dependencies=(Path(study.__file__),), report=study.report,
    ))
