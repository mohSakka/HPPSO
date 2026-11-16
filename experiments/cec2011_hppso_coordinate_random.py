"""Rerun default HPPSO on CEC2011 with per-coordinate attraction draws."""

from pathlib import Path

from experiments import cec2011_hppso_coordinate_random_t1 as pilot


if __name__ == '__main__':
    pilot.driver.main(dict(
        title='Default HPPSO with per-coordinate attraction draws',
        folder='cec2011_hppso_coordinate_random_200k', port=8793,
        sampling=dict(attraction_draws='per particle and coordinate', personality_reinitialization=True,
                      all_traits_enabled=True, damping=False),
        description='All four traits and personality reinitialization use the default settings. '
                    'Both attraction coefficients are drawn independently per particle and coordinate.',
        trial=pilot.trial, source=Path(__file__),
        dependencies=(Path(pilot.__file__), pilot.driver.base.ROOT / 'src/hppso/experiments/runners.py'),
    ))
