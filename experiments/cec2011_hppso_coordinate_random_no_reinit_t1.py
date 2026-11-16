"""T1 ablation of personality resets with per-coordinate attraction draws."""

from pathlib import Path

from experiments import cec2011_hppso_no_reinit_t1 as fixed


if __name__ == '__main__':
    fixed.study.driver.main(dict(
        title='T1 HPPSO with per-coordinate draws and fixed personalities',
        folder='cec2011_hppso_coordinate_random_no_reinit_t1_200k', port=8791, problems={1},
        sampling=dict(attraction_draws='per particle and coordinate', personality_reinitialization=False),
        description='All four traits use their default settings, and each particle retains its initial personality.',
        trial=fixed.trial, source=Path(__file__),
        dependencies=(Path(fixed.__file__), Path(fixed.study.__file__)),
    ))
