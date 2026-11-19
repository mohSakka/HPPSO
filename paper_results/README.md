# Revised paper data and reproduction

This directory contains the data used in the revised HPPSO manuscript. It identifies the experiment settings and provides the tabular inputs for checking the reported results. Full optimizer checkpoints, weights, and some evaluation traces are larger companion inputs.

| Directory | Contents |
| --- | --- |
| [Mathematical benchmarks](mathematical/) | 21,600 final scores, 120 exact problem definitions, experiment plan, and revised Wilcoxon, Friedman, and Nemenyi analyses |
| [CEC2011](cec2011/) | 1,980 final scores, experiment plan, and revised numerical and statistical results |
| [Neural networks](neural_networks/) | Per-run predictive metrics, settings, and saved split indices for 20 datasets |
| [Ablations](ablation/) | Settings, benchmark definitions, and results for the final component and sensitivity experiments |
| [Diversity and resources](diversity_and_resources/) | Spatial and behavioral diversity trajectories and matched runtime and memory measurements |
| [Openness coverage in two dimensions](openness_coverage_2d/) | Coverage estimates by function, peer rule, and seed |

## Mathematical benchmarks

Use [the server protocol](../docs/final_benchmark.md) to run the current benchmark. To regenerate statistical tables from the saved scores without rerunning optimization, run the following from the repository root:

~~~bash
uv run python -m experiments.build_coordinatewise_manuscript_results --source paper_results/mathematical --output experiments/results/rebuilt_mathematical --skip-figures
~~~

The full trial archive is required to regenerate convergence plots at their original resolution. The revised archive retained 19,200 comparator trials and replaced the 2,400 HPPSO trials with runs using coordinate-wise velocity coefficients. The exact shifts, bounds, rotations, and compositions are in `mathematical/definitions/`.

## CEC2011

The final export combines seeds 0–2 from `cec2011_pop200_fe200k` (comparators) and `cec2011_hppso_coordinate_refresh_200k` (HPPSO) with seeds 3–9 from `cec2011_coordinate_refresh_10_seeds`. The [seed extension driver](../experiments/cec2011_add_seeds.py) expects those paths under `experiments/results/`. The saved final scores suffice for statistical recomputation. The [trial-level audit script](../experiments/build_cec2011_manuscript_results.py) additionally requires the original trial archives and regenerates its manuscript tables and plots.

## Neural networks

Use the `baselines/` tree for the comparator runs and the `hppso/` tree for the HPPSO runs reported in the paper. Both trees contain the archived configurations and split indices. The combined `runs.csv` gives the 2,200 final per-run metrics and omits the superseded HPPSO baseline runs. The [training runner](../src/hppso/experiments/train_neural_network.py) accepts the optimizer, seed, budget, population, and sample cap. The dataset columns and settings identify the individual runs.

## Component studies

The saved ablation plans specify the algorithms, matched seeds, and optimizer settings. Use the [trait study runner](../experiments/trait_study.py) and [reporting script](../experiments/trait_report.py) with the corresponding experiment configuration in [experiments/configs](../experiments/configs/). Regenerating the detailed trajectory figures requires the associated trial data. [The experiment guides](../docs/) describe the recorded metrics and trial formats.
