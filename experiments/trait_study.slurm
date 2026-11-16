#!/bin/bash
#SBATCH --job-name=hppso-traits
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/traits-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m experiments.trait_study \
  --config experiments/configs/trait_study.json \
  --output trait_results --shards 6 --shard "${SLURM_ARRAY_TASK_ID}" \
  --workers "${SLURM_CPUS_PER_TASK}"
# After ALL array elements finish:
# python -m experiments.trait_report --output trait_results --sweep-curves
