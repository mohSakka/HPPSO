#!/bin/bash
#SBATCH --job-name=hppso-final
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/final-%A_%a.out
#SBATCH --error=logs/final-%A_%a.err

set -euo pipefail
# Submit from the repository root. Set partition, memory and time for your cluster.
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -m experiments.final_benchmark     --config experiments/configs/final_benchmark.json     --output final_results     --workers "${SLURM_CPUS_PER_TASK:-4}"     --shards 16 --shard "${SLURM_ARRAY_TASK_ID}"
