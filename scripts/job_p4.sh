#!/bin/bash
#SBATCH --job-name=crs_p4_big
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=crs_p4_%j.out
#SBATCH --error=crs_p4_%j.err

set -eo pipefail
set +u; source ~/crs_venv/bin/activate; set -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

cd ~/crs_project/crs_road_p4
echo "Host: $(hostname) | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Runs P4 + P2-long + dev/test/test_big (~600 Qs) 4-way TTA evals + fusion + compare
# Optional: add ZOOM=1 (+2h) or SEEDS="0 1" (+2.5h), or BIG_TEST=0 to skip test_big
BIG_TEST=1 TTA_N=4 bash p4_run.sh

echo "=== JOB P4 DONE: $(date) ==="
