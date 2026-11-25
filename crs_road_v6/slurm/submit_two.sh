#!/bin/bash
# Runs folds 0 and 1 of the 4-fold split (each fold: ~52 train logs, 20 test logs).
# Add folds 2 and 3 later with: sbatch --export=ALL,FOLD=2 slurm/job_cv.sh
cd ~/crs_project/crs_road_v6
P=$(sbatch --parsable slurm/job_prep.sh); echo "prep job $P"
F0=$(sbatch --parsable --dependency=afterok:$P --export=ALL,FOLD=0 slurm/job_cv.sh); echo "fold 0 -> $F0"
F1=$(sbatch --parsable --dependency=afterany:$F0 --export=ALL,FOLD=1 slurm/job_cv.sh); echo "fold 1 -> $F1"
echo "check with: squeue -u kritarth1"
