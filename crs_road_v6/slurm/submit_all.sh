#!/bin/bash
# Run on the login node (only submits jobs; nothing heavy).
cd ~/crs_project/crs_road_v6
P=$(sbatch --parsable slurm/job_prep.sh)
echo "prep job $P"
for F in 0 1 2 3; do
  # afterany: a failed fold doesn't block the next; resubmit that fold alone later
  if [ $F -eq 0 ]; then DEP="afterok:$P"; else DEP="afterany:$PREV"; fi
  PREV=$(sbatch --parsable --dependency=$DEP --export=ALL,FOLD=$F slurm/job_cv.sh)
  echo "fold $F -> job $PREV (after $DEP)"
done
echo "check with: squeue -u kritarth1"
