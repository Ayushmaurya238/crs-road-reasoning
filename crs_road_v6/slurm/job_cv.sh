#!/bin/bash
#SBATCH --job-name=crs_cv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=23:50:00
#SBATCH --output=crs_cv_f%a_%j.out
#SBATCH --error=crs_cv_f%a_%j.err
# One CV fold per job. Usage: sbatch --export=ALL,FOLD=0 slurm/job_cv.sh
# Resumable: just resubmit the same fold; finished stages are skipped.
set -eo pipefail
FOLD=${FOLD:?set FOLD=0..3}; NFOLDS=${NFOLDS:-4}
EPT=${EVAL_PER_TEMPLATE:-2}      # test questions per template per log
TTA_N=${TTA_N:-4}
set +u; source ~/crs_venv/bin/activate; set -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CRS_INSECURE=1
cd ~/crs_project/crs_road_v6
echo "Host: $(hostname) fold=$FOLD Start: $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
LOGS=~/crs_project/CRS_Logs; FR=~/crs_project/av2_frames
ROOT=~/crs_project/cv/fold$FOLD
DATA=$ROOT/data; CK=$ROOT/ck; RUNS=$ROOT/runs; PREDS=$ROOT/preds
mkdir -p "$CK" "$RUNS" "$PREDS"
RES() { if [ -f "$1/last.pt" ]; then echo "--resume $1/last.pt"; else echo "${2:-}"; fi; }

echo "=== [1] Fold data (split by LOG, so no log is in train and test) ==="
if [ ! -f "$DATA/qa_train.jsonl" ]; then
  python run.py real --graphs "$LOGS/annotations" --meta "$LOGS/metadata" --frames "$FR" --out "$DATA" \
     --max_per_log 300 --cv_fold $FOLD --cv_folds $NFOLDS --n_dev 8 --eval_per_template $EPT | tee "$ROOT/split.log"
  python run.py check --qa "$DATA"
fi
echo "=== [2] YOLO boxes ==="
if [ ! -f "$DATA/.detect_done" ]; then python run.py detect --scenes "$DATA/scenes" --model yolo11n.pt --bs 16; touch "$DATA/.detect_done"; fi
echo "=== [3] Crop cache + ResNet C0 (trained on this fold's train logs only) ==="
[ -f "$DATA/crops/index.jsonl" ] || python run.py cropcache --scenes "$DATA/scenes" --out "$DATA/crops"
for f in 0 1; do [ -f "$CK/c0_f$f.pt" ] || python run.py cnn --arch resnet50 --cache "$DATA/crops" --fold $f --nfolds 2 --epochs 8 --bs 64 --out "$CK/c0_f$f.pt"; done

echo "=== [4] Train P1 / P2 / P3 (same recipe as job 10229) ==="
BATCH=16
VLM="--dtype bf16 --img_side 448 --lora_r 16 --cnn_arch resnet50"
NTR=$(wc -l < "$DATA/qa_train.jsonl")
STEPS=${STEPS:-$(python -c "import math;print(max(300,math.ceil(3.0*$NTR/$BATCH)))")}
EV_STEP=$(( STEPS / 6 )); [ "$EV_STEP" -lt 50 ] && EV_STEP=50
echo "n_train=$NTR STEPS=$STEPS"
TR="--scenes $DATA/scenes --qa_train $DATA/qa_train.jsonl --qa_dev $DATA/qa_dev.jsonl --steps $STEPS --batch $BATCH --val_n 60 --eval_every $EV_STEP --ckpt_every 100 $VLM"
[ -f "$RUNS/p1/final.pt" ]  || python run.py train --variant p1 $(RES "$RUNS/p1") $TR --out "$RUNS/p1"
[ -f "$RUNS/p2/prefix.pt" ] || python run.py train --variant p2 --prefix_only $(RES "$RUNS/p2") $TR --out "$RUNS/p2"
[ -f "$RUNS/p2b/final.pt" ] || python run.py train --variant p2 $(RES "$RUNS/p2b" "--resume $RUNS/p2/prefix.pt") $TR --out "$RUNS/p2b"
[ -f "$RUNS/p3/final.pt" ]  || python run.py train --variant p3 $(RES "$RUNS/p3" "--resume $RUNS/p2/prefix.pt") --cnn_ckpts "$CK/c0_f0.pt" "$CK/c0_f1.pt" --gt_box_prob 0.3 --use_det $TR --out "$RUNS/p3"

echo "=== [5] Eval on TEST only ($TTA_N-way TTA) ==="
S=test
EV="--scenes $DATA/scenes --qa_eval $DATA/qa_${S}.jsonl $VLM --tta --tta_n $TTA_N"
[ -f "$PREDS/p0_$S.jsonl" ]       || python run.py eval --variant p0 $EV --out "$PREDS/p0_$S.jsonl"
[ -f "$PREDS/p1_$S.jsonl" ]       || python run.py eval --variant p1 --ckpt "$RUNS/p1/final.pt" $EV --out "$PREDS/p1_$S.jsonl"
[ -f "$PREDS/p2_$S.jsonl" ]       || python run.py eval --variant p2 --ckpt "$RUNS/p2b/final.pt" $EV --out "$PREDS/p2_$S.jsonl"
[ -f "$PREDS/p3_raw_$S.jsonl" ]   || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --out "$PREDS/p3_raw_$S.jsonl"
[ -f "$PREDS/p3_det_$S.jsonl" ]   || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --use_det --out "$PREDS/p3_det_$S.jsonl"
[ -f "$PREDS/p3_oracle_$S.jsonl" ] || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --oracle_box --out "$PREDS/p3_oracle_$S.jsonl"
# blank-image control for P1 (test): shows the model really uses the images
[ -f "$PREDS/p1_blank_$S.jsonl" ] || python run.py eval --variant p1 --ckpt "$RUNS/p1/final.pt" --scenes "$DATA/scenes" --qa_eval "$DATA/qa_${S}.jsonl" $VLM --blank --out "$PREDS/p1_blank_$S.jsonl"

# free disk (220 GB quota): keep final.pt, drop resume/optimizer files
find "$RUNS" -name 'last.pt' -o -name 'last_step.pt' -o -name 'best.pt' | xargs -r rm -f
echo "=== FOLD $FOLD DONE: $(date) ==="
