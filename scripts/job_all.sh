#!/bin/bash
#SBATCH --job-name=crs_all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --output=crs_all_%j.out
#SBATCH --error=crs_all_%j.err

set -eo pipefail
START_TS=$(date +%s)
set +u; source ~/crs_venv/bin/activate; set -u
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CRS_INSECURE=1

cd ~/crs_project/crs_road
echo "Host: $(hostname) | Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LOGS=~/crs_project/CRS_Logs
FR=~/crs_project/av2_frames
DATA=~/crs_project/data
CK=~/crs_project/ck
RUNS=~/crs_project/runs
PREDS=~/crs_project/preds
mkdir -p "$CK" "$RUNS" "$PREDS"

RES() { if [ -f "$1/last.pt" ]; then echo "--resume $1/last.pt"; else echo "${2:-}"; fi; }

echo "=== [1/7] Download 80 logs ==="
python download_logs.py --out "$LOGS"
python download_logs.py --out "$LOGS"

echo "=== [2/7] Download AV2 frames ==="
python - "$LOGS" "$FR" <<'PY'
import glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
logs_dir, fr_dir = sys.argv[1], sys.argv[2]
ids = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{logs_dir}/annotations/*.json"))
def go(lid):
    if os.path.exists(f"{fr_dir}/{lid}/frames.json"): return lid, True, "skip"
    for _ in range(2):
        r = subprocess.run(["python", "fetch_av2_frames.py", "--log", lid, "--meta", f"{logs_dir}/metadata", "--out", fr_dir, "--workers", "8"], capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(f"{fr_dir}/{lid}/frames.json"): return lid, True, "ok"
        time.sleep(3)
    return lid, False, "FAILED"
ok = 0
with ThreadPoolExecutor(4) as ex:
    for lid, good, msg in ex.map(go, ids):
        ok += int(good); print(lid[:8], msg, flush=True)
print(f"Frames ready: {ok}/{len(ids)}", flush=True)
if ok < min(75, len(ids)): raise SystemExit(f"Too many failed frames ({ok}/{len(ids)})")
PY
echo "=== [3/7] Convert real graphs + boxes ==="
if [ ! -f "$DATA/qa_train.jsonl" ]; then
    python run.py real --graphs "$LOGS/annotations" --meta "$LOGS/metadata" --frames "$FR" --out "$DATA" --max_per_log 300 --split 50,15,15
    python run.py check --qa "$DATA"
fi

echo "=== [4/7] YOLO11n light boxes ==="
if [ ! -f "$DATA/.detect_done" ]; then
    python run.py detect --scenes "$DATA/scenes" --model yolo11n.pt --bs 16
    python check_real.py --scenes "$DATA/scenes" --out ~/crs_project/overlays --n 5
    touch "$DATA/.detect_done"
fi

echo "=== [5/7] Crop cache + 2-fold ResNet-50 C0 ==="
[ -f "$DATA/crops/index.jsonl" ] || python run.py cropcache --scenes "$DATA/scenes" --out "$DATA/crops"
[ -f "$CK/c0_f0.pt" ] || python run.py cnn --arch resnet50 --cache "$DATA/crops" --fold 0 --nfolds 2 --epochs 8 --bs 64 --out "$CK/c0_f0.pt"
[ -f "$CK/c0_f1.pt" ] || python run.py cnn --arch resnet50 --cache "$DATA/crops" --fold 1 --nfolds 2 --epochs 8 --bs 64 --out "$CK/c0_f1.pt"

echo "=== [6/7] Probe speed & train P1 / P2 / P3 ==="
BATCH=16
VLM="--dtype bf16 --img_side 448 --lora_r 16 --cnn_arch resnet50"
[ -f "$RUNS/probe/probe.json" ] || python run.py train --variant p2 --probe --scenes "$DATA/scenes" --qa_train "$DATA/qa_train.jsonl" --out "$RUNS/probe" $VLM

if [ ! -f "$RUNS/steps.txt" ]; then
    python - "$RUNS/probe/probe.json" "$DATA/qa_train.jsonl" "$BATCH" "$START_TS" <<'PY' > "$RUNS/steps.txt"
import json, math, sys, time
sec = json.load(open(sys.argv[1]))["sec_per_sample"]
n_tr = sum(1 for _ in open(sys.argv[2]))
bs, start_ts = int(sys.argv[3]), int(sys.argv[4])
rem = min(12.0, max(3.0, 17.5 - (time.time() - start_ts) / 3600.0))
steps = max(300, min(math.ceil(3.0 * n_tr / bs), int((rem * 3600) / (2.5 * sec * bs))))
sys.stderr.write(f"probe={sec:.3f}s/sample | n_train={n_tr} | STEPS={steps} (~{2.5*steps*bs*sec/3600:.1f}h)\n")
print(steps)
PY
fi
STEPS=$(cat "$RUNS/steps.txt")
EV_STEP=$(( STEPS / 6 )); [ "$EV_STEP" -lt 50 ] && EV_STEP=50
TR="--scenes $DATA/scenes --qa_train $DATA/qa_train.jsonl --qa_dev $DATA/qa_dev.jsonl --steps $STEPS --batch $BATCH --val_n 60 --eval_every $EV_STEP --ckpt_every 100 $VLM"

[ -f "$RUNS/p1/final.pt" ]  || python run.py train --variant p1 $(RES "$RUNS/p1") $TR --out "$RUNS/p1"
[ -f "$RUNS/p2/prefix.pt" ] || python run.py train --variant p2 --prefix_only $(RES "$RUNS/p2") $TR --out "$RUNS/p2"
[ -f "$RUNS/p2b/final.pt" ] || python run.py train --variant p2 $(RES "$RUNS/p2b" "--resume $RUNS/p2/prefix.pt") $TR --out "$RUNS/p2b"
[ -f "$RUNS/p3/final.pt" ]  || python run.py train --variant p3 $(RES "$RUNS/p3" "--resume $RUNS/p2/prefix.pt") --cnn_ckpts "$CK/c0_f0.pt" "$CK/c0_f1.pt" --gt_box_prob 0.3 --use_det $TR --out "$RUNS/p3"

echo "=== [7/7] Eval (dev & test, 4-way TTA) + Compare ==="
for SPLIT in dev test; do
    EV="--scenes $DATA/scenes --qa_eval $DATA/qa_${SPLIT}.jsonl $VLM --tta --tta_n 4"
    [ -f "$PREDS/p0_${SPLIT}.jsonl" ]        || python run.py eval --variant p0 $EV --out "$PREDS/p0_${SPLIT}.jsonl"
    [ -f "$PREDS/p1_${SPLIT}.jsonl" ]        || python run.py eval --variant p1 --ckpt "$RUNS/p1/final.pt" $EV --out "$PREDS/p1_${SPLIT}.jsonl"
    [ -f "$PREDS/p2_${SPLIT}.jsonl" ]        || python run.py eval --variant p2 --ckpt "$RUNS/p2b/final.pt" $EV --out "$PREDS/p2_${SPLIT}.jsonl"
    [ -f "$PREDS/p3_raw_${SPLIT}.jsonl" ]    || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --out "$PREDS/p3_raw_${SPLIT}.jsonl"
    [ -f "$PREDS/p3_det_${SPLIT}.jsonl" ]    || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --use_det --out "$PREDS/p3_det_${SPLIT}.jsonl"
    [ -f "$PREDS/p3_oracle_${SPLIT}.jsonl" ] || python run.py eval --variant p3 --ckpt "$RUNS/p3/final.pt" $EV --oracle_box --out "$PREDS/p3_oracle_${SPLIT}.jsonl"
    echo "=== COMPARE ($SPLIT) ==="
    python run.py compare --runs P0="$PREDS/p0_${SPLIT}.jsonl" P1="$PREDS/p1_${SPLIT}.jsonl" P2="$PREDS/p2_${SPLIT}.jsonl" P3-raw="$PREDS/p3_raw_${SPLIT}.jsonl" P3="$PREDS/p3_det_${SPLIT}.jsonl" P3-oracle="$PREDS/p3_oracle_${SPLIT}.jsonl" --plot "$PREDS/p3_${SPLIT}_graph.png"
    python run.py report --preds "$PREDS/p3_det_${SPLIT}.jsonl" --vs "$PREDS/p1_${SPLIT}.jsonl"
done

[ -f "$PREDS/p1_blank_dev.jsonl" ] || python run.py eval --variant p1 --ckpt "$RUNS/p1/final.pt" --scenes "$DATA/scenes" --qa_eval "$DATA/qa_dev.jsonl" $VLM --blank --out "$PREDS/p1_blank_dev.jsonl"
python run.py report --preds "$PREDS/p1_blank_dev.jsonl" --vs "$PREDS/p1_dev.jsonl"
echo "=== ALL DONE: $(date) ==="
