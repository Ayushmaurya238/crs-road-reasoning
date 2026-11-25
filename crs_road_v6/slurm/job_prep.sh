#!/bin/bash
#SBATCH --job-name=crs_prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=crs_prep_%j.out
#SBATCH --error=crs_prep_%j.err
# CPU-only: downloads the 80 logs + AV2 frames once (skips what already exists).
set -eo pipefail
set +u; source ~/crs_venv/bin/activate; set -u
export PYTHONUNBUFFERED=1 CRS_INSECURE=1
cd ~/crs_project/crs_road_v6
LOGS=~/crs_project/CRS_Logs; FR=~/crs_project/av2_frames
python download_logs.py --out "$LOGS"; python download_logs.py --out "$LOGS"
python - "$LOGS" "$FR" <<'PY'
import glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
logs_dir, fr_dir = sys.argv[1], sys.argv[2]
ids = sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{logs_dir}/annotations/*.json"))
def go(lid):
    if os.path.exists(f"{fr_dir}/{lid}/frames.json"): return lid, True
    for _ in range(3):
        r = subprocess.run(["python","fetch_av2_frames.py","--log",lid,"--meta",f"{logs_dir}/metadata","--out",fr_dir,"--workers","8"],capture_output=True,text=True)
        if r.returncode == 0 and os.path.exists(f"{fr_dir}/{lid}/frames.json"): return lid, True
        time.sleep(3)
    return lid, False
with ThreadPoolExecutor(4) as ex: res = list(ex.map(go, ids))
ok = sum(g for _, g in res); print(f"Frames ready: {ok}/{len(ids)}")
if ok < len(ids): print("MISSING:", [l[:8] for l, g in res if not g])
if ok < 75: raise SystemExit("too many failed logs")
PY
echo PREP DONE
