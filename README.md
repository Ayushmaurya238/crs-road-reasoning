
# Graph-Grounded Visual Reasoning for Autonomous Road Understanding (CRS: P0–P4)

**IIT Roorkee · Department of Data Science and Artificial Intelligence · Deep Learning Project**

[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--2B--Instruct-blue)]()
[![Dataset](https://img.shields.io/badge/Dataset-Argoverse2%20%2F%20OpenLane--V2%20%2F%20MAPVQA-green)]()
[![License](https://img.shields.io/badge/Eval-Zero--Leakage%20Log%20Splits-orange)]()

> **Core Question:** Can we strengthen a compact 2B-parameter Vision-Language Model for autonomous road reasoning by adding **(P3) a high-resolution CNN zoom specialist** for distant traffic lights/lane lines, or **(P4) explicit scene-graph serialization and two-path score fusion** for multi-hop relational logic?

Built on top of the **Combined Road Substrate (CRS)** framework (*Wild, Luo, Pavone — KTH, TRATON, Stanford, NVIDIA, 2026*), this repository contains our full training, evaluation, diagnostic analysis, and live/replay inference pipeline on 80 real Argoverse 2 driving logs.

---

## Key Highlights

* **Strongest Deployable Baseline (`P2`):** Attaching a continuous MLP box-regression head (`L1 + GIoU + kind + camera`) under a 3-stage curriculum reaches **76.9%** on `test_big` ($N=494$) and **+17.7 pp** over `P1` on reasoning-heavy questions in 3-fold CV ($p < 0.001$).
* **Highest Architectural Ceiling (`P4` Scene-Graph Conditioning):** Conditioning the VLM on accurate road scene graphs (`P4-oracle`) reaches **87.1%** Dev, **76.2%** Test, and **80.0%** `test_big` (**+3.1 pp** over `P2`, **+8.3 pp** over raw `P4`, $p = 5.3 \times 10^{-8}$), achieving **97–100%** on counting templates. Even without oracle graphs, `P4` achieves **0 / 717 parse failures**, and dev-tuned two-path fusion (`P4-fused`, $w^\ast = 0.5$) achieves the **best non-oracle reasoning-heavy accuracy (62.2% vs. 56.7% for `P2`)**.
* **Fine-Grained Localization Unlock (`P3` Specialist Zoom):** Snapping `BoxHead` predictions to full-resolution **YOLO11n** detections boosts traffic-light IoU by **≈16×–28×** (light-target IoU `0.013 → 0.359` Dev, `0.023 → 0.363` Test; Center-Hit `0% → 50%`; small n of 12 / 10 light questions), while `P3-oracle` lifts `test_big` small-attribute accuracy from **72.1% → 81.1%** (+9.0 pp), proving that localization—not recognition—is the bottleneck.
* **Verified Scientific Controls:**
  * **Visual Grounding (`P1-blank`):** Replacing images with blank frames drops accuracy by **−13.9 pp** on Dev ($p = 0.020$) and **−17.4 pp** in 3-fold CV ($p < 10^{-3}$).
  * **Compute-Matched Control (`P2-long`):** Training `P2` for the exact same +982 Stage-C steps as `P4` yields **76.5%** vs. **76.9%** ($p = 0.894$), confirming gains are architectural, not extra compute.

---

## Where P4 Stands (computed from `preds/*.jsonl`, Test-Big, N = 494)

`P4` does **not** beat `P2` on overall accuracy with its own self-written graphs (71.7% raw, 73.9% fused vs. 76.9%). The gap is local to counting questions:

| Question family | `P2` | `P4` | `P4-fused` | `P4-oracle` (GT graph) |
| :--- | :---: | :---: | :---: | :---: |
| Counting templates (N = 114) | 78.9 | 60.5 | 67.5 | **98.2** |
| All other templates (N = 380) | 76.3 | 75.0 | 75.8 | 74.5 |

Outside counting `P4` and `P2` are statistically tied (exact McNemar p = 0.47); with the true graph, counting reaches 98.2% (22 vs. 0 discordant pairs vs. `P2`, p = 4.8e-7). The 2B graph reader under-covers large scenes (lane-F1 0.446; it writes at most 9 lanes for scenes with up to 18), so improving graph reading is the main lever.

---

## Model Family (`P0` – `P4`)

All variants share `Qwen/Qwen3-VL-2B-Instruct` (`bf16`), `448px` input resolution, LoRA rank $r=16$, 4-way option-permutation TTA, and single-GPU training on an NVIDIA RTX 6000 Ada (48 GB; `0.207 s/sample` inference).

| Model | Description |
| :--- | :--- |
| **`P0`** | Zero-shot `Qwen3-VL-2B-Instruct` baseline |
| **`P1-blank`** | `P1` evaluated on blank frames (visual grounding control) |
| **`P1`** | LoRA fine-tuned VLM (`LM loss + option-ranking loss`) |
| **`P2`** | `P1` + continuous MLP `BoxHead` (`L1 + GIoU + kind + cam`) under 3-stage curriculum |
| **`P2-long`** | Compute-matched control: `P2` trained for +982 Stage-C steps (same compute as `P4`) |
| **`P3-raw`** | `P2` + ResNet-50 specialist hint cropped from raw un-snapped `BoxHead` |
| **`P3`** | `P2` + **YOLO11n box snapping** + native-resolution **ResNet-50 C0** specialist hint |
| **`P3-oracle`** | `P3` using ground-truth target boxes (localization upper bound) |
| **`P4`** | `P2` continued with mixed graph-writing ($\mathcal{L}_{\text{graph}}$) + graph-conditioned QA |
| **`P4-nograph`** | `P4` checkpoint answering directly without a graph in the prompt |
| **`P4-fused`** | Two-path score fusion $s = w \cdot s_{\text{direct}} + (1 - w) \cdot s_{\text{graph}}$ ($w^\ast = 0.5$ on Dev) |
| **`P4-oracle`** | `P4` answerer fed ground-truth scene graphs (graph-reasoning upper bound) |

---

## Quantitative Results

### 1. Main Benchmark: Dev ($N=101$), Test ($N=122$), and Test-Big ($N=494$)

| Model | Dev (%) | Test (%) | Test-Big (%) | TB Perc. (%) | TB Reas. (%) | TB Small (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `P0` (zero-shot) | 43.6 | 43.4 | 42.7 | 47.3 | 22.2 | 41.0 |
| `P1-blank` (no img) | 66.3 | — | — | — | — | — |
| `P1` (LoRA) | 80.2 | 69.7 | 75.3 | 78.7 | 60.0 | 75.4 |
| `P2` (grounded) | 81.2 | **73.8** | **76.9** | **81.4** | 56.7 | 72.1 |
| `P2-long` (ctrl) | 79.2 | **73.8** | 76.5 | 80.7 | 57.8 | **76.2** |
| `P3-raw` | **83.2** | 72.1 | 74.3 | 77.2 | 61.1 | 68.9 |
| `P3` (+YOLO snap) | **83.2** | 73.0 | 74.7 | 77.7 | 61.1 | 70.5 |
| `P4` (pred graph) | 74.3 | 64.8 | 71.7 | 74.0 | 61.1 | 64.8 |
| `P4-nograph` | 77.2 | 72.1 | 73.7 | 76.5 | 61.1 | 68.9 |
| `P4-fused` ($w^\ast=0.5$) | 77.2 | 68.9 | 73.9 | 76.5 | **62.2** | 66.4 |
| *`P3-oracle` (GT box)* | *83.2* | *73.8* | *77.3* | *80.9* | *61.1* | ***81.1*** |
| *`P4-oracle` (GT graph)* | ***87.1*** | ***76.2*** | ***80.0*** | ***84.2*** | *61.1* | *64.8* |

### 2. Independent 3-Fold Log-Level Cross-Validation ($N = 792$)

| Model | Total % (95% Log-Clustered CI) | Small Objects (%) | Reasoning (%) |
| :--- | :---: | :---: | :---: |
| `P0` (zero-shot) | 38.9 [35.0, 42.8] | 37.7 | 30.6 |
| `P1-blank` (no img) | 56.3 [52.4, 60.5] | 38.9 | 50.0 |
| `P1` (LoRA) | 73.7 [69.9, 77.6] | 72.6 | 54.8 |
| `P2` (+BoxHead) | **76.4** [71.8, 80.9] | 70.6 | **72.6** |
| `P3` (+YOLO Zoom) | 76.1 [71.6, 80.7] | 71.4 | 68.5 |
| *`P3-oracle` (GT box)* | *76.6 [71.9, 81.4]* | *73.0* | *68.5* |

---

## Repository Structure

```text
crs-road-reasoning/
├── README.md
├── LICENSE                       # MIT
├── demo_infer.py                 # Replay (no GPU) & live GPU inference demo
├── crs_road_v6/                  # Code for P0-P3: run.py, fetch_av2_frames.py, crs/*, tests, slurm/
├── scripts/                      # HPC Slurm scripts (job_all.sh = P0-P3 pipeline, job_p4.sh = P4 job)
├── data/
│   ├── stats.json, stats_test_big.json   # split statistics of the frozen evaluation sets
│   └── earlier_run/              # QA files from an EARLIER generation run (87 dev / 113 test questions);
│                                 # not the frozen evaluation sets (see "Included vs. not included")
├── preds/                        # Recorded 4-way TTA predictions: P0-P4 + controls on dev / test / test_big
├── results/
│   ├── cv_results/               # 3-fold CV: pooled_results.txt, sacct.txt, fold0-2 predictions and logs
│   ├── p4_logs/                  # McNemar comparison logs, fusion sweep, training logs (P4, P2-long)
│   ├── p4_inputs/                # per-template CSV, zoom stats, training summary, manifest (frozen-file hashes)
│   ├── runs_logs/                # per-stage training logs (P1, P2, P2b, P2-long, P3, P4)
│   └── sacct/                    # SLURM accounting (compute budget)
├── report/                       # report.md, tables.tex, figures fig1-fig10
├── overlays/                     # projected GT / YOLO11n camera overlays (.jpg)
└── docs/                         # final presentation (.pptx)
```

---

## Quick Start & Inference Demo

### 1. Install Dependencies
```bash
pip install -r crs_road_v6/requirements.txt
```

### 2. Replay Mode (no GPU, no weights)
Replays the recorded 4-way TTA outputs from `preds/` (graph, per-option log-probs, answers) and recomputes the two-path fusion from the stored log-probs. The image must belong to the chosen question, so pass the matching `--qid`:
```bash
python demo_infer.py --mode replay --preds preds --split test \
  --qid 05853f69-f948-3d04-8d64-d4e721c0e1a5__f14_8 \
  --image overlays/05853f69-f948-3d04-8d64-d4e721c0e1a5__f14_center.jpg \
  --w 0.5 --pace 0.7 --out demo_out.png
```
To replay a question where `P4-fused` is right and `P2` is wrong (no image bundled for that scene; the image panel stays blank unless you pass `--image`):
```bash
python demo_infer.py --mode replay --preds preds --split test_big \
  --qid big_23808d42-e4df-3a0d-b713-fe20e09a4f39__f07_33 --w 0.5 --pace 0.7
```
Question text and options are not printed in replay mode: the repository does not include the frozen evaluation question files (see below).

### 3. Live GPU Inference (`P0`-`P3`)
Needs a GPU, a checkpoint, a scenes folder and a matching QA file.
1. Download `p3_infer.pt` from this repository's **[GitHub Releases](../../releases)** page.
2. Build `data/scenes` for a log with `crs_road_v6/slurm/job_prep.sh` (or `run.py real`), and fetch frames:
   ```bash
   python crs_road_v6/fetch_av2_frames.py --log 098fe60e-bab0-32e2-89bc-bedced881911 --out av2_frames
   ```
3. Run:
   ```bash
   python demo_infer.py --mode live --code_dir crs_road_v6 --variant p3 --ckpt p3_infer.pt \
     --scenes data/scenes --qa data/earlier_run/qa_demo_live.jsonl \
     --qid 098fe60e-bab0-32e2-89bc-bedced881911__f04_7
   ```
Live `P4` inference is not supported from this repository (see below).

---

## Included vs. Not Included

| Item | Status |
| :--- | :--- |
| `P0`-`P3` code, tests, Slurm scripts | included (`crs_road_v6/`, `scripts/`) |
| Recorded predictions for all models, all splits | included (`preds/`) |
| Logs, statistics, McNemar comparisons, CV (3-fold) | included (`results/`) |
| `P4` training / inference / fusion code | **not included** - it ran on the HPC cluster (`crs_road_p4/`); `P4` numbers are reproducible from `preds/` and `results/p4_logs/` |
| Frozen evaluation question files (`qa_dev` 101, `qa_test` 122, `qa_test_big` 494) | **not included** - identified by `qid` in `preds/*.jsonl`; file hashes in `results/p4_inputs/manifest.json` |
| Trained weights | `p3_infer.pt` via GitHub Releases (`v1.0-weights`); `P4` checkpoint not released |

---

## Data Sources & References

* **CRS Paper & Interactive Annotator:** [kth-rpl.github.io/CRS/annotator](https://kth-rpl.github.io/CRS/annotator/) (*Bridging Structure and Language: Graph-Based Visual Reasoning for Autonomous Road Understanding*, Wild et al., 2026)
* **OpenLane-V2:** [github.com/OpenDriveLab/OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
* **Argoverse 2 Sensor Dataset:** [arxiv.org/abs/2301.00493](https://arxiv.org/abs/2301.00493)

---

## Team Contributions (IIT Roorkee · DSAI)

| Member | Role | Key Contributions |
| :--- | :--- | :--- |
| **Kritarth Singhal** | Core Extensions (`P3` & `P4`) & HPC Pipelines | Designed and implemented `P3` (YOLO11n box snapping, native-res cropping, 2-fold OOF ResNet-50 specialist) and `P4` (scene-graph serialization, mixed graph/QA training, two-path fusion); led HPC GPU runs, graph-reader diagnostics, and oracle ceiling analysis. |
| **Suryansh Rathore** | VLM Backbone, Grounding & Curriculum (`P1`–`P2`) | Fine-tuned Qwen3-VL-2B with LoRA ($r=16$), LM + option-ranking loss (`P1`), continuous MLP `BoxHead` (`P2`), 3-stage training curriculum (Stages A/B/C), error-weighted sampling, and `P2-long` compute control. |
| **Sahil Kaler** | Evaluation, Controls & Statistical Testing | Built the zero-leakage evaluation suite: 3-fold log-level CV ($N=792$), 494-question `test_big` split, 4-way TTA, blank-image visual grounding control, log-clustered bootstrap CIs, exact McNemar tests, and result plots. |
| **Ayush Maurya** | Data Prep, Literature & Demo Delivery | Downloaded and organized the 80 Argoverse 2 / OpenLane-V2 / MAPVQA logs, verified 50/15/15 splits, mapped CRS reference baselines, and prepared the inference demo video, overlays, GitHub repository (`README.md`), and final presentation. |
