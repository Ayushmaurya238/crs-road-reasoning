
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
* **Fine-Grained Localization Unlock (`P3` Specialist Zoom):** Snapping `BoxHead` predictions to full-resolution **YOLO11n** detections boosts traffic-light IoU by **15×–27×** (Test light IoU `0.023 → 0.363`, Center-Hit `0% → 50%`), while `P3-oracle` lifts `test_big` small-attribute accuracy from **72.1% → 81.1%** (+9.0 pp), proving that localization—not recognition—is the bottleneck.
* **Verified Scientific Controls:**
  * **Visual Grounding (`P1-blank`):** Replacing images with blank frames drops accuracy by **−13.9 pp** on Dev ($p = 0.020$) and **−17.4 pp** in 3-fold CV ($p < 10^{-3}$).
  * **Compute-Matched Control (`P2-long`):** Training `P2` for the exact same +982 Stage-C steps as `P4` yields **76.5%** vs. **76.9%** ($p = 0.894$), confirming gains are architectural, not extra compute.

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
├── README.md                     # Project overview, results, and usage guide
├── demo_infer.py                 # Replay & live GPU inference demo script
├── crs_road_v6/                  # Core codebase (run.py, fetch_av2_frames.py, crs/*)
│   ├── run.py
│   ├── requirements.txt
│   ├── fetch_av2_frames.py
│   └── crs/                      # vlm.py, cnn.py, det.py, graph.py, real.py, report.py
├── scripts/                      # HPC Slurm scripts (job_all.sh, job_p4.sh)
├── data/                         # Split metadata & QA files (stats.json, qa_dev/test.jsonl)
├── preds/                        # Recorded 4-way TTA predictions (P0–P4 on dev/test/test_big)
├── results/
│   ├── cv_results/               # 3-fold CV pooled_results.txt, sacct.txt, fold0–2 logs
│   ├── p4_logs/                  # Paired McNemar comparison logs & fusion sweep logs
│   └── p4_inputs/                # Per-template CSV, zoom stats, training summary
├── report/
│   ├── report.md                 # Full technical analysis report
│   ├── tables.tex                # LaTeX results tables
│   └── figures/                  # All 10 evaluation & diagnostic figures (fig1–fig10)
├── overlays/                     # Projected GT & YOLO11n camera overlay samples (.jpg)
└── docs/                         # Final presentation deck (.pptx)
```

---

## Quick Start & Inference Demo

### 1. Install Dependencies
```bash
pip install -r crs_road_v6/requirements.txt
```

### 2. Replay Mode (No GPU Required — Inspect `P2` vs. `P3` vs. `P4` & `P4-fused`)
Replays the recorded 4-way TTA outputs from `preds/` and recomputes two-path score fusion live:
```bash
python demo_infer.py \
  --mode replay \
  --preds preds \
  --split test_big \
  --pick fused_wins \
  --image overlays/05853f69-f948-3d04-8d64-d4e721c0e1a5__f14_center.jpg \
  --w 0.5 \
  --pace 0.7 \
  --out demo_out.png
```

### 3. Live GPU Inference (`P0`–`P3` with Trained Weights)
1. Download the lightweight inference checkpoint (`p3_infer.pt`) from this repository's **[GitHub Releases](../../releases)** page.
2. Fetch Argoverse 2 frames for a test log:
   ```bash
   python crs_road_v6/fetch_av2_frames.py --log 098fe60e-bab0-32e2-89bc-bedced881911 --out av2_frames
   ```
3. Run live inference with 4-way TTA and YOLO11n detection:
   ```bash
   python demo_infer.py \
     --mode live \
     --code_dir crs_road_v6 \
     --variant p3 \
     --ckpt p3_infer.pt \
     --scenes data/scenes \
     --qa data/qa_test.jsonl \
     --qid 098fe60e-bab0-32e2-89bc-bedced881911__f04_7
   ```

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

