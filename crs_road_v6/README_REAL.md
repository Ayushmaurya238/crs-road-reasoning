# Real-data run (MAPVQA graphs + metadata boxes + YOLO snap, P0/P1/P2/P3)

1. Download graphs + metadata:
   python download_logs.py --out CRS_Logs [--limit N]
2. Download exact camera frames using metadata timestamps:
   python fetch_av2_frames.py --log <logid> --meta CRS_Logs/metadata --out av2_frames
3. Convert graphs + metadata boxes (with singleton light/crossing descriptors, occlusion-aware 3m lane-line boxes, and <box> CoT):
   python run.py real --graphs CRS_Logs/annotations --meta CRS_Logs/metadata --frames av2_frames --out data
4. (Recommended for slow GPUs / high light IoU) Cache full-res YOLO11n traffic-light boxes:
   python run.py detect --scenes data/scenes --model yolo11n.pt
5. Build real traffic crop cache & fine-tune ResNet-50 last layers (layer3 + layer4):
   python run.py cropcache --scenes data/scenes --out data/crops
   python run.py cnn --arch resnet50 --cache data/crops --fold 0 --nfolds 2 --epochs 6 --out ck/c0_f0.pt
   python run.py cnn --arch resnet50 --cache data/crops --fold 1 --nfolds 2 --epochs 6 --out ck/c0_f1.pt
6. Train P1, P2 prefix, P2 branch, P3 branch (pass --use_det to enable YOLO light snap):
   see crs_dry_run_kaggle.ipynb and crs_full_run_campus.ipynb.
7. Compare P0/P1/P2/P3 (and raw vs snapped P3 ablation) + comparison chart:
   python run.py compare --runs P0=preds/p0.jsonl P1=preds/p1.jsonl P2=preds/p2.jsonl P3-raw=preds/p3_raw.jsonl P3=preds/p3.jsonl P3-oracle=preds/p3_oracle.jsonl --plot preds/p3_comparison.png

## v5 highlights (merging v4 upgrades + bug fixes)
- Zoom kind (light / line) routed from question wording by default (--no_kind_rule to ablate). Camera forced to "center" ONLY for lights; lane lines use BoxHead camera prediction so side-camera lane lines are never cropped from center.
- Singleton fallback in crs/real.py unlocks traffic_light_status, traffic_light_change, and crossing_type when a single light/crossing has empty uniqueKey; human-annotated UUID signs also check obj_box.
- Anti-overfitting defaults: P2/P3 use lr=5e-5, lora_plus=4.0, capped loss calibration (aux<=1.0, rank<=2.0, box in [1.0,2.0]), and keep best.pt by lowest dev val_loss as final.pt (--no_keep_best to disable).
- Minimum crop floor (48 px lights, 110 px lines) for predicted boxes + eval center_hit and per-kind IoU (iou_light, iou_line, center_light, center_line) + train --probe timing check.
