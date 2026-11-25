# CELL 1
# !pip install -q "transformers>=4.57" peft pyyaml
# import torch; print(torch.cuda.get_device_name(0))

# CELL 2
# !cp -r /kaggle/input/crs-road/crs_road /kaggle/working/ ; %cd /kaggle/working/crs_road
# !python run.py synth --out data --train 10 --dev 6
# !python run.py build --scenes data/scenes --out data/qa
# !python run.py check --qa data/qa
# !python run.py cropcache --scenes data/scenes --out data/crops
# !python run.py cnn --arch resnet50 --cache data/crops --out ck/c0.pt --epochs 5

# CELL 3
# M = "--model Qwen/Qwen3-VL-2B-Instruct --scenes data/scenes --dtype fp16 --grad_ckpt --img_side 224"
# T = f"{M} --qa_train data/qa/qa_train.jsonl --qa_dev data/qa/qa_dev.jsonl --steps 40 --batch 4 --val_n 20 --eval_every 20 --log_every 5 --ckpt_every 0"
# !python run.py train --variant p2 --overfit 16 --steps 40 --batch 4 {M} --qa_train data/qa/qa_train.jsonl --out runs/g0

# CELL 4
# !python run.py train --variant p1 {T} --out runs/p1
# !python run.py train --variant p2 --prefix_only {T} --out runs/p2
# !python run.py train --variant p2 --resume runs/p2/prefix.pt {T} --out runs/p2b
# !python run.py train --variant p3 --resume runs/p2/prefix.pt --cnn_ckpts ck/c0.pt --hint_flip 0.15 {T} --out runs/p3

# CELL 5
# E = f"{M} --qa_eval data/qa/qa_dev.jsonl"
# !python run.py eval --variant p0 {E} --out preds/p0.jsonl
# !python run.py eval --variant p1 --ckpt runs/p1/final.pt {E} --out preds/p1.jsonl
# !python run.py eval --variant p2 --ckpt runs/p2b/final.pt {E} --out preds/p2.jsonl
# !python run.py eval --variant p3 --ckpt runs/p3/final.pt {E} --tta --out preds/p3.jsonl
# !python run.py eval --variant p3 --ckpt runs/p3/final.pt {E} --oracle_box --out preds/p3_oracle.jsonl

# CELL 6
# import json
# r=[json.loads(l) for l in open("preds/p3.jsonl")]
# print("hints produced:", sum(1 for x in r if x.get("hint")), "of", len(r)); print([x["hint"] for x in r if x.get("hint")][:3])
# !python run.py report --preds preds/p3.jsonl --vs preds/p1.jsonl
