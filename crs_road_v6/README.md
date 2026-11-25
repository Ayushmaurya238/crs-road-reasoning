# CRS road reasoning (P0 / P1 / P2 / P3 with ResNet CNN Zoom)

pip install -r requirements.txt

## Fast 1-command dry run (synth or real, slow-GPU friendly)
python run.py dryrun --mode synth --out dry_synth --dtype fp16 --grad_ckpt --img_side 224 --lora_r 8 --cnn_arch resnet50 --steps 30 --batch 4
python run.py dryrun --mode real --graphs CRS_Logs/annotations --meta CRS_Logs/metadata --frames av2_frames --out dry_real --nlogs 6 --dtype fp16 --grad_ckpt --img_side 224 --lora_r 8 --cnn_arch resnet50 --steps 40 --batch 4

## Real data pipeline (MAPVQA graphs + metadata + AV2 frames)
python download_logs.py --out CRS_Logs
python fetch_av2_frames.py --log <logid> --meta CRS_Logs/metadata --out av2_frames
python run.py real --graphs CRS_Logs/annotations --meta CRS_Logs/metadata --frames av2_frames --out data --split 50,15,15
python check_real.py --scenes data/scenes --out overlays --n 5
python run.py cropcache --scenes data/scenes --out data/crops

## ResNet-50 CNN Zoom fine-tuning on real traffic images (freezes early layers, fine-tunes layer3+layer4 in C0 and layer4 in Stage C)
python run.py cnn --arch resnet50 --cache data/crops --fold 0 --nfolds 2 --epochs 6 --out ck/c0_f0.pt
python run.py cnn --arch resnet50 --cache data/crops --fold 1 --nfolds 2 --epochs 6 --out ck/c0_f1.pt

## VLM training (add --load_4bit on <16GB GPUs; --dtype bf16 on Ampere/Ada)
python run.py train --variant p1 --scenes data/scenes --qa_train data/qa_train.jsonl --qa_dev data/qa_dev.jsonl --out runs/p1 --epochs 2 --batch 8 --img_side 336 --lora_r 8 --grad_ckpt --dtype fp16
python run.py train --variant p2 --prefix_only --scenes data/scenes --qa_train data/qa_train.jsonl --qa_dev data/qa_dev.jsonl --out runs/p2 --epochs 2 --batch 8 --img_side 336 --lora_r 8 --grad_ckpt --dtype fp16
python run.py train --variant p2 --resume runs/p2/prefix.pt --scenes data/scenes --qa_train data/qa_train.jsonl --qa_dev data/qa_dev.jsonl --out runs/p2b --epochs 2 --batch 8 --img_side 336 --lora_r 8 --grad_ckpt --dtype fp16
python run.py train --variant p3 --resume runs/p2/prefix.pt --cnn_arch resnet50 --cnn_ckpts ck/c0_f0.pt ck/c0_f1.pt --gt_box_prob 0.3 --scenes data/scenes --qa_train data/qa_train.jsonl --qa_dev data/qa_dev.jsonl --out runs/p3 --epochs 2 --batch 8 --img_side 336 --lora_r 8 --grad_ckpt --dtype fp16

## Evaluate & plot P0 / P1 / P2 / P3 comparison graph
python run.py eval --variant p3 --ckpt runs/p3/final.pt --scenes data/scenes --qa_eval data/qa_dev.jsonl --out preds/p3.jsonl --tta --tta_n 2 --img_side 336 --dtype fp16
python run.py compare --runs P0=preds/p0.jsonl P1=preds/p1.jsonl P2=preds/p2.jsonl P3=preds/p3.jsonl --plot preds/p3_comparison.png
