import argparse
import json
import os
import subprocess
import sys

from crs.common import read_jsonl


def add_vlm_args(p):
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--variant", choices=["p0", "p1", "p2", "p3"], default="p2")
    p.add_argument("--scenes", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--img_side", type=int, default=448)
    p.add_argument("--crop_size", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blank", action="store_true")
    p.add_argument("--cnn_arch", default="resnet50")
    p.add_argument("--cnn_ckpts", nargs="*", default=[])
    p.add_argument("--hint_flip", type=float, default=0.0)
    p.add_argument("--gt_box_prob", type=float, default=0.3)
    p.add_argument("--cot_frac", type=float, default=0.5)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tta_n", type=int, default=4)
    p.add_argument("--oracle_box", action="store_true")
    p.add_argument("--use_det", action="store_true")
    p.add_argument("--no_kind_rule", action="store_true")


def apply_variant(a):
    preset = {
        "p0": ("none", 1.0, False, 1e-4),
        "p1": ("qv", 1.0, False, 1e-4),
        "p2": ("all", 4.0, True, 5e-5),
        "p3": ("all", 4.0, True, 5e-5),
    }[a.variant]
    if getattr(a, "lora", None) is None:
        a.lora = preset[0]
    if getattr(a, "lora_plus", None) is None:
        a.lora_plus = preset[1]
    if getattr(a, "link_weighted", None) is None:
        a.link_weighted = preset[2]
    if hasattr(a, "lr") and a.lr is None:
        a.lr = preset[3]


def sh(cmd):
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_dryrun(a):
    py = sys.executable
    out = a.out
    data = os.path.join(out, "data")
    crops = os.path.join(out, "crops")
    ck = os.path.join(out, "ck")
    runs = os.path.join(out, "runs")
    preds = os.path.join(out, "preds")
    os.makedirs(preds, exist_ok=True)

    if a.mode == "synth":
        sh([py, "run.py", "synth", "--out", data, "--train", "6", "--dev", "4", "--seed", str(a.seed)])
        sh([py, "run.py", "build", "--scenes", f"{data}/scenes", "--out", f"{data}/qa", "--seed", str(a.seed)])
        qa_dir = f"{data}/qa"
    else:
        cmd = [py, "run.py", "real", "--graphs", a.graphs, "--frames", a.frames, "--out", data, "--max_per_log", str(a.max_per_log), "--limit_logs", str(a.nlogs), "--seed", str(a.seed)]
        if a.meta:
            cmd += ["--meta", a.meta]
        sh(cmd)
        qa_dir = data

    sh([py, "run.py", "check", "--qa", qa_dir])
    if a.use_det:
        sh([py, "run.py", "detect", "--scenes", f"{data}/scenes", "--model", a.det_model])
    sh([py, "run.py", "cropcache", "--scenes", f"{data}/scenes", "--out", crops])
    cnn_cmd = [py, "run.py", "cnn", "--arch", a.cnn_arch, "--cache", crops, "--out", f"{ck}/c0.pt", "--epochs", str(a.cnn_epochs), "--bs", "32"]
    if a.extra_lights:
        cnn_cmd += ["--extra_lights", a.extra_lights]
    sh(cnn_cmd)

    vlm = ["--model", a.model, "--scenes", f"{data}/scenes", "--dtype", a.dtype, "--img_side", str(a.img_side), "--lora_r", str(a.lora_r), "--cnn_arch", a.cnn_arch]
    if a.grad_ckpt:
        vlm.append("--grad_ckpt")
    if a.load_4bit:
        vlm.append("--load_4bit")
    det_flag = ["--use_det"] if a.use_det else []

    eval_ev = max(10, min(25, a.steps // 4))
    tr = vlm + ["--qa_train", f"{qa_dir}/qa_train.jsonl", "--qa_dev", f"{qa_dir}/qa_dev.jsonl", "--steps", str(a.steps), "--batch", str(a.batch), "--val_n", "16", "--eval_every", str(eval_ev), "--log_every", "5", "--ckpt_every", "0"]
    if not a.skip_g0:
        sh([py, "run.py", "train", "--variant", "p2", "--overfit", "8", "--steps", "20", "--batch", "4", "--qa_train", f"{qa_dir}/qa_train.jsonl", "--out", f"{runs}/g0"] + vlm)

    sh([py, "run.py", "train", "--variant", "p1", "--out", f"{runs}/p1"] + tr)
    sh([py, "run.py", "train", "--variant", "p2", "--prefix_only", "--out", f"{runs}/p2"] + tr)
    sh([py, "run.py", "train", "--variant", "p2", "--resume", f"{runs}/p2/prefix.pt", "--out", f"{runs}/p2b"] + tr)
    sh([py, "run.py", "train", "--variant", "p3", "--resume", f"{runs}/p2/prefix.pt", "--cnn_ckpts", f"{ck}/c0.pt", "--hint_flip", "0.15", "--gt_box_prob", "0.4", "--out", f"{runs}/p3"] + tr + det_flag)

    ev = vlm + ["--qa_eval", f"{qa_dir}/qa_dev.jsonl", "--eval_limit", str(a.eval_limit)]
    sh([py, "run.py", "eval", "--variant", "p0", "--out", f"{preds}/p0.jsonl"] + ev)
    sh([py, "run.py", "eval", "--variant", "p1", "--ckpt", f"{runs}/p1/final.pt", "--out", f"{preds}/p1.jsonl"] + ev)
    sh([py, "run.py", "eval", "--variant", "p2", "--ckpt", f"{runs}/p2b/final.pt", "--out", f"{preds}/p2.jsonl"] + ev)
    cmp_runs = [f"P0={preds}/p0.jsonl", f"P1={preds}/p1.jsonl", f"P2={preds}/p2.jsonl"]
    if a.use_det:
        sh([py, "run.py", "eval", "--variant", "p3", "--ckpt", f"{runs}/p3/final.pt", "--out", f"{preds}/p3_raw.jsonl"] + ev)
        sh([py, "run.py", "eval", "--variant", "p3", "--ckpt", f"{runs}/p3/final.pt", "--out", f"{preds}/p3.jsonl"] + ev + det_flag)
        cmp_runs += [f"P3-raw={preds}/p3_raw.jsonl", f"P3={preds}/p3.jsonl"]
    else:
        sh([py, "run.py", "eval", "--variant", "p3", "--ckpt", f"{runs}/p3/final.pt", "--out", f"{preds}/p3.jsonl"] + ev)
        cmp_runs += [f"P3={preds}/p3.jsonl"]
    sh([py, "run.py", "eval", "--variant", "p3", "--ckpt", f"{runs}/p3/final.pt", "--oracle_box", "--out", f"{preds}/p3_oracle.jsonl"] + ev)
    cmp_runs += [f"P3-oracle={preds}/p3_oracle.jsonl"]
    sh([py, "run.py", "compare", "--runs", *cmp_runs, "--plot", f"{preds}/p3_comparison.png"])


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("synth")
    p.add_argument("--out", required=True)
    p.add_argument("--train", type=int, default=10)
    p.add_argument("--dev", type=int, default=20)
    p.add_argument("--test", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p = sp.add_parser("openlane")
    p.add_argument("--pkl", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out", required=True)

    p = sp.add_parser("real")
    p.add_argument("--graphs", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--meta", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--max_per_log", type=int, default=300)
    p.add_argument("--limit_logs", type=int, default=0)
    p.add_argument("--split", default="50,15,15")
    p.add_argument("--cv_fold", type=int, default=None)
    p.add_argument("--cv_folds", type=int, default=4)
    p.add_argument("--n_dev", type=int, default=8)
    p.add_argument("--eval_per_template", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p = sp.add_parser("build")
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap_per_template", type=int, default=3)
    p.add_argument("--max_per_scene", type=int, default=40)

    p = sp.add_parser("check")
    p.add_argument("--qa", required=True)

    p = sp.add_parser("detect")
    p.add_argument("--scenes", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--device", default=None)

    p = sp.add_parser("cropcache")
    p.add_argument("--scenes", required=True)
    p.add_argument("--out", required=True)

    p = sp.add_parser("bosch")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)

    p = sp.add_parser("pretrain")
    p.add_argument("--arch", required=True)
    p.add_argument("--gtsrb", required=True)
    p.add_argument("--bosch", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)

    p = sp.add_parser("cnn")
    p.add_argument("--arch", default="resnet50")
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init", default=None)
    p.add_argument("--extra_lights", default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--nfolds", type=int, default=2)
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--size", type=int, default=96)

    p = sp.add_parser("train")
    add_vlm_args(p)
    p.add_argument("--qa_train", required=True)
    p.add_argument("--qa_dev", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--lora_plus", type=float, default=None)
    p.add_argument("--link_weighted", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--stage_a_steps", type=int, default=0)
    p.add_argument("--stage_b_steps", type=int, default=0)
    p.add_argument("--prefix_only", action="store_true")
    p.add_argument("--resume", default=None)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--val_n", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--ckpt_every", type=int, default=50)
    p.add_argument("--overfit", type=int, default=0)
    p.add_argument("--no_calibrate", action="store_true")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--no_keep_best", action="store_true")

    p = sp.add_parser("eval")
    add_vlm_args(p)
    p.add_argument("--qa_eval", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--lora", default=None)
    p.add_argument("--eval_limit", type=int, default=0)

    p = sp.add_parser("report")
    p.add_argument("--preds", required=True)
    p.add_argument("--vs", default=None)

    p = sp.add_parser("compare")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--plot", default=None)

    p = sp.add_parser("dryrun")
    p.add_argument("--mode", choices=["synth", "real"], default="synth")
    p.add_argument("--out", default="dry_run")
    p.add_argument("--graphs", default="CRS_Logs/annotations")
    p.add_argument("--meta", default="CRS_Logs/metadata")
    p.add_argument("--frames", default="av2_frames")
    p.add_argument("--nlogs", type=int, default=6)
    p.add_argument("--max_per_log", type=int, default=60)
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="fp16")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--img_side", type=int, default=224)
    p.add_argument("--cnn_arch", default="resnet50")
    p.add_argument("--extra_lights", default=None)
    p.add_argument("--cnn_epochs", type=int, default=4)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--eval_limit", type=int, default=40)
    p.add_argument("--use_det", action="store_true")
    p.add_argument("--det_model", default="yolo11n.pt")
    p.add_argument("--skip_g0", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    if a.cmd == "synth":
        from crs import synth
        print(synth.generate(a.out, a.train, a.dev, a.test, a.seed), "scenes")
    elif a.cmd == "openlane":
        from crs import openlane
        os.makedirs(a.out, exist_ok=True)
        print(openlane.convert(a.pkl, a.data_root, a.out), "scenes")
    elif a.cmd == "real":
        from crs import real
        st, rep = real.convert(
            a.graphs, a.frames, a.out,
            tuple(int(x) for x in a.split.split(",")),
            a.max_per_log, a.seed, meta_dir=a.meta, limit_logs=a.limit_logs,
            cv_fold=a.cv_fold, cv_folds=a.cv_folds, n_dev=a.n_dev, eval_per_template=a.eval_per_template,
        )
        print(json.dumps(st, indent=1))
        for r in rep:
            print("log %s split=%s questions=%d frames_missing_images=%d meta=%d" % r)
    elif a.cmd == "build":
        from crs import graph
        print(json.dumps(graph.build_dataset(a.scenes, a.out, a.seed, 2, a.cap_per_template, a.max_per_scene), indent=1))
    elif a.cmd == "check":
        from crs import graph
        print(json.dumps(graph.check_dataset(a.qa), indent=1))
    elif a.cmd == "detect":
        from crs import det
        print(json.dumps(det.detect_scenes(a.scenes, a.model, a.conf, a.imgsz, a.bs, a.device), indent=1))
    elif a.cmd == "cropcache":
        from crs import cnn
        print(cnn.build_crop_cache(a.scenes, a.out), "crops")
    elif a.cmd == "bosch":
        from crs import cnn
        print(cnn.prepare_bosch(a.root, a.out), "crops")
    elif a.cmd == "pretrain":
        import torch
        from crs import cnn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(cnn.pretrain_traffic(a.arch, a.gtsrb, a.bosch, a.out, a.epochs, device=dev))
    elif a.cmd == "cnn":
        import torch
        from crs import cnn
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(cnn.train_c0(
            a.arch, a.cache, a.out, a.init, a.epochs, bs=a.bs, size=a.size, fold=a.fold, nfolds=a.nfolds,
            pretrained=not a.no_pretrained, device=dev, last_only=not a.full_finetune, extra_lights=a.extra_lights,
        ))
    elif a.cmd == "train":
        from crs import vlm
        apply_variant(a)
        vlm.train(a)
    elif a.cmd == "eval":
        from crs import vlm
        apply_variant(a)
        print(json.dumps(vlm.run_eval(a)))
    elif a.cmd == "report":
        from crs import report
        rows = read_jsonl(a.preds)
        for k, v in report.summarize(rows).items():
            print(k, v)
        if a.vs:
            print(report.compare(a.preds, a.vs))
    elif a.cmd == "compare":
        from crs import report
        pairs = [tuple(x.split("=", 1)) for x in a.runs]
        report.compare_all(pairs, a.plot)
    elif a.cmd == "dryrun":
        run_dryrun(a)


if __name__ == "__main__":
    main()
