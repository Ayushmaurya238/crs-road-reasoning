#!/usr/bin/env python
"""CRS demo: show one question being answered by P2 / P3 / P4 / P4-fused.

--mode replay (default): replays the RECORDED model outputs in preds/*.jsonl
    (graph, logprobs, answers) and recomputes the two-path fusion from the stored
    logprobs. No GPU, no weights. It is NOT live inference and says so on screen.
--mode live: real forward pass for P0-P3 using crs/vlm.py (needs GPU, checkpoint,
    scenes dir and the qa jsonl). P4 live is not possible from this repo (no P4 code).

Examples
  python demo_infer.py --preds ../preds --split test_big --pick fused_wins --qa qa_test_big.jsonl --image path/to/center.jpg
  python demo_infer.py --mode live --code_dir crs_road_v6 --variant p2 --ckpt runs/p2/final.pt \
      --scenes data/scenes --qa qa_test_big.jsonl --qid <qid>
"""
import argparse, glob, json, os, sys, time
import numpy as np

L = "ABCD"
G, R, B, Y, D, X = "\033[92m", "\033[91m", "\033[94m", "\033[93m", "\033[2m", "\033[0m"
PACE = 1.0

def say(s="", t=0.0):
    print(s, flush=True); time.sleep(t * PACE)

def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

def load_preds(d, split):
    out = {}
    for tag, f in (("p2", "p2"), ("p3", "p3_det"), ("p4", "p4"), ("p4f", "p4_fused")):
        out[tag] = {r["qid"]: r for r in rows(f"{d}/{f}_{split}.jsonl")}
    return out

def fuse(r, w):
    s = (1 - w) * np.array(r["lp_hint"]) + w * np.array(r["lp_nohint"])
    return int(np.argmax(s)), s

def pick(P, how):
    f = P["p4f"]
    for q, r in f.items():
        if how == "fused_wins" and r["correct"] and not P["p2"][q]["correct"]: return q
        if how == "p3_wins" and q in P["p3"] and P["p3"][q]["correct"] and not P["p2"][q]["correct"]: return q
        if how == "all_correct" and all(P[k][q]["correct"] for k in ("p2", "p3", "p4f") if q in P[k]): return q
    sys.exit(f"no sample found for --pick {how}")

def show_image(a, rec, ans_txt):
    import matplotlib; 
    try: get_ipython; inline = True
    except NameError: inline = False; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, ax = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.5, 1]}, facecolor="#0F172A")
    if a.image and os.path.exists(a.image): ax[0].imshow(Image.open(a.image))
    else: ax[0].text(.5, .5, "pass --image <frame.jpg>", ha="center", color="w")
    ax[0].axis("off"); ax[0].set_title(f"{rec['scene_id'][:13]}…  [{rec['template']}]", color="w", fontsize=14)
    ax[1].axis("off"); ax[1].text(0, 1, ans_txt, va="top", family="monospace", color="w", fontsize=13)
    plt.tight_layout(); fig.savefig(a.out, dpi=120, facecolor=fig.get_facecolor())
    if inline: plt.show()
    print(f"saved {a.out}")

def replay(a):
    P = load_preds(a.preds, a.split)
    qid = a.qid or pick(P, a.pick)
    r4, rf = P["p4"][qid], P["p4f"][qid]
    qa = next((q for q in rows(a.qa) if q["qid"] == qid), None) if a.qa else None
    gt = rf["answer"]
    say(f"{Y}[REPLAY] recorded outputs of the trained models, not live inference{X}", 1)
    say(f"\n{B}QUESTION{X} ({rf['template']})")
    if qa:
        say(qa["question"]); [say(f"  {L[i]}. {o}") for i, o in enumerate(qa["options"])]
    else: say("  (pass --qa qa_*.jsonl to print question + options)")
    say("", 1)
    say(f"{B}P3 grounding{X}  kind={rf['pred_kind']}  cam={rf['pred_cam']}  snapped_to_YOLO={bool(rf['snapped'])}", 1.5)
    say(f"{B}P4 scene graph{X}  parse_ok={rf['graph_ok']}  lane-F1={rf['gscore']['lane_f1']:.2f}")
    for line in rf["graph_pred"].splitlines(): say("  " + line, 0.4)
    say("", 1)
    say(f"{B}4-way TTA twpath fusion{X}  score = (1-w)*lp_graph + w*lp_nograph   w={a.w}")
    pf, s = fuse(rf, a.w)
    for i in range(4): say(f"  {L[i]}: graph {rf['lp_hint'][i]:8.2f}   no-graph {rf['lp_nohint'][i]:8.2f}   fused {s[i]:8.2f}", 0.4)
    assert a.w != 0.5 or pf == rf["pred"], "fusion recompute mismatch"
    say("", 1)
    res = lambda k, r: f"{k}: {L[r['pred']]} {G+'CORRECT'+X if r['correct'] else R+'WRONG'+X}"
    say(res("P2      ", P["p2"][qid]), .3); say(res("P3      ", P["p3"][qid]) if qid in P["p3"] else "", .3)
    say(res("P4 raw  ", r4), .3)
    final = f"{G}Pred: {L[pf]} | GT: {L[gt]} | fused w={a.w}{X}" if pf == gt else f"{R}Pred: {L[pf]} | GT: {L[gt]}{X}"
    say(final, 4)
    txt = f"P4-fused ({a.w}): {L[pf]}   GT: {L[gt]}\n\nP2: {L[P['p2'][qid]['pred']]}  P4: {L[r4['pred']]}\n\nscene graph:\n{rf['graph_pred']}"
    show_image(a, rf, txt)

def live(a):
    sys.path.insert(0, a.code_dir); import torch
    import run as R_; from crs import vlm as V
    p = argparse.ArgumentParser(); R_.add_vlm_args(p); args = p.parse_args(["--scenes", a.scenes, "--variant", a.variant, "--tta", "--use_det"])
    meta = torch.load(a.ckpt, map_location="cpu", weights_only=False) if a.ckpt else None
    if meta:
        for k in ("variant", "model", "cnn_arch", "dtype", "lora_r", "load_4bit"):
            if meta["args"].get(k) is not None: setattr(args, k, meta["args"][k])
        lora = meta["args"]["lora"]
    else: lora, args.variant = "none", "p0"
    ctx = V.setup(args, train=False, lora=lora); V.attach_heads(ctx, args)
    if args.variant == "p3":
        from crs import cnn as C; ctx.bank = C.CnnBank(args.cnn_arch, ctx.hs, len(meta["bank"]), ctx.dev)
    if meta: V.load_ckpt(ctx, a.ckpt, None, load_opt=False)
    ds = V.QADataset(a.qa, a.scenes); ds.rows = [q for q in ds.rows if q["qid"] == a.qid]; ctx.model.eval()
    q = ds.rows[0]; say(f"{B}QUESTION{X} {q['question']}")
    [say(f"  {L[i]}. {o}") for i, o in enumerate(q["options"])]
    t = time.time(); rec = V.predict(ctx, ds, 0, args); dt = time.time() - t
    say(f"{B}grounding{X} kind={rec.get('pred_kind')} cam={rec.get('pred_cam')} hint={rec.get('hint')}")
    say((G if rec["correct"] else R) + f"Pred: {L[rec['pred']]} | GT: {L[rec['answer']]} | {dt:.2f}s" + X, 4)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["replay", "live"], default="replay")
    ap.add_argument("--preds", default="preds"); ap.add_argument("--split", default="test_big")
    ap.add_argument("--qid"); ap.add_argument("--pick", default="fused_wins", choices=["fused_wins", "p3_wins", "all_correct"])
    ap.add_argument("--qa"); ap.add_argument("--image"); ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--out", default="demo_out.png"); ap.add_argument("--pace", type=float, default=1.0)
    ap.add_argument("--code_dir", default="crs_road_v6"); ap.add_argument("--variant", default="p2")
    ap.add_argument("--ckpt"); ap.add_argument("--scenes")
    a = ap.parse_args(); PACE = a.pace
    live(a) if a.mode == "live" else replay(a)
