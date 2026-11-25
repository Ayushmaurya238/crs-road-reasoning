import json
import shutil
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import cnn as C
from . import det as D
from .common import CAMS, KINDS, LETTERS, crop_square, iou_xyxy, make_scaler, read_json, read_jsonl, seed_all, write_jsonl

QV = r".*layers\.\d+\.self_attn\.(q_proj|v_proj)"
ALL = r".*layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))"
KIND_W = (0.25, 4.0, 3.0)


def rule_kind(q):
    s = q.strip().lower()
    if s.startswith("what is the status of") or s.startswith("did the status of"):
        return "light"
    if s.startswith("what is the") and "lane line" in s.split(" of ")[0]:
        return "line"
    return "none"


def kind_from_rule(args):
    return not getattr(args, "no_kind_rule", False)


class BoxHead(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Linear(512, 10))
        with torch.no_grad():
            self.net[-1].bias[2:4].fill_(-3.0)

    def forward(self, x):
        o = self.net(x.float())
        return o[..., :4].sigmoid(), o[..., 4:7], o[..., 7:10]


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)


def giou_loss(a, b):
    eps = 1e-6
    ix1, iy1 = torch.max(a[:, 0], b[:, 0]), torch.max(a[:, 1], b[:, 1])
    ix2, iy2 = torch.min(a[:, 2], b[:, 2]), torch.min(a[:, 3], b[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    aa = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    ab = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = aa + ab - inter + eps
    iou = inter / union
    ex1, ey1 = torch.min(a[:, 0], b[:, 0]), torch.min(a[:, 1], b[:, 1])
    ex2, ey2 = torch.max(a[:, 2], b[:, 2]), torch.max(a[:, 3], b[:, 3])
    enc = (ex2 - ex1) * (ey2 - ey1) + eps
    return (1 - (iou - (enc - union) / enc)).mean()


class Ctx:
    pass


def hidden_size(model):
    cfg = model.config
    return getattr(getattr(cfg, "text_config", cfg), "hidden_size")


def setup(args, train=True, lora=None):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ctx = Ctx()
    ctx.args = args
    ctx.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    ctx.proc = AutoProcessor.from_pretrained(args.model)

    kw = {}
    load_4bit = getattr(args, "load_4bit", False)
    if load_4bit and ctx.dev.type == "cuda":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=ctx.dtype,
        )
        kw["device_map"] = "auto"

    def _load(extra):
        try:
            return AutoModelForImageTextToText.from_pretrained(args.model, dtype=ctx.dtype, **extra)
        except TypeError:
            return AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=ctx.dtype, **extra)

    try:
        model = _load({**kw, "attn_implementation": "sdpa"})
    except Exception:
        model = _load(kw)

    if not load_4bit:
        model.to(ctx.dev)
    if train and load_4bit and ctx.dev.type == "cuda":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    elif train and args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    mode = lora if lora is not None else args.lora
    args.lora = mode
    r = getattr(args, "lora_r", 16)
    if mode != "none":
        cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=ALL if mode == "all" else QV, bias="none")
        model = get_peft_model(model, cfg)
    ctx.model = model
    ctx.hs = hidden_size(model)
    finish_ctx(ctx)
    return ctx


def finish_ctx(ctx):
    ctx.tok = ctx.proc.tokenizer
    ctx.imend = ctx.tok.convert_tokens_to_ids("<|im_end|>")
    ids = []
    for L in LETTERS:
        t = ctx.tok.encode(L, add_special_tokens=False)
        if len(t) != 1:
            raise RuntimeError(f"letter {L} is not a single token")
        ids.append(t[0])
    ctx.LID = ids
    ctx.LIDt = torch.tensor(ids, device=ctx.dev)
    if not hasattr(ctx, "heads"):
        ctx.heads = None
    if not hasattr(ctx, "bank"):
        ctx.bank = None


class QADataset:
    def __init__(self, qa_path, scenes_dir, limit=None, max_img_cache=256):
        self.rows = read_jsonl(qa_path)
        if limit:
            self.rows = self.rows[:limit]
        self.dir = scenes_dir
        self.cache = {}
        self.img_cache = {}
        self.max_img_cache = max_img_cache

    def __len__(self):
        return len(self.rows)

    def scene(self, sid):
        if sid not in self.cache:
            if len(self.cache) > 128:
                self.cache.clear()
            self.cache[sid] = read_json(os.path.join(self.dir, sid + ".json"))
        return self.cache[sid]

    def images(self, sc, side, blank):
        fr = sc["frames"][-1]
        key = (sc["scene_id"], side, blank)
        if key not in self.img_cache:
            if len(self.img_cache) >= self.max_img_cache:
                self.img_cache.pop(next(iter(self.img_cache)))
            self.img_cache[key] = load_images(sc, side, blank)[0]
        return self.img_cache[key], fr["images"]


def make_text(question, options, mode):
    body = "You are given three front camera images (left, center, right) from an ego vehicle. The question refers to the current frame.\n"
    body += f"Question: {question}\n" + "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, options))
    tail = "Answer with the letter only" if mode == "ans" else "Reason step by step then finish with Answer and the letter"
    return body + "\n" + tail


def load_images(sc, side, blank):
    fr = sc["frames"][-1]
    imgs = []
    for c in CAMS:
        im = Image.open(fr["images"][c]).convert("RGB")
        r = side / max(im.size)
        im = im.resize((max(56, int(im.width * r)), max(56, int(im.height * r))), Image.BILINEAR)
        if blank:
            im = Image.new("RGB", im.size, (127, 127, 127))
        imgs.append(im)
    return imgs, fr["images"]


def get_item(ds, idx, epoch, train, args):
    q = ds.rows[idx]
    sc = ds.scene(q["scene_id"])
    rng = random.Random(args.seed * 1000003 + epoch * 10007 + idx)
    perm = list(range(4))
    if train:
        rng.shuffle(perm)
    opts = [q["options"][j] for j in perm]
    ans = perm.index(q["answer"])
    mode = "cot" if (train and rng.random() < args.cot_frac) else "ans"
    imgs, paths = ds.images(sc, args.img_side, args.blank)
    parts = q["cot_parts"] if args.link_weighted else [[t, 1.0] for t, _ in q["cot_parts"]]
    return {
        "q": q,
        "sc": sc,
        "scene_id": q["scene_id"],
        "imgs": imgs,
        "paths": paths,
        "text": make_text(q["question"], opts, mode),
        "opts": opts,
        "perm": perm,
        "ans": ans,
        "mode": mode,
        "parts": parts,
        "tgt": q["target"],
        "rng": rng,
    }


def base_inputs(ctx, imgs, text):
    msgs = [{"role": "user", "content": [{"type": "image"} for _ in imgs] + [{"type": "text", "text": text}]}]
    prompt = ctx.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return ctx.proc(text=[prompt], images=imgs, return_tensors="pt")


def compose(ctx, b, hint_ids, resp_ids):
    ids = b["input_ids"][0].tolist()
    k = len(ids) - 1 - ids[::-1].index(ctx.imend)
    new = ids[:k] + hint_ids + ids[k:] + resp_ids
    out = {}
    for key, val in b.items():
        if key == "input_ids":
            out[key] = torch.tensor([new])
        elif key == "attention_mask":
            out[key] = torch.ones(1, len(new), dtype=val.dtype)
        elif key in ("token_type_ids", "mm_token_type_ids"):
            v = val[0].tolist()
            out[key] = torch.tensor([v[:k] + [0] * len(hint_ids) + v[k:] + [0] * len(resp_ids)], dtype=val.dtype)
        else:
            out[key] = val
    res = {}
    for key, val in out.items():
        if key == "pixel_values":
            res[key] = val.to(ctx.dev, ctx.dtype)
        else:
            res[key] = val.to(ctx.dev)
    return res, k - 1, len(ids) + len(hint_ids)


def fwd(ctx, inputs, hidden=True):
    o = ctx.model(**inputs, output_hidden_states=hidden, use_cache=False)
    return o.logits, (o.hidden_states[-1] if hidden else None)


def build_resp(ctx, it):
    a = ctx.LID[it["ans"]]
    if it["mode"] == "ans":
        return [a, ctx.imend], [1.0, 1.0], 0
    ids, w = [], []
    for text, wt in it["parts"]:
        t = ctx.tok.encode(text, add_special_tokens=False)
        ids += t
        w += [wt] * len(t)
    t = ctx.tok.encode("\nAnswer", add_special_tokens=False)
    ids += t
    w += [1.0] * len(t)
    off = len(ids)
    ids += [a, ctx.imend]
    w += [1.0, 1.0]
    return ids, w, off


def zoom_hint(ctx, it, hq, train, oracle=False, scales=None):
    args = ctx.args
    tgt = it["tgt"]
    gt_p = getattr(args, "gt_box_prob", 0.3) if train else 0.0
    use_gt = oracle or (
        train and gt_p > 0 and tgt["kind_z"] != "none" and tgt.get("box") is not None and it["rng"].random() < gt_p
    )
    if use_gt:
        kind = tgt["kind_z"]
        if kind == "none":
            return [], None, None
        cam, bx = tgt["cam"], tgt["box"]
    else:
        box, camlog, kindlog = ctx.heads(hq)
        rule = kind_from_rule(args) and isinstance(it.get("q"), dict) and "question" in it["q"]
        kind = rule_kind(it["q"]["question"]) if rule else KINDS[int(kindlog.argmax())]
        if kind == "none":
            return [], None, None
        cam = "center" if (rule and kind == "light") else CAMS[int(camlog.argmax())]
        bx = cxcywh_to_xyxy(box).clamp(0, 1).tolist()
        if kind == "light" and getattr(args, "use_det", False) and it.get("sc") is not None:
            scam, sbox, snapped = D.snap_light_box(bx, it["sc"])
            if snapped:
                cam, bx = scam, sbox

    if scales is None:
        if not use_gt and kind == "line":
            scales = (1.4, 1.7, 2.0) if (not train and args.tta) else (1.7,)
        else:
            scales = (1.15, 1.3, 1.5) if (not train and args.tta) else (1.3,)

    im = Image.open(it["paths"][cam]).convert("RGB")
    model = ctx.bank.get(it["scene_id"])
    pr_acc, closs = None, None
    jitter = 0.05 if (use_gt and train and not oracle) else 0.0
    use_amp = ctx.dev.type == "cuda"
    for s_i, s in enumerate(scales):
        crop = crop_square(im, bx, s, args.crop_size, jitter=jitter, rng=it.get("rng"), min_side=(8.0 if use_gt else (48.0 if kind == "light" else 110.0)))
        x = C.to_tensor(crop)[None].to(ctx.dev)
        if train:
            with torch.amp.autocast("cuda", dtype=ctx.dtype, enabled=use_amp):
                out = model(x, q=hq[None])
        else:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=ctx.dtype, enabled=use_amp):
                out = model(x, q=hq[None])
        if s_i == 0 and train and tgt["kind_z"] == kind and tgt["cam"] == cam and (use_gt or iou_xyxy(bx, tgt["box"]) > 0.3):
            size_w = float(min(3.0, max(1.0, 32.0 / max(tgt["size_px"] or 32.0, 1.0))))
            closs = C.zoom_loss(out, tgt, kind, size_w)
        pr = {k: v.detach() for k, v in C.probs(out, kind).items()}
        pr_acc = pr if pr_acc is None else {k: pr_acc[k] + pr[k] for k in pr}
    pr_acc = {k: v / len(scales) for k, v in pr_acc.items()}
    text = C.hint_text(kind, pr_acc, args.hint_flip if train else 0.0, it.get("rng") if train else None)
    ids = ctx.tok.encode("\n" + text, add_special_tokens=False)
    return ids, closs, text


def losses(ctx, logits, hq, inp, rs, resp_w, ans_off, it):
    ids = inp["input_ids"][0]
    n = ids.numel()
    lg = logits[0, rs - 1 : n - 1].float()
    tgt_ids = ids[rs:n]
    w = torch.tensor(resp_w, device=lg.device, dtype=torch.float32)
    ce = F.cross_entropy(lg, tgt_ids, reduction="none")
    out = {"lm": (ce * w).sum() / w.sum()}
    pos = rs + ans_off
    rk = logits[0, pos - 1, ctx.LIDt].float()[None]
    out["rank"] = F.cross_entropy(rk, torch.tensor([it["ans"]], device=rk.device))
    if ctx.heads is not None:
        box, cam, kind = ctx.heads(hq)
        tg = it["tgt"]
        ki = KINDS.index(tg["kind_z"])
        out["kind"] = F.cross_entropy(kind[None], torch.tensor([ki], device=kind.device)) * KIND_W[ki]
        if tg["box"] is not None:
            tb = torch.tensor(tg["box"], device=box.device, dtype=torch.float32)
            out["l1"] = F.l1_loss(box, xyxy_to_cxcywh(tb))
            out["giou"] = giou_loss(cxcywh_to_xyxy(box)[None], tb[None])
            out["cam"] = F.cross_entropy(cam[None], torch.tensor([CAMS.index(tg["cam"])], device=cam.device))
            out["iou"] = torch.tensor(iou_xyxy(cxcywh_to_xyxy(box).detach().tolist(), tg["box"]))
    return out


def process_sample(ctx, it, use_hint):
    b = base_inputs(ctx, it["imgs"], it["text"])
    hint_ids, closs = [], None
    if use_hint:
        inp1, qp, _ = compose(ctx, b, [], [])
        with torch.no_grad():
            _, h1 = fwd(ctx, inp1)
            hq1 = h1[0, qp].detach().float()
        del h1
        hint_ids, closs, _ = zoom_hint(ctx, it, hq1, True)
    resp, w, off = build_resp(ctx, it)
    inp, qp, rs = compose(ctx, b, hint_ids, resp)
    lg, h = fwd(ctx, inp, hidden=ctx.heads is not None)
    hq = h[0, qp] if h is not None else None
    L = losses(ctx, lg, hq, inp, rs, w, off, it)
    if closs is not None:
        L["cnn"] = closs
    return L


def stage_weights(stage, cal, staged):
    if not staged:
        return {"lm": 1.0, "rank": 0.0, "box": 0.0, "aux": 0.0}
    if stage == "A":
        return {"lm": 0.3, "rank": 0.0, "box": 1.5, "aux": 0.5}
    return {"lm": 1.0, "rank": 0.5 * cal["rank"], "box": max(1.0, cal["box"]), "aux": 0.3 * cal["aux"]}


def total_loss(L, Wt):
    t = Wt["lm"] * L["lm"] + Wt["rank"] * L["rank"]
    if "l1" in L:
        t = t + Wt["box"] * (5.0 * L["l1"] + 2.0 * L["giou"])
    aux = 0.0
    for k in ("cam", "kind"):
        if k in L:
            aux = aux + L[k]
    t = t + Wt["aux"] * aux
    if "cnn" in L:
        t = t + L["cnn"]
    return t


def lora_params(model):
    return [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]


def grad_norm(params):
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().float().norm() ** 2)
    return math.sqrt(s)


def calibrate(ctx, it, cal):
    params = lora_params(ctx.model)
    norms = {}
    for name in ("lm", "rank", "box", "aux"):
        for p in params:
            p.grad = None
        L = process_sample(ctx, it, False)
        if name == "lm":
            loss = L["lm"]
        elif name == "rank":
            loss = L["rank"]
        elif name == "box":
            if "l1" not in L:
                continue
            loss = 5.0 * L["l1"] + 2.0 * L["giou"]
        else:
            loss = L["kind"] + L.get("cam", 0.0)
        loss.backward()
        norms[name] = grad_norm(params)
    for p in params:
        p.grad = None
    w0 = {"rank": 0.5, "box": 1.0, "aux": 0.3}
    bounds = {"rank": (0.1, 2.0), "box": (1.0, 2.0), "aux": (0.1, 1.0)}
    for k in ("rank", "box", "aux"):
        if k in norms and norms[k] > 0 and norms.get("lm", 0) > 0:
            lo, hi = bounds[k]
            cal[k] = float(min(hi, max(lo, 0.35 * norms["lm"] / (w0[k] * norms[k]))))
    return norms


def lr_mult(step, S, sb, staged):
    wu = min(1.0, (step + 1) / 10)
    if not staged:
        return wu * 0.5 * (1 + math.cos(math.pi * step / S))
    if step < sb:
        return wu * (0.4 + 0.6 * 0.5 * (1 + math.cos(math.pi * step / sb)))
    return 0.5 * 0.5 * (1 + math.cos(math.pi * (step - sb) / max(1, S - sb)))


def make_opt(ctx, args):
    A, B = [], []
    for n, p in ctx.model.named_parameters():
        if p.requires_grad:
            (B if "lora_B" in n else A).append(p)
    ratio = args.lora_plus
    groups = [{"params": A, "lr": args.lr, "base_lr": args.lr}, {"params": B, "lr": args.lr * ratio, "base_lr": args.lr * ratio}]
    if ctx.heads is not None:
        groups.append({"params": list(ctx.heads.parameters()), "lr": 1e-3, "base_lr": 1e-3})
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)


def save_ckpt(ctx, path, step, opt, cal, extra=None):
    from peft import get_peft_model_state_dict

    obj = {
        "step": step,
        "cal": cal,
        "args": {k: v for k, v in vars(ctx.args).items() if isinstance(v, (int, float, str, bool, type(None), list))},
        "lora": get_peft_model_state_dict(ctx.model) if ctx.args.lora != "none" else {},
        "heads": ctx.heads.state_dict() if ctx.heads is not None else None,
        "opt": opt.state_dict() if opt is not None else None,
        "bank": ctx.bank.state() if ctx.bank is not None else None,
    }
    if extra:
        obj.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(obj, path)


def load_ckpt(ctx, path, opt=None, load_opt=True):
    from peft import set_peft_model_state_dict

    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["lora"]:
        set_peft_model_state_dict(ctx.model, ck["lora"])
    if ck["heads"] is not None and ctx.heads is not None:
        ctx.heads.load_state_dict(ck["heads"])
    if opt is not None and load_opt and ck["opt"] is not None:
        opt.load_state_dict(ck["opt"])
    if ck.get("bank") is not None and ctx.bank is not None:
        ctx.bank.load_state(ck["bank"])
    return ck


def attach_heads(ctx, args):
    if args.variant in ("p2", "p3"):
        ctx.heads = BoxHead(ctx.hs).to(ctx.dev)


def attach_bank(ctx, args):
    if args.variant == "p3":
        if not args.cnn_ckpts:
            raise RuntimeError("p3 requires --cnn_ckpts (run cropcache and cnn first)")
        ctx.bank = C.CnnBank(args.cnn_arch, ctx.hs, len(args.cnn_ckpts), ctx.dev)
        ctx.bank.load_c0(args.cnn_ckpts)


@torch.inference_mode()
def quick_val(ctx, ds, n, use_hint=False):
    ctx.model.eval()
    ce, hit, m = 0.0, 0, min(n, len(ds))
    for i in range(m):
        it = get_item(ds, i, 0, False, ctx.args)
        b = base_inputs(ctx, it["imgs"], it["text"])
        hint_ids = []
        if use_hint and ctx.bank is not None and ctx.heads is not None:
            inp1, qp, _ = compose(ctx, b, [], [])
            _, h1 = fwd(ctx, inp1, hidden=True)
            hint_ids, _, _ = zoom_hint(ctx, it, h1[0, qp].float(), False)
        inp, _, _ = compose(ctx, b, hint_ids, [])
        lg, _ = fwd(ctx, inp, hidden=False)
        l4 = lg[0, -1, ctx.LIDt].float()
        ce += float(F.cross_entropy(l4[None], torch.tensor([it["ans"]], device=l4.device)))
        hit += int(int(l4.argmax()) == it["ans"])
    ctx.model.train()
    return {"val_loss": ce / max(1, m), "val_acc": hit / max(1, m)}


def has_grads(o):
    return any(p.grad is not None for g in o.param_groups for p in g["params"])


def _pick_cal_idx(ds, seed):
    boxed = [i for i, r in enumerate(ds.rows) if r["target"].get("box") is not None]
    rng = np.random.RandomState(seed)
    return int(boxed[rng.randint(len(boxed))]) if boxed else int(rng.randint(len(ds)))


def train(args):
    seed_all(args.seed)
    ctx = setup(args, train=True)
    attach_heads(ctx, args)
    attach_bank(ctx, args)
    ds = QADataset(args.qa_train, args.scenes, args.overfit or None)
    dv = QADataset(args.qa_dev, args.scenes) if args.qa_dev and not args.overfit else None
    N = len(ds)
    boxed_idx = [i for i, r in enumerate(ds.rows) if r["target"].get("box") is not None]
    S = args.steps or math.ceil(args.epochs * N / args.batch)
    staged = args.variant != "p1"
    sa = args.stage_a_steps or round(0.15 * S)
    sb = args.stage_b_steps or round(0.7 * S)
    opt = make_opt(ctx, args)
    copt = ctx.bank.optimizer() if ctx.bank is not None else None
    step, cal = 0, {"rank": 1.0, "box": 1.0, "aux": 1.0, "done": False}
    if args.resume:
        ck = load_ckpt(ctx, args.resume, opt, load_opt=True)
        step, cal = ck["step"], ck["cal"]
    if ctx.bank is not None:
        ctx.bank.prepare_train()
    scaler = make_scaler(ctx.dtype == torch.float16 and ctx.dev.type == "cuda")
    params = [p for g in opt.param_groups for p in g["params"]]
    if ctx.heads is None and not params:
        raise RuntimeError("no trainable parameters")
    ctx.model.train()
    os.makedirs(args.out, exist_ok=True)
    if getattr(args, "probe", False):
        pool = boxed_idx or list(range(N))
        ts = []
        for k in range(10):
            t1 = time.time()
            it = get_item(ds, pool[k % len(pool)], 0, True, args)
            L = process_sample(ctx, it, args.variant == "p3")
            total_loss(L, stage_weights("B", cal, staged)).backward()
            if ctx.dev.type == "cuda":
                torch.cuda.synchronize()
            ts.append(time.time() - t1)
        opt.zero_grad(set_to_none=True)
        if copt is not None:
            copt.zero_grad(set_to_none=True)
        sec = float(np.mean(ts[2:]))
        json.dump({"sec_per_sample": sec, "variant": args.variant}, open(os.path.join(args.out, "probe.json"), "w"))
        print(f"probe: {sec:.3f} s/sample")
        return ctx
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best_val = float("inf")
    perms, a_perms, a_cnt = {}, {}, 0
    t0 = time.time()
    start_step = step
    while step < S:
        stage = "A" if step < sa else ("B" if step < sb else "C")
        if staged and args.prefix_only and step == sb:
            save_ckpt(ctx, os.path.join(args.out, "prefix.pt"), step, opt, cal)
            break
        mult = lr_mult(step, S, sb, staged)
        for g in opt.param_groups:
            g["lr"] = g["base_lr"] * mult
        if copt is not None:
            for g in copt.param_groups:
                g["lr"] = g["base_lr"] * mult
        use_hint = args.variant == "p3" and stage == "C"
        if staged and args.variant in ("p2", "p3") and stage == "B" and not cal["done"] and not args.no_calibrate:
            it = get_item(ds, _pick_cal_idx(ds, args.seed), 0, True, args)
            norms = calibrate(ctx, it, cal)
            cal["done"] = True
            print("calibration", norms, {k: cal[k] for k in ("rank", "box", "aux")})
        Wt = stage_weights(stage, cal, staged)
        acc = {}
        for k in range(args.batch):
            if staged and stage == "A" and boxed_idx:
                ep, pos = divmod(a_cnt, len(boxed_idx))
                if ep not in a_perms:
                    a_perms[ep] = np.random.RandomState(args.seed + 7919 + ep).permutation(len(boxed_idx))
                idx = boxed_idx[int(a_perms[ep][pos])]
                a_cnt += 1
            else:
                g = step * args.batch + k
                ep, pos = divmod(g, N)
                if ep not in perms:
                    perms[ep] = np.random.RandomState(args.seed + ep).permutation(N)
                idx = int(perms[ep][pos])
            it = get_item(ds, idx, ep, True, args)
            L = process_sample(ctx, it, use_hint)
            loss = total_loss(L, Wt) / args.batch
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: { {k2: float(v.detach()) for k2, v in L.items()} }")
            (scaler.scale(loss) if scaler else loss).backward()
            for key, v in L.items():
                acc[key] = acc.get(key, 0.0) + float(v.detach()) / args.batch

        opts = [o for o in ([opt] + ([copt] if copt is not None else [])) if has_grads(o)]
        if scaler:
            for o in opts:
                scaler.unscale_(o)
        if opt in opts:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        if copt is not None and copt in opts:
            torch.nn.utils.clip_grad_norm_([p for g in copt.param_groups for p in g["params"]], 1.0)
        for o in opts:
            if scaler:
                scaler.step(o)
            else:
                o.step()
            o.zero_grad(set_to_none=True)
        if scaler and opts:
            scaler.update()

        step += 1
        rec = {"step": step, "stage": stage, "lr_mult": mult, "sec": time.time() - t0, **acc}
        if dv is not None and (step % args.eval_every == 0 or step == S):
            rec.update(quick_val(ctx, dv, args.val_n, use_hint=use_hint))
            if rec["val_loss"] < best_val and not args.prefix_only:
                best_val = rec["val_loss"]
                save_ckpt(ctx, os.path.join(args.out, "best.pt"), step, opt, cal)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % args.log_every == 0 or step == S:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        if args.ckpt_every and step % args.ckpt_every == 0:
            save_ckpt(ctx, os.path.join(args.out, "last.pt"), step, opt, cal)
    else:
        save_ckpt(ctx, os.path.join(args.out, "final.pt"), step, opt, cal)
        bp = os.path.join(args.out, "best.pt")
        if os.path.exists(bp) and not getattr(args, "no_keep_best", False):
            shutil.copy(os.path.join(args.out, "final.pt"), os.path.join(args.out, "last_step.pt"))
            shutil.copy(bp, os.path.join(args.out, "final.pt"))
            print(f"final.pt = best dev-loss checkpoint (val_loss {best_val:.4f}); last step kept as last_step.pt")
    ran = max(1, step - start_step)
    sec = (time.time() - t0) / ran
    print(f"seconds/step {sec:.2f}; projected full-run hours at {S} steps: {sec * S / 3600:.2f}")
    if args.overfit:
        r = evaluate(ctx, ds, args, os.path.join(args.out, "overfit_preds.jsonl"))
        print("overfit accuracy", r["acc"])
    return ctx


def letter_logprobs(ctx, it_imgs, question, opts, perm, hint_ids, want_hidden):
    text = make_text(question, [opts[p] for p in perm], "ans")
    b = base_inputs(ctx, it_imgs, text)
    inp, qp, _ = compose(ctx, b, hint_ids, [])
    lg, h = fwd(ctx, inp, hidden=want_hidden)
    lp = F.log_softmax(lg[0, -1, ctx.LIDt].float(), -1)
    return lp, (h[0, qp].float() if h is not None else None)


@torch.inference_mode()
def predict(ctx, ds, i, args):
    q = ds.rows[i]
    sc = ds.scene(q["scene_id"])
    imgs, paths = ds.images(sc, args.img_side, args.blank)
    ident = [0, 1, 2, 3]
    tta_n = getattr(args, "tta_n", 4) if args.tta else 1
    perms = [[(j + s) % 4 for j in range(4)] for s in range(tta_n)]
    need_h = ctx.heads is not None
    lp0, hq = letter_logprobs(ctx, imgs, q["question"], q["options"], ident, [], need_h)
    rec = {"qid": q["qid"], "template": q["template"], "scene_id": q["scene_id"], "log": q.get("log", q["scene_id"])}
    hint_ids, hint = [], None
    if hq is not None:
        box, camlog, kindlog = ctx.heads(hq)
        rule = kind_from_rule(args)
        kind = rule_kind(q["question"]) if rule else KINDS[int(kindlog.argmax())]
        pb = cxcywh_to_xyxy(box).clamp(0, 1).tolist()
        pc = "center" if (rule and kind == "light") else CAMS[int(camlog.argmax())]
        tg = q["target"]
        raw_iou = (iou_xyxy(pb, tg["box"]) if tg["cam"] == pc else 0.0) if tg["box"] is not None else None
        snapped = False
        if getattr(args, "use_det", False) and kind == "light" and not args.oracle_box:
            scam, sbox, snapped = D.snap_light_box(pb, sc)
            if snapped:
                pc, pb = scam, sbox
        rec.update({"pred_kind": kind, "pred_cam": pc, "pred_box": pb, "snapped": int(snapped)})
        if tg["box"] is not None:
            rec["raw_iou"] = raw_iou
            rec["gt_iou"] = iou_xyxy(pb, tg["box"]) if tg["cam"] == pc else 0.0
            gb = tg["box"]
            rec["center_hit"] = int(tg["cam"] == pc and gb[0] <= 0.5 * (pb[0] + pb[2]) <= gb[2] and gb[1] <= 0.5 * (pb[1] + pb[3]) <= gb[3])
            rec["kind_z"] = tg.get("kind_z", "none")
        if ctx.bank is not None:
            it = {"tgt": tg, "sc": sc, "paths": paths, "scene_id": q["scene_id"], "q": q}
            hint_ids, _, hint = zoom_hint(ctx, it, hq, False, oracle=args.oracle_box)
            hint_ids = hint_ids or []
    tot = torch.zeros(4)
    for perm in perms:
        if perm == ident and not hint_ids:
            lp = lp0
        else:
            lp, _ = letter_logprobs(ctx, imgs, q["question"], q["options"], perm, hint_ids, False)
        for j in range(4):
            tot[perm[j]] += float(lp[j])
    pred = int(tot.argmax())
    tg = q["target"]
    rec.update({"pred": pred, "answer": q["answer"], "correct": int(pred == q["answer"]), "hint": hint, "size_px": tg.get("size_px"), "small": q["small"], "reasoning": q["reasoning"], "depth": q["depth"]})
    return rec


def evaluate(ctx, ds, args, out_path):
    ctx.model.eval()
    rows = [predict(ctx, ds, i, args) for i in range(len(ds))]
    write_jsonl(out_path, rows)
    ious = [r["gt_iou"] for r in rows if "gt_iou" in r]
    res = {
        "acc": float(np.mean([r["correct"] for r in rows])),
        "n": len(rows),
        "hints": sum(1 for r in rows if r.get("hint")),
        "snapped": sum(1 for r in rows if r.get("snapped")),
    }
    if ious:
        res["mean_iou"] = float(np.mean(ious))
        res["acc@0.5"] = float(np.mean([x > 0.5 for x in ious]))
        res["center_hit"] = float(np.mean([r["center_hit"] for r in rows if "center_hit" in r]))
        for kz in ("light", "line", "none"):
            sub = [r for r in rows if r.get("kind_z") == kz and "gt_iou" in r]
            if sub:
                res[f"iou_{kz}"] = round(float(np.mean([r["gt_iou"] for r in sub])), 3)
                res[f"center_{kz}"] = round(float(np.mean([r["center_hit"] for r in sub])), 3)
    return res


def run_eval(args):
    seed_all(args.seed)
    meta = None
    if args.ckpt:
        meta = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        for k in ("variant", "model", "cnn_arch", "dtype", "lora_r", "load_4bit"):
            if k in meta["args"] and meta["args"][k] is not None:
                setattr(args, k, meta["args"][k])
        lora = meta["args"]["lora"]
        if meta.get("bank") is not None and not args.cnn_ckpts:
            args.cnn_ckpts = [None] * len(meta["bank"])
    else:
        lora = "none"
        args.variant = "p0"
    ctx = setup(args, train=False, lora=lora)
    attach_heads(ctx, args)
    if args.variant == "p3":
        ctx.bank = C.CnnBank(args.cnn_arch, ctx.hs, len(meta["bank"]), ctx.dev)
    if meta is not None:
        load_ckpt(ctx, args.ckpt, None, load_opt=False)
    limit = getattr(args, "eval_limit", 0) or None
    ds = QADataset(args.qa_eval, args.scenes, limit=limit)
    return evaluate(ctx, ds, args, args.out)
