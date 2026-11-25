import glob
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from .common import (
    CAMS,
    COLOR_NAME,
    LCOLOR,
    LIGHT,
    LTYPE,
    TYPE_NAME,
    crop_square,
    fold_of,
    make_scaler,
    read_json,
    read_jsonl,
    write_jsonl,
)

HEADS = {"light": len(LIGHT), "ltype": len(LTYPE), "lcolor": len(LCOLOR)}
UNFREEZE = {
    "googlenet": ["inception5a", "inception5b"],
    "resnet50": ["layer4"],
    "convnext_tiny": ["features.7"],
    "efficientnet_b0": ["features.7", "features.8"],
}
UNFREEZE_C0 = {
    "googlenet": ["inception4e", "inception5a", "inception5b"],
    "resnet50": ["layer3", "layer4"],
    "convnext_tiny": ["features.6", "features.7"],
    "efficientnet_b0": ["features.6", "features.7", "features.8"],
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(img):
    t = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (t - MEAN) / STD


def augment(img, size, rng):
    if rng.random() < 0.5:
        s = rng.uniform(0.35, 1.0)
        k = max(8, int(size * s))
        img = img.resize((k, k), Image.BILINEAR).resize((size, size), Image.BILINEAR)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.65, 1.35))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.2))
    return img


def build_backbone(arch, pretrained=True):
    tv = torchvision.models
    w = None
    if arch == "googlenet":
        if pretrained:
            w = tv.GoogLeNet_Weights.DEFAULT
        m = tv.googlenet(weights=w, aux_logits=False)
        m.fc = nn.Identity()
        return m, 1024
    if arch == "resnet50":
        if pretrained:
            w = tv.ResNet50_Weights.DEFAULT
        m = tv.resnet50(weights=w)
        m.fc = nn.Identity()
        return m, 2048
    if arch == "convnext_tiny":
        if pretrained:
            w = tv.ConvNeXt_Tiny_Weights.DEFAULT
        m = tv.convnext_tiny(weights=w)
        m.classifier[2] = nn.Identity()
        return m, 768
    if arch == "efficientnet_b0":
        if pretrained:
            w = tv.EfficientNet_B0_Weights.DEFAULT
        m = tv.efficientnet_b0(weights=w)
        m.classifier = nn.Identity()
        return m, 1280
    raise ValueError(arch)


class Fusion(nn.Module):
    def __init__(self, dim, qdim):
        super().__init__()
        self.q = nn.Linear(qdim, 256)
        self.mlp = nn.Sequential(nn.Linear(dim + 256, 512), nn.GELU())
        self.out = nn.ModuleDict({k: nn.Linear(512, n) for k, n in HEADS.items()})
        for o in self.out.values():
            nn.init.zeros_(o.weight)
            nn.init.zeros_(o.bias)

    def forward(self, e, q):
        z = self.mlp(torch.cat([e, F.gelu(self.q(q.float()))], -1))
        return {k: o(z) for k, o in self.out.items()}


class ZoomCNN(nn.Module):
    def __init__(self, arch, pretrained=True, qdim=0):
        super().__init__()
        self.arch = arch
        self.backbone, self.dim = build_backbone(arch, pretrained)
        self.heads = nn.ModuleDict({k: nn.Linear(self.dim, n) for k, n in HEADS.items()})
        self.fusion = Fusion(self.dim, qdim) if qdim else None

    def forward(self, x, q=None):
        e = self.backbone(x)
        out = {k: h(e) for k, h in self.heads.items()}
        if self.fusion is not None and q is not None:
            d = self.fusion(e, q)
            out = {k: out[k] + d[k] for k in out}
        return out


def set_trainable_last(model, last_only=True, prefixes=None):
    pref = prefixes if prefixes is not None else UNFREEZE[model.arch]
    for n, p in model.backbone.named_parameters():
        p.requires_grad = (not last_only) or any(n.startswith(u) for u in pref)


def freeze_bn_for_frozen(module):
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            ps = list(m.parameters())
            if ps and not any(p.requires_grad for p in ps):
                m.eval()


def focal(logits, y, w=None, gamma=2.0):
    m = y != -100
    if int(m.sum()) == 0:
        return logits.sum() * 0.0
    lp = F.log_softmax(logits[m].float(), -1)
    lpt = lp.gather(1, y[m][:, None])[:, 0]
    loss = -((1 - lpt.exp()) ** gamma) * lpt
    if w is not None:
        loss = loss * w[m]
    return loss.mean()


def zoom_loss(out, tgt, kind, size_w=1.0):
    dev = next(iter(out.values())).device
    w = torch.tensor([size_w], device=dev)
    if kind == "light":
        return focal(out["light"], torch.tensor([tgt["light"]], device=dev), w)
    return focal(out["ltype"], torch.tensor([tgt["ltype"]], device=dev), w) + focal(
        out["lcolor"], torch.tensor([tgt["lcolor"]], device=dev), w
    )


def probs(out, kind):
    if kind == "light":
        return {"light": F.softmax(out["light"].float(), -1)[0]}
    return {"ltype": F.softmax(out["ltype"].float(), -1)[0], "lcolor": F.softmax(out["lcolor"].float(), -1)[0]}


def hint_text(kind, pr, flip=0.0, rng=None):
    def read(key, names):
        p = pr[key]
        i = int(p.argmax())
        c = float(p[i])
        if flip > 0 and rng is not None and rng.random() < flip:
            i = rng.randrange(len(names))
        return (names[i] if c >= 0.5 else "uncertain"), c

    if kind == "light":
        s, c = read("light", LIGHT)
        return f"[Zoom] light: status={s} ({c:.2f})"
    t, ct = read("ltype", [TYPE_NAME[x] for x in LTYPE])
    col, cc = read("lcolor", [COLOR_NAME[x] for x in LCOLOR])
    return f"[Zoom] lane line: type={t} ({ct:.2f}), colour={col} ({cc:.2f})"


class CnnBank:
    def __init__(self, arch, qdim, nf, device):
        self.arch, self.nf, self.device = arch, nf, device
        self.models = [ZoomCNN(arch, False, qdim).to(device).eval() for _ in range(nf)]

    def get(self, sid):
        return self.models[fold_of(sid, self.nf)] if self.nf > 1 else self.models[0]

    def load_c0(self, paths):
        for m, p in zip(self.models, paths):
            sd = torch.load(p, map_location="cpu", weights_only=False)["state"]
            miss = m.load_state_dict(sd, strict=False)
            bad = [k for k in miss.missing_keys if not k.startswith("fusion")]
            if bad or miss.unexpected_keys:
                raise RuntimeError(f"checkpoint mismatch: {bad} {miss.unexpected_keys}")

    def state(self):
        return [m.state_dict() for m in self.models]

    def load_state(self, sds):
        for m, sd in zip(self.models, sds):
            m.load_state_dict(sd)

    def prepare_train(self):
        for m in self.models:
            set_trainable_last(m, True, UNFREEZE[m.arch])
            for p in list(m.heads.parameters()) + list(m.fusion.parameters()):
                p.requires_grad = True
            m.eval()

    def optimizer(self, lr_bb=2e-5, lr_head=1e-4, lr_fusion=5e-4):
        bb, hd, fu = [], [], []
        for m in self.models:
            bb += [p for p in m.backbone.parameters() if p.requires_grad]
            hd += list(m.heads.parameters())
            fu += list(m.fusion.parameters())
        groups = [
            {"params": bb, "lr": lr_bb, "base_lr": lr_bb},
            {"params": hd, "lr": lr_head, "base_lr": lr_head},
            {"params": fu, "lr": lr_fusion, "base_lr": lr_fusion},
        ]
        return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.01)


def build_crop_cache(scenes_dir, out_dir, side=128, expand=2.0):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(scenes_dir, "*.json"))):
        sc = read_json(f)
        sz = sc["image_size"]
        cache = {}

        def img(path):
            if path not in cache:
                if len(cache) >= 3:
                    cache.clear()
                cache[path] = Image.open(path).convert("RGB")
            return cache[path]

        for t, fr in enumerate(sc["frames"]):
            for j, lt in enumerate(fr.get("lights", [])):
                if lt["status"] not in LIGHT:
                    continue
                w, h = sz[lt["cam"]]
                b = [lt["box"][0] / w, lt["box"][1] / h, lt["box"][2] / w, lt["box"][3] / h]
                lid = str(lt["id"]).replace("/", "_").replace(" ", "_")
                fn = os.path.join(out_dir, f"{sc['scene_id']}_{lid}_{t}_{j}.png")
                crop_square(img(fr["images"][lt["cam"]]), b, expand, side).save(fn)
                px = min(lt["box"][2] - lt["box"][0], lt["box"][3] - lt["box"][1])
                rows.append({"file": fn, "scene_id": sc["scene_id"], "split": sc["split"], "light": LIGHT.index(lt["status"]), "ltype": -100, "lcolor": -100, "side_px": px})
        fr = sc["frames"][-1]
        for j, ln in enumerate(sc.get("lines", [])):
            if ln["type"] not in LTYPE or ln["color"] not in LCOLOR:
                continue
            w, h = sz[ln["cam"]]
            b = [ln["box"][0] / w, ln["box"][1] / h, ln["box"][2] / w, ln["box"][3] / h]
            lid = str(ln["id"]).replace("/", "_").replace(" ", "_")
            fn = os.path.join(out_dir, f"{sc['scene_id']}_{lid}_{j}.png")
            crop_square(img(fr["images"][ln["cam"]]), b, expand, side).save(fn)
            px = min(ln["box"][2] - ln["box"][0], ln["box"][3] - ln["box"][1])
            rows.append({"file": fn, "scene_id": sc["scene_id"], "split": sc["split"], "light": -100, "ltype": LTYPE.index(ln["type"]), "lcolor": LCOLOR.index(ln["color"]), "side_px": px})
        cache.clear()
    write_jsonl(os.path.join(out_dir, "index.jsonl"), rows)
    return len(rows)


def _extra_light_rows(extra_dir):
    if not extra_dir or not os.path.isdir(extra_dir):
        return []
    out = []
    for li, name in enumerate(LIGHT):
        d = os.path.join(extra_dir, name)
        if not os.path.isdir(d):
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for fn in sorted(glob.glob(os.path.join(d, ext))):
                out.append({"file": fn, "scene_id": "__extra__", "split": "train", "light": li, "ltype": -100, "lcolor": -100, "side_px": 24.0, "precropped": True})
    return out


class CropSet(Dataset):
    def __init__(self, rows, size=96, train=False, expand=1.3, cache_expand=2.0, seed=0):
        self.rows, self.size, self.train = rows, size, train
        self.f = expand / cache_expand
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = Image.open(r["file"]).convert("RGB")
        rng = random.Random(self.rng.random() + i) if self.train else None
        if r.get("precropped"):
            im = im.resize((self.size, self.size), Image.BILINEAR)
        else:
            n = im.size[0]
            f, cx, cy = self.f, 0.5, 0.5
            if self.train:
                f *= math.exp(rng.gauss(0, 0.15))
                cx += rng.gauss(0, 0.08) * f
                cy += rng.gauss(0, 0.08) * f
            box = (int(round((cx - f / 2) * n)), int(round((cy - f / 2) * n)), int(round((cx + f / 2) * n)), int(round((cy + f / 2) * n)))
            im = im.crop(box).resize((self.size, self.size), Image.BILINEAR)
        if self.train:
            im = augment(im, self.size, rng)
        w = float(min(3.0, max(1.0, 32.0 / max(r["side_px"], 1.0))))
        y = torch.tensor([r["light"], r["ltype"], r["lcolor"]])
        return to_tensor(im), y, torch.tensor(w)


def eval_cnn(model, rows, device, size, bs=64):
    model.eval()
    dl = DataLoader(CropSet(rows, size, False), batch_size=bs, shuffle=False)
    hit = {k: 0 for k in HEADS}
    tot = {k: 0 for k in HEADS}
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    with torch.no_grad():
        for x, y, _ in dl:
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x.to(device))
            for j, k in enumerate(HEADS):
                m = y[:, j] != -100
                if m.any():
                    pred = out[k].argmax(-1).cpu()
                    hit[k] += int((pred[m] == y[m, j]).sum())
                    tot[k] += int(m.sum())
    return {k: hit[k] / tot[k] for k in HEADS if tot[k] > 0}


def _c0_param_groups(model, lr_bb, lr_head, last_only):
    last_top = UNFREEZE[model.arch]
    if last_only:
        set_trainable_last(model, True, UNFREEZE_C0[model.arch])
    else:
        set_trainable_last(model, False)
    hi, lo = [], []
    for n, p in model.backbone.named_parameters():
        if not p.requires_grad:
            continue
        if any(n.startswith(u) for u in last_top):
            hi.append(p)
        else:
            lo.append(p)
    groups = []
    if lo:
        groups.append({"params": lo, "lr": lr_bb * 0.3})
    if hi:
        groups.append({"params": hi, "lr": lr_bb})
    groups.append({"params": list(model.heads.parameters()), "lr": lr_head})
    return groups


def train_c0(
    arch,
    cache_dir,
    out_path,
    init=None,
    epochs=5,
    bs=64,
    lr_bb=1e-4,
    lr_head=1e-3,
    size=96,
    fold=None,
    nfolds=2,
    pretrained=True,
    device="cuda",
    seed=0,
    last_only=True,
    extra_lights=None,
):
    torch.manual_seed(seed)
    rows = read_jsonl(os.path.join(cache_dir, "index.jsonl"))
    tr = [r for r in rows if r["split"] == "train"]
    if fold is not None:
        tr = [r for r in tr if fold_of(r["scene_id"], nfolds) != fold]
    extra = _extra_light_rows(extra_lights)
    if extra:
        tr = tr + extra
        print(f"c0 added {len(extra)} real traffic light crops from {extra_lights}")
    if not tr:
        raise RuntimeError(f"no training crops found in {cache_dir} (fold={fold})")
    dv = [r for r in rows if r["split"] == "dev"]
    model = ZoomCNN(arch, pretrained).to(device)
    if init:
        model.backbone.load_state_dict(torch.load(init, map_location="cpu", weights_only=False)["backbone"])
    groups = _c0_param_groups(model, lr_bb, lr_head, last_only=last_only)
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    nw = 2 if use_cuda else 0
    dl = DataLoader(CropSet(tr, size, True, seed=seed), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=use_cuda, drop_last=len(tr) > bs)
    total = max(1, epochs * len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=total, pct_start=0.1)
    scaler = make_scaler(use_cuda)
    best_score, best_state, best_metrics = -1.0, None, {}
    for ep in range(epochs):
        model.train()
        if last_only:
            freeze_bn_for_frozen(model.backbone)
        run = 0.0
        for x, y, w in dl:
            x, y, w = x.to(device, non_blocking=use_cuda), y.to(device, non_blocking=use_cuda), w.to(device, non_blocking=use_cuda)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x)
                loss = sum(focal(out[k], y[:, j], w) for j, k in enumerate(HEADS))
            opt.zero_grad(set_to_none=True)
            if scaler:
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    sched.step()
            else:
                loss.backward()
                opt.step()
                sched.step()
            run += loss.item()
        m = eval_cnn(model, dv, device, size, bs=bs) if dv else {}
        sc = float(np.mean(list(m.values()))) if m else -run / max(1, len(dl))
        if sc >= best_score:
            best_score = sc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = m
        m_str = " ".join(f"{k}={v:.3f}" for k, v in m.items())
        print(f"c0 {arch} (last_only={last_only}) ep{ep} loss {run / max(1, len(dl)):.4f} {m_str}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"arch": arch, "state": best_state or model.state_dict(), "metrics": best_metrics}, out_path)
    return best_metrics


class _Imgs(Dataset):
    def __init__(self, items, size, train, seed=0):
        self.items, self.size, self.train, self.rng = items, size, train, random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, y = self.items[i]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        if self.train:
            img = augment(img, self.size, random.Random(self.rng.random() + i))
        return to_tensor(img), y


def prepare_bosch(root, out_dir, expand=1.5, size=96):
    import yaml

    names = {"red": "red", "green": "green", "yellow": "yellow", "off": "off"}
    with open(os.path.join(root, "train.yaml")) as f:
        ann = yaml.safe_load(f)
    n = 0
    for it in ann:
        path = os.path.normpath(os.path.join(root, it["path"]))
        if not os.path.exists(path) or not it["boxes"]:
            continue
        img = Image.open(path).convert("RGB")
        w, h = img.size
        for j, b in enumerate(it["boxes"]):
            lab = str(b["label"]).lower()
            cls = next((v for k, v in names.items() if lab.startswith(k)), None)
            if cls is None or b.get("occluded"):
                continue
            box = [b["x_min"] / w, b["y_min"] / h, b["x_max"] / w, b["y_max"] / h]
            if (b["x_max"] - b["x_min"]) < 4 or (b["y_max"] - b["y_min"]) < 4:
                continue
            d = os.path.join(out_dir, cls)
            os.makedirs(d, exist_ok=True)
            crop_square(img, box, expand, size).save(os.path.join(d, f"{os.path.basename(path)[:-4]}_{j}.png"))
            n += 1
    return n


def pretrain_traffic(arch, gtsrb_root, bosch_dir, out_path, epochs=3, bs=64, size=96, lr_head=1e-3, lr_conv=3e-5, device="cuda", pretrained=True, seed=0):
    torch.manual_seed(seed)
    tvd = torchvision.datasets
    gt_tr = tvd.GTSRB(gtsrb_root, split="train", download=True)
    gt_va = tvd.GTSRB(gtsrb_root, split="test", download=True)
    gt_tr = [(s[0], s[1]) for s in gt_tr._samples]
    gt_va = [(s[0], s[1]) for s in gt_va._samples]
    classes = sorted(d for d in os.listdir(bosch_dir) if os.path.isdir(os.path.join(bosch_dir, d)))
    bo = []
    for ci, c in enumerate(classes):
        for fn in glob.glob(os.path.join(bosch_dir, c, "*.png")):
            bo.append((fn, ci))
    rng = random.Random(seed)
    rng.shuffle(bo)
    nv = max(1, len(bo) // 10)
    bo_va, bo_tr = bo[:nv], bo[nv:]
    backbone, dim = build_backbone(arch, pretrained)
    backbone = backbone.to(device)
    for n, p in backbone.named_parameters():
        p.requires_grad = any(n.startswith(u) for u in UNFREEZE_C0[arch])
    h_sign = nn.Linear(dim, 43).to(device)
    h_light = nn.Linear(dim, len(classes)).to(device)
    opt = torch.optim.AdamW(
        [
            {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": lr_conv},
            {"params": list(h_sign.parameters()) + list(h_light.parameters()), "lr": lr_head},
        ],
        weight_decay=0.01,
    )
    use_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    nw = 2 if use_cuda else 0
    dl_s = DataLoader(_Imgs(gt_tr, size, True, seed), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=use_cuda, drop_last=True)
    dl_l = DataLoader(_Imgs(bo_tr, size, True, seed), batch_size=bs, shuffle=True, num_workers=nw, pin_memory=use_cuda, drop_last=len(bo_tr) > bs)
    scaler = make_scaler(use_cuda)

    def cycle(dl):
        while True:
            for b in dl:
                yield b

    it_l = cycle(dl_l)
    for ep in range(epochs):
        backbone.train()
        freeze_bn_for_frozen(backbone)
        run = 0.0
        for xs, ys in dl_s:
            xl, yl = next(it_l)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = F.cross_entropy(h_sign(backbone(xs.to(device))), ys.to(device)) + F.cross_entropy(h_light(backbone(xl.to(device))), yl.to(device))
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            run += loss.item()
        print(f"c-0.5 {arch} ep{ep} loss {run / max(1, len(dl_s)):.4f}")

    def acc(items, head):
        backbone.eval()
        hit = 0
        with torch.no_grad():
            for x, y in DataLoader(_Imgs(items, size, False), batch_size=128):
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    hit += int((head(backbone(x.to(device))).argmax(-1).cpu() == y).sum())
        return hit / max(1, len(items))

    metrics = {"sign_acc": acc(gt_va, h_sign), "light_acc": acc(bo_va, h_light), "light_classes": classes}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"arch": arch, "backbone": backbone.state_dict(), "metrics": metrics}, out_path)
    return metrics
