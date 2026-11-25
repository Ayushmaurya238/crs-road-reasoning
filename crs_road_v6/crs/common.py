import json
import math
import os
import random
import zlib

import numpy as np
from PIL import Image

CAMS = ["left", "center", "right"]
LETTERS = ["A", "B", "C", "D"]
LIGHT = ["red", "green", "yellow"]
LTYPE = ["solid", "dashed", "double_solid", "double_dashed", "none"]
LCOLOR = ["white", "yellow", "none"]
TYPE_NAME = {
    "solid": "solid",
    "dashed": "dashed",
    "double_solid": "double solid",
    "double_dashed": "double dashed",
    "none": "no visible line",
}
COLOR_NAME = {"white": "white", "yellow": "yellow", "none": "not visible"}
KINDS = ["none", "light", "line"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def fold_of(scene_id, n):
    return zlib.crc32(scene_id.split("__f")[0].encode()) % n


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def crop_square(img, box, expand=1.3, size=96, jitter=0.0, rng=None, min_side=8.0):
    w, h = img.size
    x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1, min_side) * expand
    if jitter > 0 and rng is not None:
        cx += rng.gauss(0, jitter) * side
        cy += rng.gauss(0, jitter) * side
        side *= math.exp(rng.gauss(0, jitter))
    left, top = cx - side / 2, cy - side / 2
    region = (int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side)))
    return img.convert("RGB").crop(region).resize((size, size), Image.BILINEAR)


def make_scaler(enabled=True):
    if not enabled:
        return None
    import torch
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()
