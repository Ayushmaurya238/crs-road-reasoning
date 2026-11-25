import os
import pickle
from collections import defaultdict

from PIL import Image

from .common import CAMS, fold_of, write_json

CAM_NAMES = {"left": "ring_front_left", "center": "ring_front_center", "right": "ring_front_right"}
LIGHT_ATTR = {1: "red", 2: "green", 3: "yellow"}
SIGN_ATTR = {
    4: "go straight",
    5: "turn left",
    6: "turn right",
    7: "no left turn",
    8: "no right turn",
    9: "u-turn",
    10: "no u-turn",
    11: "slight left",
    12: "slight right",
}
CAT_LIGHT = 1
CAT_SIGN = 2


def _box(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def convert(pkl_path, data_root, out_dir, window=4, min_side=12, test_frac=0.2, dev_frac=0.2, seed_mod=10):
    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)
    items = infos.values() if isinstance(infos, dict) else infos
    segs = defaultdict(list)
    for info in items:
        segs[str(info["segment_id"])].append(info)
    n = 0
    for seg, frames in segs.items():
        frames.sort(key=lambda x: x["timestamp"])
        if len(frames) < window:
            continue
        r = fold_of(seg, seed_mod) / seed_mod
        split = "test" if r < test_frac else ("dev" if r < test_frac + dev_frac else "train")
        for s in range(0, len(frames) - window + 1, window):
            win = frames[s : s + window]
            sc = scene_from_window(win, seg, s, split, data_root, min_side)
            if sc is not None:
                write_json(os.path.join(out_dir, sc["scene_id"] + ".json"), sc)
                n += 1
    return n


def scene_from_window(win, seg, start, split, data_root, min_side):
    last = win[-1]
    paths = []
    for info in win:
        paths.append({c: os.path.join(data_root, info["sensor"][CAM_NAMES[c]]["image_path"]) for c in CAMS})
    size = {c: list(Image.open(paths[-1][c]).size) for c in CAMS}
    ids = {}
    for te in last["annotation"]["traffic_element"]:
        b = _box(te["points"])
        if min(b[2] - b[0], b[3] - b[1]) < min_side:
            continue
        ids[te["id"]] = te
    lights_static, signs, frames = [], [], []
    for te_id, te in ids.items():
        if te["category"] == CAT_LIGHT:
            lights_static.append({"id": f"light_{te_id}", "lane_ids": []})
        elif te["category"] == CAT_SIGN and te["attribute"] in SIGN_ATTR:
            signs.append(
                {
                    "id": f"sign_{te_id}",
                    "meaning": SIGN_ATTR[te["attribute"]],
                    "lane_ids": [],
                    "cam": "center",
                    "box": _box(te["points"]),
                }
            )
    keep = {s["id"] for s in lights_static}
    for info, pth in zip(win, paths):
        fl = []
        for te in info["annotation"]["traffic_element"]:
            lid = f"light_{te['id']}"
            if lid in keep and te["attribute"] in LIGHT_ATTR:
                fl.append({"id": lid, "cam": "center", "box": _box(te["points"]), "status": LIGHT_ATTR[te["attribute"]]})
        frames.append({"images": pth, "lights": fl})
    cur = {x["id"] for x in frames[-1]["lights"]}
    lights_static = [s for s in lights_static if s["id"] in cur]
    if not lights_static and not signs:
        return None
    return {
        "scene_id": f"{seg}_{start}",
        "split": split,
        "image_size": size,
        "lanes_complete": False,
        "lanes": [],
        "lines": [],
        "vehicles": [],
        "signs": signs,
        "lights_static": lights_static,
        "frames": frames,
    }
