import os
import random

from PIL import Image, ImageDraw

from .common import CAMS, LIGHT, write_json

W, H = 320, 240
NFRAMES = 4
STATUS_RGB = {"red": (220, 30, 30), "green": (30, 200, 60), "yellow": (240, 200, 30)}
VEHICLE_RGB = {
    "red": (200, 40, 40),
    "blue": (40, 70, 200),
    "black": (30, 30, 30),
    "silver": (190, 190, 200),
    "white": (240, 240, 240),
}
LINE_RGB = {"white": (235, 235, 235), "yellow": (235, 200, 20)}
MEANINGS = ["no left turn", "no u-turn", "speed limit"]


def _clamp_box(b):
    return [max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])]


def _draw_line(d, x, ltype, color):
    if ltype == "none":
        return
    rgb = LINE_RGB[color]
    xs = [x - 3, x + 3] if ltype.startswith("double") else [x]
    for xx in xs:
        if ltype in ("solid", "double_solid"):
            d.line([(xx, 90), (xx, 230)], fill=rgb, width=3)
        else:
            for y in range(90, 230, 24):
                d.line([(xx, y), (xx, y + 14)], fill=rgb, width=3)


def _render_center(sc, t, rng_seed):
    r = random.Random(rng_seed)
    img = Image.new("RGB", (W, H), (130, 160, 200))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 90, W, H], fill=(85 + r.randint(0, 10), 88, 92))
    for ln in sc["lines"]:
        _draw_line(d, ln["x"], ln["type"], ln["color"])
    for v in sc["vehicles"]:
        b = v["box"]
        d.rectangle(b, fill=VEHICLE_RGB[v["color"]])
        d.rectangle([b[0] + 3, b[1] + 3, b[2] - 3, b[1] + 12], fill=(60, 80, 100))
    for s in sc["signs"]:
        d.rectangle(s["box"], fill=(250, 250, 250), outline=(200, 0, 0))
    for lt in sc["frames"][t]["lights"]:
        b = lt["box"]
        d.rectangle(b, fill=(20, 20, 20))
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for k, name in enumerate(LIGHT):
            yy = b[1] + 6 + k * 9
            on = name == lt["status"]
            col = STATUS_RGB[name] if on else (50, 50, 50)
            d.ellipse([cx - 3, yy - 3, cx + 3, yy + 3], fill=col)
    return img


def _render_side(seed):
    r = random.Random(seed)
    img = Image.new("RGB", (W, H), (120, 130, 120))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        x, y = r.randint(0, W - 40), r.randint(0, H - 40)
        d.rectangle([x, y, x + r.randint(10, 40), y + r.randint(10, 40)], fill=(r.randint(60, 180),) * 3)
    return img


def make_scene(sid, root, rng, split):
    n = rng.randint(2, 5)
    m = rng.randint(0, n - 1)
    ego = rng.randint(m, n - 1)

    def xk(k):
        return int(20 + k * (W - 40) / n)

    def xc(i):
        return (xk(i) + xk(i + 1)) / 2

    lanes = []
    for i in range(n):
        lanes.append(
            {
                "id": f"lane_{i}",
                "index": i,
                "direction": "opposite" if i < m else "same",
                "type": rng.choices(["vehicle", "bus", "bike"], [0.7, 0.15, 0.15])[0],
                "ego": i == ego,
            }
        )
    lines = []
    for k in range(n + 1):
        if k in (0, n):
            t, c = "solid", "white"
        elif k == m and m > 0:
            t, c = "double_solid", "yellow"
        elif k < m:
            t, c = "dashed", "yellow"
        else:
            t, c = rng.choice([("dashed", "white"), ("dashed", "white"), ("solid", "white"), ("double_dashed", "white")])
        if k not in (0, n) and rng.random() < 0.07:
            t, c = "none", "none"
        x = xk(k)
        lines.append(
            {
                "id": f"line_{k}",
                "index": k,
                "type": t,
                "color": c,
                "x": x,
                "cam": "center",
                "box": _clamp_box([x - 24, 116, x + 24, 164]),
                "left_of": f"lane_{k}" if k < n else None,
                "right_of": f"lane_{k - 1}" if k > 0 else None,
            }
        )
    vehicles = []
    for i in range(n):
        if rng.random() < 0.5:
            lw = xk(i + 1) - xk(i)
            vw = max(8, int(lw * 0.5))
            x1 = xc(i) - vw / 2
            vehicles.append(
                {
                    "id": f"veh_{len(vehicles)}",
                    "category": rng.choice(["car", "bus", "truck"]),
                    "color": rng.choice(list(VEHICLE_RGB)),
                    "lane_id": f"lane_{i}",
                    "cam": "center",
                    "box": _clamp_box([x1, 165, x1 + vw, 215]),
                }
            )
    same_lanes = [i for i in range(m, n)]
    light_lanes = rng.sample(same_lanes, k=min(len(same_lanes), rng.randint(0, 3)))
    lights_static, seqs = [], {}
    for j, i in enumerate(light_lanes):
        lid = f"light_{j}"
        base = rng.choice(LIGHT)
        if rng.random() < 0.35:
            c = rng.randint(1, NFRAMES - 1)
            other = rng.choice([s for s in LIGHT if s != base])
            seq = [base] * c + [other] * (NFRAMES - c)
        else:
            seq = [base] * NFRAMES
        seqs[lid] = (seq, _clamp_box([xc(i) - 6, 22, xc(i) + 6, 52]))
        lights_static.append({"id": lid, "lane_ids": [f"lane_{i}"]})
    signs = []
    if rng.random() < 0.5 and same_lanes:
        i = rng.choice(same_lanes)
        signs.append(
            {
                "id": "sign_0",
                "meaning": rng.choice(MEANINGS),
                "lane_ids": [f"lane_{i}"],
                "cam": "center",
                "box": _clamp_box([xc(i) - 9, 58, xc(i) + 9, 76]),
            }
        )
    sdir = os.path.join(root, "images", sid)
    os.makedirs(sdir, exist_ok=True)
    frames = []
    sc = {
        "scene_id": sid,
        "split": split,
        "image_size": {c: [W, H] for c in CAMS},
        "lanes_complete": True,
        "lanes": lanes,
        "lines": lines,
        "vehicles": vehicles,
        "signs": signs,
        "lights_static": lights_static,
        "frames": [],
    }
    for t in range(NFRAMES):
        fl = [
            {"id": lid, "cam": "center", "box": box, "status": seq[t]}
            for lid, (seq, box) in seqs.items()
        ]
        frames.append({"images": {}, "lights": fl})
    sc["frames"] = frames
    seed0 = rng.randint(0, 10**9)
    for t in range(NFRAMES):
        for c in CAMS:
            p = os.path.abspath(os.path.join(sdir, f"f{t}_{c}.png"))
            img = _render_center(sc, t, seed0 + t) if c == "center" else _render_side(seed0 + t * 7 + len(c))
            img.save(p)
            frames[t]["images"][c] = p
    return sc


def generate(root, n_train, n_dev, n_test, seed=0):
    rng = random.Random(seed)
    sdir = os.path.join(root, "scenes")
    os.makedirs(sdir, exist_ok=True)
    counts = 0
    for split, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        for i in range(n):
            sid = f"{split}_{i:04d}"
            write_json(os.path.join(sdir, sid + ".json"), make_scene(sid, root, rng, split))
            counts += 1
    return counts
