import argparse
import glob
import os
import random

from PIL import Image, ImageDraw

from crs import graph
from crs.common import read_json

COL = {"light": (255, 0, 0), "sign": (0, 120, 255), "vehicle": (0, 200, 0), "line": (255, 160, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", default="overlays")
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.scenes, "*.json")))
    problems = 0
    rng = random.Random(0)
    tot = {"light": 0, "sign": 0, "vehicle": 0, "line": 0}
    for f in files:
        sc = read_json(f)
        fr = sc["frames"][-1]
        for c, p in fr["images"].items():
            if not os.path.exists(p):
                print("MISSING IMAGE", p)
                problems += 1
                continue
            if list(Image.open(p).size) != list(sc["image_size"][c]):
                print("SIZE MISMATCH", p, Image.open(p).size, sc["image_size"][c])
                problems += 1
        items = [("light", x["cam"], x["box"], x["id"] + ":" + x["status"]) for x in fr.get("lights", [])]
        items += [("sign", x["cam"], x["box"], x.get("meaning", "")) for x in sc.get("signs", [])]
        items += [("vehicle", x["cam"], x["box"], x.get("category", "")) for x in sc.get("vehicles", [])]
        items += [("line", x["cam"], x["box"], x["type"] + "_" + x.get("color", "")) for x in sc.get("lines", [])]
        for kind, cam, b, lab in items:
            tot[kind] += 1
            w, h = sc["image_size"][cam]
            if not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                print("BOX OUT OF IMAGE", sc["scene_id"], kind, b, (w, h))
                problems += 1
        if "lanes" in sc:
            qs = graph.gen_scene(sc, random.Random(0), "dev")
            print(sc["scene_id"], "questions:", len(qs), sorted({q["template"] for q in qs}))
    print("scenes:", len(files), "boxes:", tot, "problems:", problems)
    for f in rng.sample(files, min(a.n, len(files))):
        sc = read_json(f)
        fr = sc["frames"][-1]
        ims = {c: Image.open(p).convert("RGB") for c, p in fr["images"].items()}
        draws = {c: ImageDraw.Draw(im) for c, im in ims.items()}
        for lt in fr.get("lights", []):
            draws[lt["cam"]].rectangle(lt["box"], outline=COL["light"], width=3)
            draws[lt["cam"]].text((lt["box"][0], max(0, lt["box"][1] - 12)), lt["status"], fill=COL["light"])
        for kind, key in (("sign", "signs"), ("vehicle", "vehicles"), ("line", "lines")):
            for x in sc.get(key, []):
                draws[x["cam"]].rectangle(x["box"], outline=COL[kind], width=3)
                lab = x.get("type", x.get("category", x.get("meaning", "")))
                if lab:
                    draws[x["cam"]].text((x["box"][0], max(0, x["box"][1] - 12)), lab, fill=COL[kind])
        for c, im in ims.items():
            im.save(os.path.join(a.out, f"{sc['scene_id']}_{c}.jpg"))


if __name__ == "__main__":
    main()
