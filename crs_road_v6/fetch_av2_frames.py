import argparse
import bisect
import json
import os
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

from netutil import http_get

HOST = "https://argoverse.s3.us-east-1.amazonaws.com"
BASE = "datasets/av2/sensor/{split}/{log}/sensors/cameras/{cam}/"
CAMS = {"left": "ring_front_left", "center": "ring_front_center", "right": "ring_front_right"}
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_ts(prefix):
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        root = ET.fromstring(http_get(HOST + "/?" + urllib.parse.urlencode(q)))
        for c in root.findall(NS + "Contents"):
            key = c.find(NS + "Key").text
            if key.endswith(".jpg"):
                out.append(int(os.path.basename(key)[:-4]))
        if root.find(NS + "IsTruncated").text == "true":
            token = root.find(NS + "NextContinuationToken").text
        else:
            break
    return sorted(out)


def nearest(ts_sorted, t):
    i = bisect.bisect_left(ts_sorted, t)
    c = ts_sorted[max(0, i - 1) : i + 1]
    return min(c, key=lambda x: abs(x - t))


def build_plan(tc, tl, tr, nframes, stride, offset):
    last = offset + stride * (nframes - 1)
    if offset < 0 or last >= len(tc):
        raise SystemExit(f"need center index up to {last} but only {len(tc)} images")
    plan = []
    for f in range(nframes):
        t = tc[offset + stride * f]
        plan.append({"frame": f, "center": t, "left": nearest(tl, t), "right": nearest(tr, t)})
    return plan


def plan_from_meta(meta_path):
    with open(meta_path) as f:
        d = json.load(f)
    s = d["sensor"]
    n = min(len(s[name]) for name in CAMS.values())
    return [{"frame": f, **{c: int(s[name][f]) for c, name in CAMS.items()}} for f in range(n)]


def resolve_meta(meta_arg, log):
    if not meta_arg:
        return None
    if os.path.isdir(meta_arg):
        p = os.path.join(meta_arg, f"{log}.json")
        return p if os.path.exists(p) else None
    return meta_arg if os.path.exists(meta_arg) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default="av2_frames")
    ap.add_argument("--split", default="train")
    ap.add_argument("--meta", default=None)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--anchor_ts", type=int)
    ap.add_argument("--anchor_frame", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list_only", action="store_true")
    a = ap.parse_args()

    mp = resolve_meta(a.meta, a.log)
    if mp:
        plan = plan_from_meta(mp)[: a.nframes]
        print("plan from metadata:", mp, "frames:", len(plan))
    else:
        ts = {c: list_ts(BASE.format(split=a.split, log=a.log, cam=n)) for c, n in CAMS.items()}
        tc = ts["center"]
        offset = 0
        if a.anchor_ts is not None:
            if a.anchor_ts not in tc:
                raise SystemExit("anchor timestamp not found in center camera listing")
            idx = tc.index(a.anchor_ts)
            if a.anchor_frame is not None:
                offset = idx - a.stride * a.anchor_frame
        plan = build_plan(tc, ts["left"], ts["right"], a.nframes, a.stride, offset)

    dl = [abs(p["left"] - p["center"]) / 1e6 for p in plan]
    dr = [abs(p["right"] - p["center"]) / 1e6 for p in plan]
    print("max |left-center| ms: %.1f   max |right-center| ms: %.1f" % (max(dl), max(dr)))
    if a.list_only:
        return

    jobs = []
    for p in plan:
        for c, n in CAMS.items():
            url = f"{HOST}/{BASE.format(split=a.split, log=a.log, cam=n)}{p[c]}.jpg"
            dst = os.path.join(a.out, a.log, c, f"{p['frame']:02d}.jpg")
            jobs.append((url, dst))

    def get(j):
        url, dst = j
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(http_get(url))

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(get, jobs))
    with open(os.path.join(a.out, a.log, "frames.json"), "w") as f:
        json.dump(plan, f)
    print("saved", len(jobs), "images to", os.path.join(a.out, a.log))


if __name__ == "__main__":
    main()
