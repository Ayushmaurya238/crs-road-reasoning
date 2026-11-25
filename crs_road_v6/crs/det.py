import glob
import math
import os

from .common import iou_xyxy, read_json, write_json


def detect_scenes(scenes_dir, model_name="yolo11n.pt", conf=0.15, imgsz=1280, bs=8, device=None):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("ultralytics is required for `run.py detect`. Run: pip install -q ultralytics") from e

    files = sorted(glob.glob(os.path.join(scenes_dir, "*.json")))
    if not files:
        raise SystemExit(f"no scene json files found in {scenes_dir}")

    scenes = [(f, read_json(f)) for f in files]
    paths = sorted({sc["frames"][-1]["images"]["center"] for _, sc in scenes if "center" in sc["frames"][-1]["images"]})
    model = YOLO(model_name)
    cache = {}
    kw = {"classes": [9], "conf": conf, "imgsz": imgsz, "verbose": False}
    if device:
        kw["device"] = device

    for i in range(0, len(paths), bs):
        batch = paths[i : i + bs]
        results = model.predict(batch, **kw)
        for p, res in zip(batch, results):
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().tolist()
                confs = res.boxes.conf.cpu().tolist()
                for b, c in zip(xyxy, confs):
                    dets.append({"cam": "center", "box": [round(x, 2) for x in b], "conf": round(float(c), 4)})
            cache[p] = dets

    tot_det, gt_tot, gt_hit = 0, 0, 0
    for f, sc in scenes:
        fr = sc["frames"][-1]
        p = fr["images"].get("center")
        dets = cache.get(p, [])
        fr["det_lights"] = dets
        tot_det += len(dets)
        for lt in fr.get("lights", []):
            if lt.get("cam") != "center":
                continue
            gt_tot += 1
            if any(iou_xyxy(lt["box"], d["box"]) >= 0.3 for d in dets):
                gt_hit += 1
        write_json(f, sc)

    rec = gt_hit / gt_tot if gt_tot else float("nan")
    return {"scenes": len(scenes), "images": len(paths), "det_lights": tot_det, "gt_lights": gt_tot, "gt_recall@0.3": rec}


def snap_light_box(pred_box, sc, max_dist=0.30):
    fr = sc["frames"][-1]
    dets = fr.get("det_lights") or []
    if not dets:
        return None, pred_box, False
    pcx = 0.5 * (pred_box[0] + pred_box[2])
    pcy = 0.5 * (pred_box[1] + pred_box[3])
    best = None
    for d in dets:
        if isinstance(d, dict):
            cam = d.get("cam", "center")
            raw = d["box"]
        else:
            cam = "center"
            raw = d[:4]
        w, h = sc["image_size"][cam]
        nb = [max(0.0, min(1.0, raw[0] / w)), max(0.0, min(1.0, raw[1] / h)), max(0.0, min(1.0, raw[2] / w)), max(0.0, min(1.0, raw[3] / h))]
        dcx, dcy = 0.5 * (nb[0] + nb[2]), 0.5 * (nb[1] + nb[3])
        dist = math.hypot(pcx - dcx, pcy - dcy)
        if best is None or dist < best[0]:
            best = (dist, cam, nb)
    if best is not None and best[0] <= max_dist:
        return best[1], best[2], True
    return None, pred_box, False
