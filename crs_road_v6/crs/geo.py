import numpy as np

from .common import read_json

CAMN = {"center": "ring_front_center", "left": "ring_front_left", "right": "ring_front_right"}
DEFAULT_SIZE = {"center": (1550, 2048), "left": (2048, 1550), "right": (2048, 1550)}
OCC_TYPES = {"regular_vehicle", "large_vehicle", "box_truck", "truck", "truck_cab", "bus", "vehicular_trailer"}


def clip_box(b, size):
    w, h = size
    x1 = max(0.0, min(float(b[0]), float(b[2])))
    y1 = max(0.0, min(float(b[1]), float(b[3])))
    x2 = min(float(w), max(float(b[0]), float(b[2])))
    y2 = min(float(h), max(float(b[1]), float(b[3])))
    if x2 - x1 < 4.0 or y2 - y1 < 4.0:
        return None
    return [x1, y1, x2, y2]


def sig(v):
    return tuple(sorted(str(x) for x in (v or [])))


def _occ_frac(box, occluders):
    if not occluders:
        return 0.0
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    best = 0.0
    for ob in occluders:
        ix1, iy1 = max(box[0], ob[0]), max(box[1], ob[1])
        ix2, iy2 = min(box[2], ob[2]), min(box[3], ob[3])
        if ix2 > ix1 and iy2 > iy1:
            f = (ix2 - ix1) * (iy2 - iy1) / area
            if f > best:
                best = f
    return best


class Meta:
    def __init__(self, path, sizes=None):
        d = read_json(path)
        self.d = d
        self.sizes = {c: tuple(sizes[c]) if sizes and c in sizes else DEFAULT_SIZE[c] for c in CAMN}
        self.cal = {}
        for c, n in CAMN.items():
            k = d["calibration"][n]
            self.cal[c] = (
                float(k["fx"]),
                float(k["fy"]),
                float(k["cx"]),
                float(k["cy"]),
                np.array(k["R"], dtype=float),
                np.array(k["t"], dtype=float).reshape(3),
            )
        self.pose_t = np.array(d["pose"]["translations"], dtype=float)
        self.pose_R = np.array(d["pose"]["rotations"], dtype=float)
        self.te = {str(t["id"]): t for t in d.get("traffic_elements", [])}
        self.obj = {str(o["uuid"]): o for o in d.get("objects", [])}
        self.lines = d.get("lane_lines", {})
        self.by_sig = {}
        for k, v in self.lines.items():
            self.by_sig.setdefault((sig(v.get("left_of")), sig(v.get("right_of"))), []).append(k)
        self.ts = {c: [int(x) for x in d["sensor"][n]] for c, n in CAMN.items() if n in d.get("sensor", {})}
        self._occ = {}
        for o in d.get("objects", []):
            if o.get("type") not in OCC_TYPES:
                continue
            for c, n in CAMN.items():
                for fs, b in o.get("positions", {}).get(n, {}).items():
                    cb = clip_box(b, self.sizes[c])
                    if cb:
                        self._occ.setdefault((c, int(fs)), []).append(cb)
        self.cache = {}

    def te_box(self, uid, f):
        t = self.te.get(str(uid))
        if not t:
            return None
        p = t.get("points", {}).get(str(f))
        if not p:
            return None
        if isinstance(p[0], (int, float)):
            raw = p[:4]
        else:
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            raw = [min(xs), min(ys), max(xs), max(ys)]
        b = clip_box(raw, self.sizes["center"])
        return ("center", b) if b else None

    def obj_box(self, uid, f):
        o = self.obj.get(str(uid))
        if not o:
            return None
        best = None
        for c, n in CAMN.items():
            b = o.get("positions", {}).get(n, {}).get(str(f))
            if b:
                b = clip_box(b, self.sizes[c])
                if b:
                    a = (b[2] - b[0]) * (b[3] - b[1])
                    if best is None or a > best[0]:
                        best = (a, c, b)
        return (best[1], best[2]) if best else None

    def project(self, pts, f, cam):
        fx, fy, cx, cy, R, t = self.cal[cam]
        ego = (pts - self.pose_t[f]) @ self.pose_R[f]
        pc = ego @ R.T + t
        z = pc[:, 2]
        zs = np.where(z > 1e-3, z, np.nan)
        return fx * pc[:, 0] / zs + cx, fy * pc[:, 1] / zs + cy, z

    def find_line(self, uuid, left_of, right_of, mark):
        if uuid is not None and str(uuid) in self.lines:
            return str(uuid)
        c = self.by_sig.get((sig(left_of), sig(right_of)), [])
        if len(c) > 1 and mark is not None:
            c = [k for k in c if str(self.lines[k].get("mark_type", "")).upper() == str(mark).upper()]
        return c[0] if len(c) == 1 else None

    def line_box(self, key, f, d0=5.0, span=3.0):
        ck = (key, f)
        if ck in self.cache:
            return self.cache[ck]
        if key not in self.lines or f < 0 or f >= len(self.pose_t):
            self.cache[ck] = None
            return None
        g = np.array(self.lines[key].get("geometry", []), dtype=float)
        if g.ndim != 2 or len(g) < 2:
            self.cache[ck] = None
            return None
        dense = []
        for a, b in zip(g[:-1], g[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.25))
            dense.append(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None])
        pts = np.concatenate(dense)
        best = None
        for c in CAMN:
            u, v, z = self.project(pts, f, c)
            w, h = self.sizes[c]
            ok = (z > 2.0) & (z < 35.0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if ok.sum() < 3:
                continue
            zmin = float(z[ok].min())
            zmax = float(z[ok].max())
            occs = self._occ.get((c, f), [])
            starts = list(np.arange(max(zmin, d0), min(zmax, 28.0) + 0.1, 1.5))
            if not starts:
                starts = [zmin]
            for lo in starts:
                sel = ok & (z >= lo) & (z <= lo + span)
                if sel.sum() < 3:
                    continue
                b = [float(u[sel].min()) - 8.0, float(v[sel].min()) - 8.0, float(u[sel].max()) + 8.0, float(v[sel].max()) + 8.0]
                b = clip_box(b, (w, h))
                if not b:
                    continue
                occ = _occ_frac(b, occs)
                cam_pen = 0 if c == "center" else 1
                score = (1 if occ > 0.25 else 0, round(occ, 2), cam_pen, float(lo))
                if best is None or score < best[0]:
                    best = (score, c, b)
        res = (best[1], best[2]) if best else None
        self.cache[ck] = res
        return res
