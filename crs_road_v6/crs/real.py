import glob
import hashlib
import os
import random
from collections import Counter

from PIL import Image

from .common import CAMS, LCOLOR, LIGHT, LTYPE, read_json, write_json, write_jsonl
from .geo import Meta
from .graph import FILL, HEAVY, NOTARGET, SMALL, cap, mk_options, valid

TW = {
    "lane": "lane", "traffic_light": "traffic light", "sign": "sign", "regular_vehicle": "vehicle",
    "large_vehicle": "large vehicle", "box_truck": "box truck", "truck": "truck", "truck_cab": "truck",
    "bus": "bus", "bicyclist": "bicyclist", "pedestrian": "pedestrian", "motorcycle": "motorcycle",
    "construction_cone": "construction cone", "lane_line": "lane line", "crossing": "pedestrian crossing",
    "intersection": "intersection", "stop_sign": "stop sign", "split": "lane split",
    "road_marking": "road marking", "ego": "ego vehicle",
}
ACTORS = {"regular_vehicle", "large_vehicle", "box_truck", "truck", "truck_cab", "bus"}
OBJ_TYPES = ACTORS | {"pedestrian", "bicyclist", "motorcycle", "bicycle", "wheeled_device", "vehicular_trailer"}
DIRN = {"same": "Same direction as the ego.", "opposite": "Opposite direction to the ego."}
STATUSES = ["red", "green", "yellow"]
STYLES = ["white outlined rectangle", "zebra crossing", "ladder crossing", "unmarked crossing"]
LANE_STD = ["standard vehicle lane", "bike lane", "bus lane", "parking lane", "turn lane"]


def parse_mark(m):
    m = str(m).upper()
    if m == "NONE":
        return "no visible line", "not visible"
    col = "yellow" if "YELLOW" in m else "white" if "WHITE" in m else "blue" if "BLUE" in m else None
    if col is None:
        return None
    if "DOUBLE" in m:
        t = "double dashed" if "DASH" in m else "double solid"
    elif "SOLID" in m and "DASH" in m:
        t = "solid and dashed"
    elif "DASH" in m:
        t = "dashed"
    elif "SOLID" in m:
        t = "solid"
    else:
        return None
    return t, col


def mark_keys(pm):
    t, c = pm
    tk = "none" if t == "no visible line" else t.replace(" ", "_")
    ck = "none" if c == "not visible" else c
    return tk, ck


def lane_label(v):
    v = str(v).strip()
    if v.upper() in ("VEHICLE", "STANDARD"):
        return "standard vehicle lane"
    if v.upper() == "BIKE":
        return "bike lane"
    if v.upper() == "BUS":
        return "bus lane"
    s = v.lower()
    return s if "lane" in s else s.replace("vehicle", "vehicle lane", 1)


def safe_id(s):
    return str(s).replace("/", "_").replace(" ", "_")


class D:
    __slots__ = ("text", "depth", "chain", "uses", "labs")

    def __init__(self, text, depth, chain, uses, labs=frozenset()):
        self.text, self.depth, self.chain, self.uses, self.labs = text, depth, chain, frozenset(uses), frozenset(labs)


class FG:
    def __init__(self, d, f):
        self.f = str(f)
        self.meta = d.get("_meta")
        self._bx = {}
        self.nodes = {n["label"]: n for n in d["nodes"]}
        self.vis = {l for l, n in self.nodes.items() if self.visible(n)}
        self.edges, self.eid = [], {}
        for e in d["edges"]:
            t = e["temporalLabels"]
            lab = t if isinstance(t, str) else t.get(self.f) if isinstance(t, dict) else (t[0] if t else None)
            if lab and e["source"] in self.vis and e["target"] in self.vis:
                x = dict(src=e["source"], lab=lab, tgt=e["target"], id=e["label"])
                self.edges.append(x)
                self.eid[x["id"]] = x
        self.uuid2lane = {str(n["helpers"].get("uuid")): l for l, n in self.nodes.items() if n["type"] == "lane"}
        self.left = self._left_closure()

    def box(self, l):
        if self.meta is None:
            return None
        if l in self._bx:
            return self._bx[l]
        n = self.nodes[l]
        t, f, res = n["type"], int(self.f), None
        h = n.get("helpers", {})
        u = h.get("uuid")
        try:
            if t == "traffic_light" and u is not None:
                res = self.meta.te_box(u, f)
            elif t == "sign" and u is not None:
                res = self.meta.te_box(u, f) or self.meta.obj_box(u, f)
            elif t in OBJ_TYPES and u is not None:
                res = self.meta.obj_box(u, f)
            elif t == "lane_line":
                k = self.meta.find_line(u, h.get("left_of"), h.get("right_of"), self.prop(l, "mark_type"))
                res = self.meta.line_box(k, f) if k else None
        except Exception:
            res = None
        self._bx[l] = res
        return res

    def tgt(self, l, kind, **lab):
        b = self.box(l)
        if b is None:
            return dict(NOTARGET)
        cam, bx = b
        w, h = self.meta.sizes[cam]
        return {
            "kind": kind,
            "cam": cam,
            "box": [bx[0] / w, bx[1] / h, bx[2] / w, bx[3] / h],
            "size_px": float(((bx[2] - bx[0]) * (bx[3] - bx[1])) ** 0.5),
            "node": l,
            "kind_z": kind if kind in ("light", "line") and lab else "none",
            **lab,
        }

    def visible(self, n):
        if n["type"] == "ego":
            return True
        v = n.get("visibleAt")
        return bool(v.get(self.f)) if isinstance(v, dict) else False

    def typ(self, l):
        return self.nodes[l]["type"]

    def ids(self, typ):
        return sorted(l for l in self.vis if self.typ(l) == typ)

    def prop(self, l, k):
        v = self.nodes[l].get("properties", {}).get(k)
        return v.get(self.f) if isinstance(v, dict) else v

    def tw(self, l):
        t = self.typ(l)
        if t == "traffic_light" and self.prop(l, "type") == "pedestrian":
            return "pedestrian traffic light"
        return TW.get(t, t.replace("_", " "))

    def complete(self, l):
        return bool(self.nodes[l]["helpers"].get("is_complete", {}).get(self.f))

    def lane_of(self, l):
        for e in self.edges:
            if e["src"] == l and e["lab"] == "is in" and self.typ(e["tgt"]) == "lane":
                return e["tgt"]
        c = self.nodes[l]["helpers"].get("context", {}).get(self.f)
        x = self.uuid2lane.get(str(c)) if c is not None else None
        return x if x in self.vis else None

    def _left_closure(self):
        lt = set()
        for e in self.edges:
            if self.typ(e["src"]) == "lane" and self.typ(e["tgt"]) == "lane":
                if e["lab"] == "is left of":
                    lt.add((e["src"], e["tgt"]))
                elif e["lab"] == "is right of":
                    lt.add((e["tgt"], e["src"]))
        ch = True
        while ch:
            ch = False
            for a, b in list(lt):
                for c, d in list(lt):
                    if b == c and (a, d) not in lt:
                        lt.add((a, d))
                        ch = True
        return lt

    def prop_text(self, l, k):
        v = self.prop(l, k)
        if v in (None, ""):
            return None
        v = str(v)
        t = self.tw(l)
        if k == "description":
            return f"the {v}"
        if k == "meaning":
            return f"the {t} with meaning '{v}'"
        if k == "location":
            return f"the {t} located at {v}"
        if k == "mark_type":
            return f"the lane line with marking {v.lower().replace('_', ' ')}"
        if k == "turn_direction":
            return f"the lane whose turn direction is {v}"
        if k == "shape":
            return f"the {t} with {v} shape"
        return f"the {t} whose {k.replace('_', ' ')} is {v}"

    def descs(self, l, depth, ban=frozenset()):
        n = self.nodes[l]
        if n["type"] == "ego":
            return [D("the ego vehicle", 0, ["Identify the ego vehicle"], {l})]
        out = []
        uks = n["helpers"].get("uniqueKey", {}).get(self.f, [])
        if not uks and n["type"] in ("traffic_light", "crossing") and len(self.ids(n["type"])) == 1:
            t = f"the {self.tw(l)}"
            return [D(t, 0, [f"Identify {t}"], {l})]
        for a in uks:
            if a == "is unique":
                t = f"the {self.tw(l)}"
                out.append(D(t, 0, [f"Identify {t}"], {l}))
            elif a in n.get("properties", {}):
                t = self.prop_text(l, a)
                if t:
                    out.append(D(t, 0, [f"Identify {t}"], {l}))
            elif a in self.eid and depth > 0:
                e = self.eid[a]
                if e["src"] == l:
                    other, fmt = e["tgt"], lambda od: f"the {self.tw(l)} that {e['lab']} {od}"
                elif e["tgt"] == l:
                    other, fmt = e["src"], lambda od: f"the {self.tw(l)} that {od} {e['lab']}"
                else:
                    continue
                if other in ban:
                    continue
                for od in self.descs(other, depth - 1, ban | {l}):
                    t = fmt(od.text)
                    out.append(D(t, od.depth + 1, od.chain + [f"Resolve {t}"], od.uses | {l}, od.labs | {e["lab"]}))
        return out

    def pick(self, l, rng, md=2, bad_nodes=(), bad_types=(), bad_labs=(), bad_text=()):
        ds = [
            d for d in self.descs(l, md)
            if not (set(bad_nodes) & d.uses)
            and not (set(bad_labs) & d.labs)
            and not any(self.typ(u) in bad_types for u in d.uses if u != l)
            and not any(b in d.text for b in bad_text)
        ]
        return rng.choice(ds) if ds else None

    def entities(self):
        lights, lines, signs, vehicles = [], [], [], []
        for l in self.ids("traffic_light"):
            st = self.prop(l, "status")
            b = self.box(l)
            if st in LIGHT and b is not None:
                uid = safe_id(self.nodes[l].get("helpers", {}).get("uuid") or l)
                lights.append({"id": uid, "cam": b[0], "box": b[1], "status": st})
        for l in self.ids("lane_line"):
            pm = parse_mark(self.prop(l, "mark_type"))
            b = self.box(l)
            if pm and b is not None:
                tk, ck = mark_keys(pm)
                if tk in LTYPE and ck in LCOLOR:
                    lines.append({"id": safe_id(l), "cam": b[0], "box": b[1], "type": tk, "color": ck})
        for l in self.ids("sign"):
            b = self.box(l)
            if b is not None:
                uid = safe_id(self.nodes[l].get("helpers", {}).get("uuid") or l)
                signs.append({"id": uid, "cam": b[0], "box": b[1], "meaning": str(self.prop(l, "meaning") or "")})
        for l in sorted(self.vis):
            if self.typ(l) in ACTORS:
                b = self.box(l)
                if b is not None:
                    vehicles.append({"id": safe_id(l), "cam": b[0], "box": b[1], "category": self.typ(l)})
        return lights, lines, signs, vehicles


def parts(ds, tsent, correct):
    out, k = [], 1
    for ci, d in enumerate(ds):
        for j, s in enumerate(d.chain):
            out.append([f"Step {k}: {s}.\n", 2.0 if ci == 0 and j == 0 else 1.0])
            k += 1
    out.append([f"Step {k}: {tsent}\n", 2.0])
    out.append([f"Conclusion: the correct answer is: {correct}", 1.0])
    return out


def mk(name, question, correct, pool, ds, tsent, rng, tgt=None):
    o = mk_options(correct, pool, rng)
    if o is None:
        return None
    opts, ai = o
    if tgt and tgt.get("box") is not None and tgt.get("cam") is not None:
        b = [int(round(x * 1000)) for x in tgt["box"]]
        tsent = f"Locate the target in the {tgt['cam'].upper()} view at <box>({b[0]},{b[1]},{b[2]},{b[3]})</box>. {tsent}"
    return {
        "template": name,
        "question": question,
        "options": opts,
        "answer": ai,
        "answer_text": correct,
        "cot_parts": parts(ds, tsent, correct),
        "depth": max([d.depth for d in ds] + [0]) + (1 if name in HEAVY else 0),
        "target": tgt or dict(NOTARGET),
        "reasoning": "heavy" if name in HEAVY else "perception",
        "small": name in SMALL,
    }


def join(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def t_lane_direction(g, rng):
    c = [l for l in g.ids("lane") if g.prop(l, "direction") in DIRN]
    if not c:
        return None
    l = rng.choice(c)
    d = g.pick(l, rng, bad_text=("ego",))
    if d is None:
        return None
    a = DIRN[g.prop(l, "direction")]
    return mk("lane_direction", f"What is the direction of {d.text}?", a, list(DIRN.values()), [d], f"That lane has direction: {a}", rng)


def t_lane_type(g, rng):
    c = [l for l in g.ids("lane") if g.prop(l, "type")]
    if not c:
        return None
    l = rng.choice(c)
    d = g.pick(l, rng)
    if d is None:
        return None
    a = lane_label(g.prop(l, "type"))
    return mk("lane_type", f"What lane is {d.text}?", a, LANE_STD, [d], f"That lane is: {a}", rng)


def line_target(g, rng):
    c = []
    for e in g.edges:
        if g.typ(e["src"]) == "lane_line" and g.typ(e["tgt"]) == "lane" and e["lab"] in ("is left marking of", "is right marking of"):
            side = "left" if "left" in e["lab"] else "right"
            same = [x for x in g.edges if x["tgt"] == e["tgt"] and x["lab"] == e["lab"] and g.typ(x["src"]) == "lane_line"]
            pm = parse_mark(g.prop(e["src"], "mark_type"))
            if len(same) == 1 and pm:
                c.append((e["src"], e["tgt"], side, pm))
    if not c:
        return None
    ln, lane, side, pm = rng.choice(c)
    d = g.pick(lane, rng, bad_types=("lane_line",))
    return (d, side, pm, ln) if d else None


def t_line(kind):
    def f(g, rng):
        r = line_target(g, rng)
        if r is None:
            return None
        d, side, (t, c), ln = r
        tk, ck = mark_keys((t, c))
        if tk in LTYPE and ck in LCOLOR:
            tg = g.tgt(ln, "line", ltype=LTYPE.index(tk), lcolor=LCOLOR.index(ck))
        else:
            tg = g.tgt(ln, "line")
        if kind == "line_marking":
            a = "no visible line" if t == "no visible line" else f"{t} {c}"
            pool = ["no visible line"] + [f"{x} {y}" for x in ("solid", "dashed", "double solid", "double dashed", "solid and dashed") for y in ("white", "yellow")]
            noun = "marking"
        elif kind == "line_type":
            a, noun = t, "type"
            pool = ["solid", "dashed", "double solid", "double dashed", "solid and dashed", "no visible line"]
        else:
            a, noun = c, "color"
            pool = ["white", "yellow", "blue", "not visible"]
        return mk(kind, f"What is the {side} lane line {noun} of {d.text}?", a, pool, [d], f"The {side} lane line of that lane reads: {a}", rng, tg)
    return f


def t_light_status(g, rng):
    c = [l for l in g.ids("traffic_light") if g.prop(l, "status") in STATUSES]
    if not c:
        return None
    l = rng.choice(c)
    d = g.pick(l, rng)
    if d is None:
        return None
    a = g.prop(l, "status")
    return mk("traffic_light_status", f"What is the status of {d.text}?", a, STATUSES + ["off"], [d], f"The light shows: {a}", rng, g.tgt(l, "light", light=LIGHT.index(a)))


def make_change(d_json, f, rng):
    if f < 3:
        return None
    gs = [FG(d_json, f - k) for k in (3, 2, 1, 0)]
    g = gs[-1]
    c = []
    for l in g.ids("traffic_light"):
        seq = [x.prop(l, "status") if l in x.vis else None for x in gs]
        if all(s in STATUSES for s in seq):
            c.append((l, seq))
    if not c:
        return None
    l, seq = rng.choice(c)
    a, b = seq[0], seq[-1]
    if len(set(seq)) == 1:
        ans = f"No, it stayed {a}"
    elif a != b:
        ans = f"Yes, it was {a} but switched to {b}"
    else:
        return None
    d = g.pick(l, rng)
    if d is None:
        return None
    pool = (
        [f"No, it stayed {s}" for s in STATUSES]
        + [f"Yes, it was {a} but switched to {s}" for s in STATUSES if s != a]
        + [f"Yes, it was {s} but switched to {a}" for s in STATUSES if s != a]
    )
    return mk("traffic_light_change", f"Did the status of {d.text} change over the last 4 frames?", ans, pool, [d], f"Over the four frames the light reads: {', '.join(seq)}", rng, g.tgt(l, "light", light=LIGHT.index(seq[-1])))


def lanes_off_intersection(g):
    ls = g.ids("lane")
    if not ls or not all(g.complete(l) for l in ls):
        return None
    if any(g.typ(e["src"]) == "lane" and g.typ(e["tgt"]) == "intersection" for e in g.edges):
        return None
    if any(g.typ(e["tgt"]) == "intersection" and g.typ(e["src"]) == "ego" for e in g.edges):
        return None
    return ls


def t_counting_generic(g, rng):
    ls = lanes_off_intersection(g)
    if not ls:
        return None
    same = rng.random() < 0.5
    k = sum(1 for l in ls if (g.prop(l, "direction") == "same") == same)
    where = "in the ego direction" if same else "in the direction opposite to the ego"
    fmt = lambda n: f"There are no lanes {where}." if n == 0 else f"There {'is' if n == 1 else 'are'} {n} lane{'' if n == 1 else 's'} {where}."
    pool = [fmt(n) for n in {max(k - 1, 0), k + 1, k + 2, max(k - 2, 0)} if n != k]
    d = D("the road", 0, ["Count every visible lane on the road ahead"], set())
    return mk("counting_generic", f"How many lanes are there {where}?", fmt(k), pool, [d], f"The count {where} is {k}", rng)


def t_counting_per_direction(g, rng):
    ls = lanes_off_intersection(g)
    if not ls:
        return None
    a = sum(1 for l in ls if g.prop(l, "direction") == "same")
    b = len(ls) - a
    fmt = lambda x, y: f"{x} lane{'' if x == 1 else 's'} in the ego direction and {y} lane{'' if y == 1 else 's'} in the opposite direction."
    pool = [fmt(max(a + p, 0), max(b + q, 0)) for p, q in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))]
    d = D("the road", 0, ["Count every visible lane on the road ahead"], set())
    return mk("counting_per_direction", "How many lanes are there?", fmt(a, b), pool, [d], f"Ego direction {a}, opposite direction {b}", rng)


def t_pair_lane_direction(g, rng):
    ls = [l for l in g.ids("lane") if g.prop(l, "direction") in DIRN]
    if len(ls) < 2:
        return None
    l1, l2 = rng.sample(ls, 2)
    d1, d2 = g.pick(l1, rng, bad_text=("ego",)), g.pick(l2, rng, bad_text=("ego",))
    if d1 is None or d2 is None:
        return None
    O = [
        "Yes, both are in the ego direction.",
        "Yes, both are opposite to the ego direction.",
        "No, the first is in the ego direction and the second is opposite.",
        "No, the first is opposite to the ego direction and the second is in the ego direction.",
    ]
    s1, s2 = g.prop(l1, "direction") == "same", g.prop(l2, "direction") == "same"
    a = O[0] if s1 and s2 else O[1] if not s1 and not s2 else O[2] if s1 else O[3]
    return mk("pairwise_lane_direction", f"Do {d1.text} and {d2.text} share the same direction?", a, O, [d1, d2], f"First lane: {g.prop(l1, 'direction')}, second lane: {g.prop(l2, 'direction')}", rng)


def actor_list(g):
    res = []
    for l in sorted(g.vis):
        if g.typ(l) in ACTORS or g.typ(l) == "ego":
            ln = g.lane_of(l)
            if ln:
                res.append((l, ln))
    return res


def t_pair_vehicle_location(g, rng):
    A = actor_list(g)
    if len(A) < 2:
        return None
    (v1, l1), (v2, l2) = rng.sample(A, 2)
    d1, d2 = g.pick(v1, rng, bad_labs=("is in",)), g.pick(v2, rng, bad_labs=("is in",))
    if d1 is None or d2 is None:
        return None
    same = "Yes, they are in the same lane."
    left = "No, the first is in a lane left of the second's lane."
    right = "No, the first is in a lane right of the second's lane."
    if l1 == l2:
        a = same
    elif (l1, l2) in g.left:
        a = left
    elif (l2, l1) in g.left:
        a = right
    else:
        return None
    return mk("pairwise_vehicle_location", f"Are {d1.text} and {d2.text} in the same lane?", a, [same, left, right], [d1, d2], "I locate the lane of each and compare them", rng)


def lane_texts(g, rng, ban_nodes, exclude=()):
    out = {}
    for l in g.ids("lane"):
        if l in exclude:
            continue
        d = g.pick(l, rng, bad_nodes=ban_nodes)
        if d is not None:
            out[l] = d
    return out


def t_vehicle_position(g, rng):
    A = actor_list(g)
    if not A:
        return None
    v, lane = rng.choice(A)
    dv = g.pick(v, rng, bad_labs=("is in",))
    if dv is None:
        return None
    T = lane_texts(g, rng, {v})
    if lane not in T:
        return None
    a = cap(T[lane].text) + "."
    pool = [cap(d.text) + "." for l, d in T.items() if l != lane]
    return mk("vehicle_position", f"Which lane is {dv.text} in?", a, pool, [dv], f"That actor is in {T[lane].text}", rng, g.tgt(v, "vehicle"))


def t_controls(typ, name):
    def f(g, rng):
        c = []
        for s in g.ids(typ):
            S = sorted({e["tgt"] for e in g.edges if e["src"] == s and e["lab"] == "controls" and g.typ(e["tgt"]) == "lane"})
            if 1 <= len(S) <= 3:
                c.append((s, S))
        if not c:
            return None
        s, S = rng.choice(c)
        d = g.pick(s, rng, bad_labs=("controls",))
        if d is None:
            return None
        T = lane_texts(g, rng, {s})
        if any(x not in T for x in S):
            return None
        rest = [x for x in T if x not in S]
        pool = []
        for x in rest:
            pool += [(x,), tuple(sorted(set(S) | {x}))]
        if len(S) > 1:
            pool += [tuple(S[:-1]), tuple(S[1:])]
        render = lambda xs: join([T[x].text for x in xs])
        a = cap(render(S)) + "."
        decoys = [cap(render(p)) + "." for p in dict.fromkeys(pool) if tuple(p) != tuple(S)]
        tg = g.tgt(s, "sign" if typ == "sign" else "light_ctx")
        return mk(name, f"Which lane(s) does {d.text} control?", a, decoys, [d], f"It controls {render(S)}", rng, tg)
    return f


def t_crossing_type(g, rng):
    c = [l for l in g.ids("crossing") if g.prop(l, "marking_style")]
    if not c:
        return None
    l = rng.choice(c)
    d = g.pick(l, rng)
    if d is None:
        return None
    a = str(g.prop(l, "marking_style"))
    return mk("crossing_type", f"How is {d.text} marked?", a, STYLES, [d], f"The marking style is: {a}", rng)


def t_counting_crossing(g, rng):
    cs = g.ids("crossing")
    if not cs or not all(g.complete(c) for c in cs):
        return None
    k = len(cs)
    fmt = lambda n: f"There {'is' if n == 1 else 'are'} {n} pedestrian crossing{'' if n == 1 else 's'}."
    pool = [fmt(n) for n in {max(k - 1, 1), k + 1, k + 2, max(k - 2, 1)} if n != k]
    d = D("the intersection", 0, ["Count every visible pedestrian crossing"], set())
    return mk("counting_crossing", "How many pedestrian crossings are there?", fmt(k), pool, [d], f"The count is {k}", rng)


TEMPLATES = {
    "lane_direction": t_lane_direction,
    "lane_type": t_lane_type,
    "line_marking": t_line("line_marking"),
    "line_type": t_line("line_type"),
    "line_color": t_line("line_color"),
    "traffic_light_status": t_light_status,
    "counting_generic": t_counting_generic,
    "counting_per_direction": t_counting_per_direction,
    "pairwise_lane_direction": t_pair_lane_direction,
    "pairwise_vehicle_location": t_pair_vehicle_location,
    "vehicle_position": t_vehicle_position,
    "sign_controls_lane": t_controls("sign", "sign_controls_lane"),
    "traffic_light_controls_lane": t_controls("traffic_light", "traffic_light_controls_lane"),
    "crossing_type": t_crossing_type,
    "counting_crossing": t_counting_crossing,
}


def frame_questions(d, f, rng, tries=6, per=2):
    g = FG(d, f)
    out = {}
    for name, fn in TEMPLATES.items():
        got = {}
        for _ in range(tries):
            q = fn(g, rng)
            if q is not None and valid(q):
                got[(q["question"], tuple(sorted(q["options"])))] = q
            if len(got) >= per:
                break
        out[name] = list(got.values())
    got = {}
    for _ in range(tries):
        q = make_change(d, f, rng)
        if q is not None and valid(q):
            got[(q["question"], tuple(sorted(q["options"])))] = q
    out["traffic_light_change"] = list(got.values())[:per]
    return out, g


def rr(cands, k, rng):
    by = {}
    for q in cands:
        by.setdefault(q["template"], {}).setdefault(q["answer_text"], []).append(q)
    pools = {}
    for t, dd in by.items():
        for v in dd.values():
            rng.shuffle(v)
        order, pool = list(dd.values()), []
        while any(order):
            for v in order:
                if v:
                    pool.append(v.pop())
        pools[t] = pool[::-1]
    out, names = [], list(pools)
    while len(out) < k and any(pools.values()):
        rng.shuffle(names)
        for t in names:
            if pools[t] and len(out) < k:
                out.append(pools[t].pop())
    return out


def assign_splits(logs, ratios, seed):
    logs = sorted(logs)
    random.Random(seed).shuffle(logs)
    n = len(logs)
    a, b, c = ratios
    tot = a + b + c
    nt, nd = max(1, round(n * a / tot)), round(n * b / tot)
    if n == 1:
        return {logs[0]: "train"}
    if n == 2:
        return {logs[0]: "train", logs[1]: "dev"}
    if n == 3:
        return {logs[0]: "train", logs[1]: "dev", logs[2]: "test"}
    out = {}
    for i, l in enumerate(logs):
        out[l] = "train" if i < nt else "dev" if i < nt + nd else "test"
    return out


def assign_cv(logs, fold, nfolds, n_dev, seed):
    """K-fold by log: every log is test in exactly one fold."""
    logs = sorted(logs)
    random.Random(seed).shuffle(logs)
    if not 0 <= fold < nfolds:
        raise SystemExit(f"cv_fold must be in [0, {nfolds - 1}]")
    chunks = [logs[i::nfolds] for i in range(nfolds)]
    test = set(chunks[fold])
    dev = set(chunks[(fold + 1) % nfolds][:n_dev])
    return {l: "test" if l in test else "dev" if l in dev else "train" for l in logs}


def pick_eval(cands, k, rng):
    """k questions per template, preferring distinct frames and distinct question text."""
    out = []
    for t in sorted({q["template"] for q in cands}):
        pool = [q for q in cands if q["template"] == t]
        rng.shuffle(pool)
        sel, used_f, seen = [], set(), set()
        for q in pool:
            if q["_f"] in used_f or q["question"] in seen:
                continue
            sel.append(q); used_f.add(q["_f"]); seen.add(q["question"])
            if len(sel) >= k:
                break
        if len(sel) < k:
            ids = {id(q) for q in sel}
            for q in pool:
                if id(q) in ids or q["question"] in seen:
                    continue
                sel.append(q); seen.add(q["question"])
                if len(sel) >= k:
                    break
        out.extend(sel)
    return out


def _detect_meta_dir(graphs_dir, meta_dir):
    if meta_dir:
        return meta_dir
    cand = os.path.join(os.path.dirname(os.path.abspath(graphs_dir)), "metadata")
    return cand if os.path.isdir(cand) else None


def _write_scene(sp, sid, log, split, f, imgs, fg):
    if os.path.exists(sp):
        return
    size = {c: list(Image.open(p).size) for c, p in imgs.items()}
    lights, lines, signs, vehicles = fg.entities()
    write_json(
        sp,
        {
            "scene_id": sid,
            "log": log,
            "split": split,
            "frame": f,
            "image_size": size,
            "frames": [{"images": imgs, "lights": lights}],
            "lines": lines,
            "signs": signs,
            "vehicles": vehicles,
        },
    )


def convert(graphs_dir, frames_dir, out_dir, ratios=(50, 15, 15), max_per_log=300, seed=0, nframes=32, meta_dir=None, limit_logs=0, cv_fold=None, cv_folds=4, n_dev=8, eval_per_template=1):
    files = sorted(glob.glob(os.path.join(graphs_dir, "*.json")))
    if limit_logs and limit_logs > 0:
        files = files[:limit_logs]
    if not files:
        raise SystemExit(f"no graph json files found in {graphs_dir}")
    meta_dir = _detect_meta_dir(graphs_dir, meta_dir)
    stems = [os.path.basename(f)[:-5] for f in files]
    splits = assign_cv(stems, cv_fold, cv_folds, n_dev, seed) if cv_fold is not None else assign_splits(stems, ratios, seed)
    sc_dir = os.path.join(out_dir, "scenes")
    os.makedirs(sc_dir, exist_ok=True)
    rows_by = {"train": [], "dev": [], "test": []}
    report = []
    for fpath in files:
        d = read_json(fpath)
        stem = os.path.basename(fpath)[:-5]
        log = d.get("logid") or stem
        split = splits[stem]
        rng = random.Random(int(hashlib.md5(f"{seed}{log}".encode()).hexdigest()[:8], 16))

        first_sizes = None
        for f in range(nframes):
            imgs = {c: os.path.join(frames_dir, log, c, f"{f:02d}.jpg") for c in CAMS}
            if all(os.path.exists(p) for p in imgs.values()):
                first_sizes = {c: list(Image.open(p).size) for c, p in imgs.items()}
                break

        has_meta = False
        if meta_dir:
            mp = os.path.join(meta_dir, f"{log}.json")
            if not os.path.exists(mp):
                mp = os.path.join(meta_dir, f"{stem}.json")
            if os.path.exists(mp):
                d["_meta"] = Meta(mp, first_sizes)
                has_meta = True

        cands, missing, fgs, frame_imgs = [], 0, {}, {}
        for f in range(nframes):
            imgs = {c: os.path.join(frames_dir, log, c, f"{f:02d}.jpg") for c in CAMS}
            if not all(os.path.exists(p) for p in imgs.values()):
                missing += 1
                continue
            qmap, fg = frame_questions(d, f, rng, per=2 if split == "train" else 4)
            fgs[f] = fg
            frame_imgs[f] = imgs
            for qs in qmap.values():
                for q in qs:
                    q["_f"], q["_imgs"] = f, imgs
                    cands.append(q)

        if has_meta and split in ("train", "dev"):
            for f, fg in fgs.items():
                sid = f"{log}__f{f:02d}"
                _write_scene(os.path.join(sc_dir, sid + ".json"), sid, log, split, f, frame_imgs[f], fg)

        if split == "train":
            chosen = rr(cands, max_per_log, rng)
        else:
            chosen = pick_eval(cands, eval_per_template, rng)

        for i, q in enumerate(chosen):
            f, imgs = q.pop("_f"), q.pop("_imgs")
            sid = f"{log}__f{f:02d}"
            _write_scene(os.path.join(sc_dir, sid + ".json"), sid, log, split, f, imgs, fgs[f])
            q.update(qid=f"{sid}_{i}", scene_id=sid, log=log, split=split, frame=f)
            rows_by[split].append(q)
        report.append((log[:8], split, len(chosen), missing, int(has_meta)))

    stats = {}
    for split, rows in rows_by.items():
        random.Random(seed).shuffle(rows)
        for i, q in enumerate(rows):
            sh = (i % 4 - q["answer"]) % 4
            o = q["options"]
            q["options"] = [o[(j - sh) % 4] for j in range(4)]
            q["answer"] = q["options"].index(q["answer_text"])
        if rows:
            write_jsonl(os.path.join(out_dir, f"qa_{split}.jsonl"), rows)
        stats[split] = {
            "n": len(rows),
            "logs": len({q["log"] for q in rows}),
            "boxed": sum(1 for q in rows if q["target"].get("box") is not None),
            "zoom_light": sum(1 for q in rows if q["target"].get("kind_z") == "light"),
            "zoom_line": sum(1 for q in rows if q["target"].get("kind_z") == "line"),
            "templates": dict(Counter(q["template"] for q in rows)),
            "letters": dict(sorted(Counter("ABCD"[q["answer"]] for q in rows).items())),
        }
    write_json(os.path.join(out_dir, "stats.json"), stats)
    return stats, report
