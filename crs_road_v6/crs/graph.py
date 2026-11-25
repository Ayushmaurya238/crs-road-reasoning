import glob
import math
import os
import random
from collections import Counter

from .common import (
    COLOR_NAME,
    LCOLOR,
    LIGHT,
    LTYPE,
    TYPE_NAME,
    read_json,
    write_json,
    write_jsonl,
)

NOUN = {"lane": "lane", "vehicle": "vehicle", "light": "traffic light", "sign": "sign", "line": "lane line"}
HEAVY = {"vehicle_position", "pairwise_vehicle_location", "sign_controls_lane", "traffic_light_controls_lane"}
SMALL = {"traffic_light_status", "traffic_light_change", "line_marking", "line_type", "line_color"}
LANE_NAME = {"vehicle": "standard vehicle lane", "bus": "bus lane", "bike": "bike lane"}
DIRN = {"same": "Same direction as the ego.", "opposite": "Opposite direction to the ego."}
FILL = ["None of the above.", "Cannot be determined."]
NOTARGET = {"kind": "none", "cam": None, "box": None, "size_px": None, "kind_z": "none"}


def cap(s):
    return s[0].upper() + s[1:]


def ordinal(n):
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class Desc:
    __slots__ = ("text", "depth", "chain")

    def __init__(self, text, depth, chain):
        self.text, self.depth, self.chain = text, depth, chain


class Graph:
    def __init__(self, sc):
        self.sc = sc
        self.nodes, self.out, self.by_type = {}, {}, {}
        sz = sc["image_size"]
        cur = sc["frames"][-1]

        def add(nid, typ, props, cam=None, box=None):
            nb, px = None, None
            if box is not None:
                w, h = sz[cam]
                nb = [box[0] / w, box[1] / h, box[2] / w, box[3] / h]
                px = (box[2] - box[0], box[3] - box[1])
            self.nodes[nid] = {"type": typ, "props": props, "cam": cam, "box": nb, "px": px}
            self.out[nid] = []
            self.by_type.setdefault(typ, []).append(nid)

        def edge(a, label, b):
            self.out[a].append((label, b))

        lanes = sorted(sc["lanes"], key=lambda l: l["index"])
        n = len(lanes)
        for l in lanes:
            add(
                l["id"],
                "lane",
                {
                    "direction": l["direction"],
                    "type": l["type"],
                    "ego": bool(l["ego"]),
                    "ord_l": l["index"] + 1,
                    "ord_r": n - l["index"],
                    "index": l["index"],
                },
            )
        for a, b in zip(lanes[:-1], lanes[1:]):
            edge(a["id"], "is left of", b["id"])
            edge(b["id"], "is right of", a["id"])
        for v in sc["vehicles"]:
            add(
                v["id"],
                "vehicle",
                {"category": v["category"], "desc": f"{v['color']} {v['category']}"},
                v["cam"],
                v["box"],
            )
            edge(v["id"], "is in", v["lane_id"])
            edge(v["lane_id"], "contains", v["id"])
        static = {s["id"]: s for s in sc["lights_static"]}
        for lt in cur["lights"]:
            hist = []
            for fr in sc["frames"]:
                m = [x["status"] for x in fr["lights"] if x["id"] == lt["id"]]
                hist.append(m[0] if m else None)
            add(lt["id"], "light", {"status": lt["status"], "hist": hist, "view": lt["cam"]}, lt["cam"], lt["box"])
            for lid in static[lt["id"]]["lane_ids"]:
                edge(lt["id"], "controls", lid)
                edge(lid, "is controlled by", lt["id"])
        for s in sc["signs"]:
            add(s["id"], "sign", {"meaning": s["meaning"], "view": s["cam"]}, s["cam"], s["box"])
            for lid in s["lane_ids"]:
                edge(s["id"], "controls", lid)
                edge(lid, "is controlled by", s["id"])
        for ln in sc["lines"]:
            add(ln["id"], "line", {"type": ln["type"], "color": ln["color"]}, ln["cam"], ln["box"])
            if ln["left_of"]:
                edge(ln["id"], "is left marking of", ln["left_of"])
            if ln["right_of"]:
                edge(ln["id"], "is right marking of", ln["right_of"])
        self.oset = {k: set(v) for k, v in self.out.items()}
        self.lane_order = [l["id"] for l in lanes]

    def ids(self, typ):
        return list(self.by_type.get(typ, []))

    def p(self, nid, key):
        return self.nodes[nid]["props"][key]

    def prop_phrases(self, nid):
        n = self.nodes[nid]
        t, pr = n["type"], n["props"]
        out = []
        if t == "lane":
            if pr["ego"]:
                out.append(("ego", True, "the ego lane"))
            out.append(("ord_r", pr["ord_r"], f"the {ordinal(pr['ord_r'])} lane from the right"))
            out.append(("ord_l", pr["ord_l"], f"the {ordinal(pr['ord_l'])} lane from the left"))
            if pr["type"] != "vehicle":
                out.append(("type", pr["type"], f"the {LANE_NAME[pr['type']]}"))
        elif t == "vehicle":
            out.append(("desc", pr["desc"], f"the {pr['desc']}"))
            out.append(("category", pr["category"], f"the {pr['category']}"))
        elif t == "light":
            out.append(("status", pr["status"], f"the traffic light with status {pr['status']}"))
            out.append(("view", pr["view"], f"the traffic light in the {pr['view'].upper()} view"))
        elif t == "sign":
            out.append(("meaning", pr["meaning"], f"the sign with meaning {pr['meaning']}"))
            out.append(("view", pr["view"], f"the sign in the {pr['view'].upper()} view"))
        return out

    def descs(self, nid, depth, fp=frozenset(), fe=frozenset(), seen=()):
        t = self.nodes[nid]["type"]
        same = self.by_type[t]
        res = []
        if len(same) == 1:
            ph = f"the {NOUN[t]}"
            res.append(Desc(ph, 0, [(nid, ph)]))
        for key, val, phrase in self.prop_phrases(nid):
            if key in fp:
                continue
            if sum(1 for k in same if self.nodes[k]["props"].get(key) == val) == 1:
                res.append(Desc(phrase, 0, [(nid, phrase)]))
        if depth > 0:
            for label, m in self.out[nid]:
                if label in fe or m in seen or m == nid:
                    continue
                if self.nodes[m]["type"] == "line":
                    continue
                for c in self.descs(m, depth - 1, fp, fe, seen + (nid,)):
                    if [k for k in same if (label, m) in self.oset[k]] == [nid]:
                        text = f"the {NOUN[t]} that {label} {c.text}"
                        res.append(Desc(text, c.depth + 1, c.chain + [(nid, text)]))
        return res

    def pick(self, nid, rng, maxd, fp=(), fe=()):
        c = self.descs(nid, maxd, frozenset(fp), frozenset(fe))
        if not c:
            return None
        by = {}
        for d in c:
            by.setdefault(d.depth, []).append(d)
        return rng.choice(by[rng.choice(sorted(by))])

    def loc(self, nid):
        n = self.nodes[nid]
        if n["box"] is None:
            return ""
        b = [int(round(x * 1000)) for x in n["box"]]
        return f", visible in the {n['cam'].upper()} view at <box>({b[0]},{b[1]},{b[2]},{b[3]})</box>"

    def target(self, nid, kind, **lab):
        n = self.nodes[nid]
        size = math.sqrt(n["px"][0] * n["px"][1]) if n["px"] else None
        return {
            "kind": kind,
            "cam": n["cam"],
            "box": n["box"],
            "size_px": size,
            "node": nid,
            "kind_z": kind if kind in ("light", "line") else "none",
            **lab,
        }


def mk_options(correct, pool, rng):
    decoys = [p for p in dict.fromkeys(pool) if p != correct]
    rng.shuffle(decoys)
    opts = [correct] + decoys[:3]
    for f in FILL:
        if len(opts) < 4 and f not in opts:
            opts.append(f)
    if len(opts) != 4:
        return None
    rng.shuffle(opts)
    return opts, opts.index(correct)


def make_cot(G, chains, tgt_sentence, correct):
    parts = []
    k = 1
    for ci, chain in enumerate(chains):
        for j, (nid, phrase) in enumerate(chain):
            verb = "Identify" if j == 0 else "Then resolve"
            w = 2.0 if (ci == 0 and j == 0) else 1.0
            parts.append([f"Step {k}: {verb} {phrase}{G.loc(nid)}.\n", w])
            k += 1
    parts.append([f"Step {k}: {tgt_sentence}\n", 2.0])
    parts.append([f"Conclusion: the correct answer is: {correct}", 1.0])
    return parts


def build_q(G, name, question, correct, pool, rng, descs, target, tgt_sentence):
    o = mk_options(correct, pool, rng)
    if o is None:
        return None
    opts, ai = o
    depth = max([d.depth for d in descs] + [0]) + (1 if name in HEAVY else 0)
    return {
        "template": name,
        "question": question,
        "options": opts,
        "answer": ai,
        "answer_text": correct,
        "cot_parts": make_cot(G, [d.chain for d in descs], tgt_sentence, correct),
        "depth": depth,
        "target": target,
        "reasoning": "heavy" if name in HEAVY else "perception",
        "small": name in SMALL,
    }


def lane_options(G, lane, rng, md, fe):
    texts = {}
    for l in G.ids("lane"):
        d = G.pick(l, rng, md, fe=fe)
        if d is None:
            return None, None
        texts[l] = cap(d.text) + "."
    return texts[lane], [t for k, t in texts.items() if k != lane]


def t_lane_direction(G, rng, md):
    cand = [l for l in G.ids("lane") if not G.p(l, "ego")]
    if not cand:
        return None
    l = rng.choice(cand)
    d = G.pick(l, rng, md, fp={"direction"})
    if d is None:
        return None
    ans = DIRN[G.p(l, "direction")]
    return build_q(G, "lane_direction", f"What is the direction of {d.text}?", ans, list(DIRN.values()), rng, [d], NOTARGET, f"That lane has direction: {ans}")


def t_lane_type(G, rng, md):
    if not G.ids("lane"):
        return None
    l = rng.choice(G.ids("lane"))
    d = G.pick(l, rng, md, fp={"type"})
    if d is None:
        return None
    ans = cap(LANE_NAME[G.p(l, "type")]) + "."
    pool = [cap(v) + "." for v in LANE_NAME.values()]
    return build_q(G, "lane_type", f"What lane is {d.text}?", ans, pool, rng, [d], NOTARGET, f"That lane is a {LANE_NAME[G.p(l, 'type')]}.")


def _line_of(G, lane, side):
    want = "is left marking of" if side == "left" else "is right marking of"
    for ln in G.ids("line"):
        if (want, lane) in G.oset[ln]:
            return ln
    return None


def _line_name(t, c):
    return TYPE_NAME["none"] if t == "none" else f"{TYPE_NAME[t]} {COLOR_NAME[c]}"


def _t_line(G, rng, md, name):
    if not G.ids("lane"):
        return None
    l = rng.choice(G.ids("lane"))
    side = rng.choice(["left", "right"])
    ln = _line_of(G, l, side)
    d = G.pick(l, rng, md)
    if ln is None or d is None:
        return None
    t, c = G.p(ln, "type"), G.p(ln, "color")
    tg = G.target(ln, "line", ltype=LTYPE.index(t), lcolor=LCOLOR.index(c))
    if name == "line_type":
        ans = TYPE_NAME[t]
        pool = list(TYPE_NAME.values())
        q = f"What is the {side} lane line type of {d.text}?"
    elif name == "line_color":
        ans = COLOR_NAME[c]
        pool = list(COLOR_NAME.values())
        q = f"What is the {side} lane line color of {d.text}?"
    else:
        ans = _line_name(t, c)
        pool = [_line_name(a, b) for a in LTYPE for b in LCOLOR if (a == "none") == (b == "none")]
        q = f"What is the {side} lane line marking of {d.text}?"
    return build_q(G, name, q, ans, pool, rng, [d], tg, f"The {side} lane line is {_line_name(t, c)}.")


def t_line_type(G, rng, md):
    return _t_line(G, rng, md, "line_type")


def t_line_color(G, rng, md):
    return _t_line(G, rng, md, "line_color")


def t_line_marking(G, rng, md):
    return _t_line(G, rng, md, "line_marking")


def t_light_status(G, rng, md):
    cand = [l for l in G.ids("light") if G.p(l, "status") in LIGHT]
    if not cand:
        return None
    l = rng.choice(cand)
    d = G.pick(l, rng, md, fp={"status"})
    if d is None:
        return None
    st = G.p(l, "status")
    tg = G.target(l, "light", light=LIGHT.index(st))
    return build_q(G, "traffic_light_status", f"What is the status of {d.text}?", st, LIGHT, rng, [d], tg, f"The traffic light shows {st}.")


def t_light_change(G, rng, md):
    cand = [l for l in G.ids("light") if all(h in LIGHT for h in G.p(l, "hist"))]
    if not cand:
        return None
    l = rng.choice(cand)
    d = G.pick(l, rng, md, fp={"status"})
    if d is None:
        return None
    h = G.p(l, "hist")
    n = len(h)

    def txt(a, b):
        return f"Yes, it was {a} but switched to {b}." if a != b else f"No, it stayed {a}."

    ans = txt(h[0], h[-1])
    pool = [txt(a, b) for a in LIGHT for b in LIGHT]
    tg = G.target(l, "light", light=LIGHT.index(h[-1]))
    return build_q(
        G,
        "traffic_light_change",
        f"Did the status of {d.text} change over the last {n} frames?",
        ans,
        pool,
        rng,
        [d],
        tg,
        f"Its status sequence over the frames is {', '.join(h)}.",
    )


def _count_phrase(k, kind):
    noun = "lane" if k == 1 else "lanes"
    verb = "is" if k == 1 else "are"
    where = "in the ego direction" if kind == "same" else "in the opposite direction"
    if k == 0:
        return f"There are no lanes {where}."
    return f"There {verb} {k} {noun} {where}."


def t_counting_generic(G, rng, md):
    if not G.sc.get("lanes_complete"):
        return None
    kind = rng.choice(["same", "opposite"])
    k = sum(1 for l in G.ids("lane") if G.p(l, "direction") == kind)
    lo = 1 if kind == "same" else 0
    ans = _count_phrase(k, kind)
    pool = [_count_phrase(j, kind) for j in range(max(lo, k - 2), k + 3)]
    q = "How many lanes are there in the ego direction?" if kind == "same" else "How many lanes are there in the direction opposite to the ego?"
    return build_q(G, "counting_generic", q, ans, pool, rng, [], NOTARGET, f"Counting the lanes gives {k}.")


def t_counting_per_direction(G, rng, md):
    if not G.sc.get("lanes_complete"):
        return None
    a = sum(1 for l in G.ids("lane") if G.p(l, "direction") == "same")
    b = len(G.ids("lane")) - a

    def txt(x, y):
        return f"{x} in the ego direction and {y} in the opposite direction."

    pool = [txt(x, y) for x in range(max(1, a - 2), a + 3) for y in range(max(0, b - 2), b + 3)]
    return build_q(G, "counting_per_direction", "How many lanes are there?", txt(a, b), pool, rng, [], NOTARGET, f"There are {a + b} lanes in total.")


def t_pairwise_lane_direction(G, rng, md):
    lanes = G.ids("lane")
    if len(lanes) < 2:
        return None
    l1, l2 = rng.sample(lanes, 2)
    d1 = G.pick(l1, rng, md, fp={"direction"})
    d2 = G.pick(l2, rng, md, fp={"direction"})
    if d1 is None or d2 is None:
        return None
    a, b = G.p(l1, "direction") == "same", G.p(l2, "direction") == "same"
    opts = {
        (True, True): "Yes, both are in the ego direction.",
        (False, False): "Yes, both are opposite to the ego direction.",
        (True, False): "No, the first is in the ego direction and the second is opposite.",
        (False, True): "No, the first is opposite and the second is in the ego direction.",
    }
    return build_q(
        G,
        "pairwise_lane_direction",
        f"Do {d1.text} and {d2.text} share the same direction?",
        opts[(a, b)],
        list(opts.values()),
        rng,
        [d1, d2],
        NOTARGET,
        "I compare the directions of both lanes.",
    )


def t_pairwise_vehicle_location(G, rng, md):
    vs = G.ids("vehicle")
    if len(vs) < 2:
        return None
    v1, v2 = rng.sample(vs, 2)
    d1 = G.pick(v1, rng, md, fe={"is in"})
    d2 = G.pick(v2, rng, md, fe={"is in"})
    if d1 is None or d2 is None:
        return None
    lane1 = [m for lab, m in G.out[v1] if lab == "is in"][0]
    lane2 = [m for lab, m in G.out[v2] if lab == "is in"][0]
    i1, i2 = G.p(lane1, "index"), G.p(lane2, "index")
    same = "Yes, they are in the same lane."
    left = "No, the first is in a lane left of the second's lane."
    right = "No, the first is in a lane right of the second's lane."
    ans = same if i1 == i2 else (left if i1 < i2 else right)
    return build_q(G, "pairwise_vehicle_location", f"Are {d1.text} and {d2.text} in the same lane?", ans, [same, left, right], rng, [d1, d2], NOTARGET, "I locate the lane of each vehicle and compare them.")


def t_vehicle_position(G, rng, md):
    vs = G.ids("vehicle")
    if not vs:
        return None
    v = rng.choice(vs)
    dv = G.pick(v, rng, md, fe={"is in"})
    if dv is None:
        return None
    lane = [m for lab, m in G.out[v] if lab == "is in"][0]
    ans, pool = lane_options(G, lane, rng, md, {"contains"})
    if ans is None:
        return None
    tg = G.target(v, "vehicle")
    return build_q(G, "vehicle_position", f"Which lane is {dv.text} in?", ans, pool, rng, [dv], tg, f"That vehicle is in {ans[0].lower() + ans[1:-1]}.")


def _t_controls(G, rng, md, typ, name):
    cand = [s for s in G.ids(typ) if len([m for lab, m in G.out[s] if lab == "controls"]) == 1]
    if not cand:
        return None
    s = rng.choice(cand)
    d = G.pick(s, rng, md, fe={"controls"})
    if d is None:
        return None
    lane = [m for lab, m in G.out[s] if lab == "controls"][0]
    ans, pool = lane_options(G, lane, rng, md, {"is controlled by"})
    if ans is None:
        return None
    tg = G.target(s, "sign" if typ == "sign" else "light_ctx")
    return build_q(G, name, f"Which lane does {d.text} control?", ans, pool, rng, [d], tg, f"It controls {ans[0].lower() + ans[1:-1]}.")


def t_sign_controls_lane(G, rng, md):
    return _t_controls(G, rng, md, "sign", "sign_controls_lane")


def t_light_controls_lane(G, rng, md):
    return _t_controls(G, rng, md, "light", "traffic_light_controls_lane")


TEMPLATES = {
    "lane_direction": t_lane_direction,
    "lane_type": t_lane_type,
    "line_marking": t_line_marking,
    "line_type": t_line_type,
    "line_color": t_line_color,
    "traffic_light_status": t_light_status,
    "traffic_light_change": t_light_change,
    "counting_generic": t_counting_generic,
    "counting_per_direction": t_counting_per_direction,
    "pairwise_lane_direction": t_pairwise_lane_direction,
    "pairwise_vehicle_location": t_pairwise_vehicle_location,
    "vehicle_position": t_vehicle_position,
    "sign_controls_lane": t_sign_controls_lane,
    "traffic_light_controls_lane": t_light_controls_lane,
}


def valid(q):
    o = q["options"]
    return len(o) == 4 and len(set(o)) == 4 and 0 <= q["answer"] < 4 and o[q["answer"]] == q["answer_text"] and sum(1 for x in o if x == q["answer_text"]) == 1


def gen_scene(sc, rng, split, md=2, cap_per_template=3, max_per_scene=40):
    G = Graph(sc)
    out = []
    want = 1 if split != "train" else cap_per_template
    tries = 40 if split != "train" else 12 * want
    for name, fn in TEMPLATES.items():
        got = {}
        for _ in range(tries):
            q = fn(G, rng, md)
            if q is None or not valid(q):
                continue
            key = (q["question"], tuple(sorted(q["options"])))
            if key not in got:
                got[key] = q
            if len(got) >= want:
                break
        out.extend(got.values())
    rng.shuffle(out)
    if split == "train":
        out = out[:max_per_scene]
    for i, q in enumerate(out):
        q["qid"] = f"{sc['scene_id']}_{i}"
        q["scene_id"] = sc["scene_id"]
        q["split"] = split
    return out


def rotate(q, pos):
    shift = (pos - q["answer"]) % 4
    o = q["options"]
    q["options"] = [o[(i - shift) % 4] for i in range(4)]
    q["answer"] = q["options"].index(q["answer_text"])


def build_dataset(scenes_dir, out_dir, seed=0, md=2, cap_per_template=3, max_per_scene=40):
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(scenes_dir, "*.json")))
    by_split = {"train": [], "dev": [], "test": []}
    for f in files:
        sc = read_json(f)
        by_split[sc["split"]].extend(gen_scene(sc, rng, sc["split"], md, cap_per_template, max_per_scene))
    stats = {}
    for split, rows in by_split.items():
        rng.shuffle(rows)
        for i, q in enumerate(rows):
            rotate(q, i % 4)
        write_jsonl(os.path.join(out_dir, f"qa_{split}.jsonl"), rows)
        stats[split] = {
            "n": len(rows),
            "templates": dict(Counter(q["template"] for q in rows)),
            "letters": dict(Counter("ABCD"[q["answer"]] for q in rows)),
            "scenes": len({q["scene_id"] for q in rows}),
        }
    write_json(os.path.join(out_dir, "stats.json"), stats)
    return stats


def check_dataset(out_dir):
    from .common import read_jsonl

    report = {}
    tr = read_jsonl(os.path.join(out_dir, "qa_train.jsonl"))
    maj = {}
    for t in {q["template"] for q in tr}:
        maj[t] = Counter(q["answer_text"] for q in tr if q["template"] == t).most_common(1)[0][0]
    for split in ("train", "dev", "test"):
        p = os.path.join(out_dir, f"qa_{split}.jsonl")
        if not os.path.exists(p):
            continue
        rows = read_jsonl(p)
        bad = [q["qid"] for q in rows if not valid(q)]
        letters = Counter("ABCD"[q["answer"]] for q in rows)
        hit = [q["options"][q["answer"]] == maj.get(q["template"]) for q in rows]
        report[split] = {
            "n": len(rows),
            "invalid": len(bad),
            "letter_spread": max(letters.values()) - min(letters.values()) if letters else 0,
            "majority_text_prior_acc": sum(hit) / max(1, len(hit)),
        }
    return report
