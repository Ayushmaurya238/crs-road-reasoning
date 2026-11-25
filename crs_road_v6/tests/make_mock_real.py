import json
import os
import random
import sys

from PIL import Image

F = range(32)


def node(label, typ, props=None, helpers=None, vis=True):
    return {"label": label, "type": typ, "properties": props or {}, "helpers": helpers or {},
            "visibleAt": {str(f): True for f in F} if vis else {}}


def make_log(logid, seed=0):
    full = lambda v: {str(f): v for f in F}
    lane = lambda lab, u, d, t, desc, uk: node(lab, "lane", {"type": t, "direction": d, "description": desc},
                                               {"uuid": u, "is_complete": full(True), "uniqueKey": {str(f): uk for f in F}})
    nodes = [
        node("Ego", "ego", helpers={"context": full("1")}),
        lane("Lane-1", "1", "same", "VEHICLE", "right curb lane", ["description"]),
        lane("Lane-2", "2", "same", "VEHICLE", "", ["e1"]),
        lane("Lane-3", "3", "opposite", "VEHICLE", "left curb lane", ["description"]),
        lane("Lane-4", "4", "opposite", "BIKE", "", ["e2"]),
        node("TrafficLight-1", "traffic_light", {"status": {str(f): ("red" if f < 16 else "green") for f in F}, "type": "vehicle"},
             {"uuid": "tl_01", "uniqueKey": {str(f): ["is unique"] for f in F}}),
        node("TrafficSign-1", "sign", {"meaning": "no_right_turn"}, {"uuid": "rs_01", "uniqueKey": {str(f): ["is unique"] for f in F}}),
        node("Bus-1", "bus", {"description": "red bus"}, {"uuid": "b1", "context": full("1"), "uniqueKey": {str(f): ["description"] for f in F}}),
        node("RegularVehicle-1", "regular_vehicle", {"description": "black sedan"}, {"uuid": "v1", "context": full("2"), "uniqueKey": {str(f): ["description"] for f in F}}),
        node("LaneLine-1", "lane_line", {"mark_type": "DASHED_WHITE"}, {"left_of": ["2"], "right_of": ["1"], "uniqueKey": {str(f): ["e3"] for f in F}}),
        node("LaneLine-2", "lane_line", {"mark_type": "DOUBLE_SOLID_YELLOW"}, {"left_of": ["3"], "right_of": ["2"], "uniqueKey": {str(f): ["e4"] for f in F}}),
        node("Crossing-1", "crossing", {"marking_style": "zebra crossing"}, {"is_complete": full(True), "uniqueKey": {str(f): ["is unique"] for f in F}}),
    ]
    E = [("e1", "Lane-2", "is left of", "Lane-1"), ("e2", "Lane-4", "is left of", "Lane-3"), ("e3", "LaneLine-1", "is right marking of", "Lane-2"),
         ("e4", "LaneLine-2", "is left marking of", "Lane-2"), ("e5", "Lane-1", "is right of", "Lane-2"), ("e6", "Bus-1", "is in", "Lane-1"),
         ("e7", "RegularVehicle-1", "is in", "Lane-2"), ("e8", "TrafficLight-1", "controls", "Lane-1"), ("e9", "TrafficSign-1", "controls", "Lane-2"),
         ("e10", "Lane-3", "is left of", "Lane-2")]
    edges = [{"source": s, "target": t, "temporalLabels": lab, "category": "location", "label": i} for i, s, lab, t in E]
    return {"nodes": nodes, "edges": edges, "counterRef": {}, "logid": logid}


def make_meta(logid):
    R_center = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    cal = {
        "ring_front_center": {"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0, "R": R_center, "t": [[0.0], [1.5], [0.0]]},
        "ring_front_left": {"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0, "R": R_center, "t": [[-1.0], [1.5], [0.0]]},
        "ring_front_right": {"fx": 120.0, "fy": 120.0, "cx": 80.0, "cy": 60.0, "R": R_center, "t": [[1.0], [1.5], [0.0]]},
    }
    I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return {
        "source_id": logid,
        "target_id": "00001",
        "sensor": {c: [str(1000000000 + f * 100000000) for f in F] for c in cal},
        "calibration": cal,
        "pose": {"translations": [[float(f) * 0.5, 0.0, 0.0] for f in F], "rotations": [I3 for _ in F]},
        "traffic_elements": [
            {"id": "tl_01", "category": 1, "points": {str(f): [[60.0, 20.0], [76.0, 40.0]] for f in F}, "status": {str(f): ("red" if f < 16 else "green") for f in F}},
            {"id": "rs_01", "category": 0, "points": {str(f): [[100.0, 30.0], [120.0, 50.0]] for f in F}, "meaning": "no_right_turn"},
        ],
        "objects": [
            {"uuid": "b1", "type": "bus", "positions": {"ring_front_center": {str(f): [20.0, 50.0, 60.0, 95.0] for f in F}}, "context": {str(f): "1" for f in F}},
            {"uuid": "v1", "type": "regular_vehicle", "positions": {"ring_front_center": {str(f): [85.0, 55.0, 125.0, 95.0] for f in F}}, "context": {str(f): "2" for f in F}},
        ],
        "lane_lines": {
            "line_1": {"geometry": [[5.0, -1.5, 0.0], [35.0, -1.5, 0.0]], "mark_type": "DASHED_WHITE", "left_of": ["2"], "right_of": ["1"], "visibleAt": list(F)},
            "line_2": {"geometry": [[5.0, 1.5, 0.0], [35.0, 1.5, 0.0]], "mark_type": "DOUBLE_SOLID_YELLOW", "left_of": ["3"], "right_of": ["2"], "visibleAt": list(F)},
        },
    }


def make(root, nlogs=4):
    g = os.path.join(root, "annotations")
    m = os.path.join(root, "metadata")
    fr = os.path.join(root, "frames")
    os.makedirs(g, exist_ok=True)
    os.makedirs(m, exist_ok=True)
    ids = []
    for i in range(nlogs):
        lid = "%08x-0000-0000-0000-%012x" % (i + 1, i + 1)
        ids.append(lid)
        json.dump(make_log(lid, i), open(os.path.join(g, lid + ".json"), "w"))
        json.dump(make_meta(lid), open(os.path.join(m, lid + ".json"), "w"))
        for c in ("left", "center", "right"):
            d = os.path.join(fr, lid, c)
            os.makedirs(d, exist_ok=True)
            for f in F:
                Image.new("RGB", (160, 120), (i * 40, f * 7, 90)).save(os.path.join(d, f"{f:02d}.jpg"))
    return g, m, fr, ids


if __name__ == "__main__":
    print(make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4))
