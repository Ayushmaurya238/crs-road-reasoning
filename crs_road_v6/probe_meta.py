import json
import re
import sys
from collections import Counter

BOX = re.compile(r"box|bbox|xyxy|xywh|rect|pos|point|2d", re.I)
TS = re.compile(r"time|ts|stamp|image|img|file|path|url|cam", re.I)


def is_nums(v, n=None):
    return isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) and (n is None or len(v) == n)


def walk(x, path, hits, seen, depth=0):
    if isinstance(x, dict):
        key = re.sub(r"\d+", "#", "/".join(path))
        if key not in seen and depth < 7:
            seen[key] = 1
            ks = list(x)[:12]
            print("  " * depth + (path[-1] if path else "<root>") + " {dict, %d keys}: %s" % (len(x), ks))
        for k, v in list(x.items())[:60]:
            if BOX.search(str(k)) and (is_nums(v) or isinstance(v, (dict, list))):
                hits["box"].append(("/".join(path + [str(k)]), str(v)[:80]))
            if TS.search(str(k)):
                hits["ts"].append(("/".join(path + [str(k)]), str(v)[:80]))
            walk(v, path + [str(k)], hits, seen, depth + 1)
    elif isinstance(x, list):
        key = re.sub(r"\d+", "#", "/".join(path)) + "[]"
        if key not in seen and depth < 7:
            seen[key] = 1
            print("  " * depth + (path[-1] if path else "<root>") + " [list, %d items] first: %s" % (len(x), str(x[0])[:90] if x else "-"))
        if is_nums(x, 4):
            hits["box4"].append(("/".join(path), str(x)))
        for v in x[:5]:
            walk(v, path + ["#"], hits, seen, depth + 1)


def main():
    d = json.load(open(sys.argv[1]))
    hits = {"box": [], "ts": [], "box4": []}
    walk(d, [], hits, {})
    for k in ("box", "box4", "ts"):
        c = Counter(re.sub(r"\d+", "#", p) for p, _ in hits[k])
        print("\n== %s-like paths (%d) ==" % (k, len(hits[k])))
        for p, n in c.most_common(15):
            ex = next(v for q, v in hits[k] if re.sub(r"\d+", "#", q) == p)
            print(n, p, "e.g.", ex)


if __name__ == "__main__":
    main()
