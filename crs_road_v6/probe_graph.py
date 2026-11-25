import json
import sys
from collections import Counter

d = json.load(open(sys.argv[1]))
print("logid:", d.get("logid"))
print("counterRef:", d.get("counterRef"))
nodes, edges = d["nodes"], d["edges"]
print("nodes:", len(nodes), "edges:", len(edges))


def short(o, n=700):
    s = json.dumps(o)
    return s if len(s) <= n else s[:n] + " ..."


seen = set()
for nd in nodes:
    t = nd.get("type")
    if t not in seen:
        seen.add(t)
        print("\n== node type:", t)
        print(short(nd))

def rel(e):
    t = e.get("temporalLabels")
    if isinstance(t, dict):
        return "/".join(sorted(set(map(str, t.values()))))
    return str(t)


print("\n== relations (temporalLabels):")
for k, v in Counter(rel(e) for e in edges).most_common():
    print(" ", v, k)
print("\n== edge samples:")
for e in edges[:6]:
    print(short(e, 300))

fr = Counter()
for nd in nodes:
    for key in ("visibleAt",):
        v = nd.get(key)
        if isinstance(v, (list, dict)):
            fr[type(v).__name__] += 1
print("\nvisibleAt types:", dict(fr))
for nd in nodes:
    if nd.get("type") == "ego":
        print("\nego:", short(nd, 1500))
        break
