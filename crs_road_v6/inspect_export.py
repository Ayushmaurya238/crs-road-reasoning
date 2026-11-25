import json
import sys


def show(o, key="root", depth=0, maxdepth=6, seen=None):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key}: dict({len(o)}) keys={list(o)[:12]}")
        if depth < maxdepth:
            for k in list(o)[:12]:
                show(o[k], str(k), depth + 1, maxdepth)
    elif isinstance(o, list):
        print(f"{pad}{key}: list({len(o)})")
        if o and depth < maxdepth:
            show(o[0], key + "[0]", depth + 1, maxdepth)
    else:
        s = repr(o)
        print(f"{pad}{key}: {type(o).__name__} = {s[:80]}")


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        show(json.load(f))
