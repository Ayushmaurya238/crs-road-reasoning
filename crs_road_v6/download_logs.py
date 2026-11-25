import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from netutil import http_get

BASE = os.environ.get("CRS_BASE", "https://kth-rpl.github.io/CRS/annotator/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="CRS_Logs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all logs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for sub in ("annotations", "metadata"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    logs = json.loads(http_get(BASE + "logs.json"))
    ids = sorted(l["logid"] for l in logs)
    if a.limit:
        ids = ids[: a.limit]
    print("logs to fetch:", len(ids), "of", len(logs))

    def one(logid):
        ok = True
        for sub in ("annotations", "metadata"):
            p = os.path.join(a.out, sub, logid + ".json")
            if os.path.exists(p) and os.path.getsize(p) > 0:
                continue
            try:
                data = http_get(BASE + f"{sub}/{logid}.json")
                json.loads(data)
                with open(p, "wb") as f:
                    f.write(data)
            except Exception as e:
                print("FAILED", sub, logid, e)
                ok = False
        return ok

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, ids))
    print("completed %d/%d" % (sum(res), len(ids)), "->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
