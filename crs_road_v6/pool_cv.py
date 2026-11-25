"""Pool CV-fold test predictions and report paper-style statistics.
Usage: python pool_cv.py ~/crs_project/cv      (expects fold*/preds/<model>_test.jsonl)
All CIs are bootstrap over LOGS (questions in one log are not independent)."""
import json, sys, glob, os, random, collections, math
root = sys.argv[1] if len(sys.argv) > 1 else "."
MODELS = ["p0", "p1", "p2", "p3_raw", "p3_det", "p3_oracle", "p1_blank"]
PAPER = {"p0": 0.420, "p1": 0.692, "p1_blank": 0.587}
def load(m):
    d = {}
    for f in sorted(glob.glob(f"{root}/fold*/preds/{m}_test.jsonl")):
        for l in open(f):
            r = json.loads(l); r["fold"] = f.split("fold")[-1].split("/")[0]; d[r["qid"]] = r
    return d
M = {m: load(m) for m in MODELS}; M = {m: v for m, v in M.items() if v}
folds = sorted({r["fold"] for r in M["p1"].values()}) if "p1" in M else []
print("folds found:", folds, {m: len(v) for m, v in M.items()})
def by_log(rows):
    g = collections.defaultdict(list)
    for r in rows: g[r["log"]].append(r)
    return g
def acc(rows): return sum(r["correct"] for r in rows) / max(1, len(rows))
def boot(rows, f=acc, B=2000, seed=0):
    g = list(by_log(rows).values()); rng = random.Random(seed); v = []
    for _ in range(B):
        s = [r for x in (rng.choice(g) for _ in g) for r in x]; v.append(f(s))
    v.sort(); return v[int(.025 * B)], v[int(.975 * B)]
def pdiff(a, b, filt=lambda r: True, B=2000, seed=0):
    ks = [k for k in a if k in b and filt(a[k])]
    if not ks: return None
    g = collections.defaultdict(list)
    for k in ks: g[a[k]["log"]].append(a[k]["correct"] - b[k]["correct"])
    G = list(g.values()); mean = sum(sum(x) for x in G) / len(ks)
    rng = random.Random(seed); v = []
    for _ in range(B):
        s = [d for x in (rng.choice(G) for _ in G) for d in x]; v.append(sum(s) / len(s))
    v.sort(); w = sum(d > 0 for x in G for d in x); l = sum(d < 0 for x in G for d in x)
    # exact two-sided sign test on discordant pairs
    n = w + l; p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n)
    return mean, v[int(.025 * B)], v[int(.975 * B)], w, l, p, len(ks)
print("\n== Accuracy, pooled over all folds (test only, 95% log-bootstrap CI) ==")
print(f"{'model':10s} {'n':>5s} {'acc':>6s}  {'95% CI':>15s}  {'paper':>6s}  small   reasoning   per-fold")
for m, v in M.items():
    r = list(v.values()); lo, hi = boot(r)
    sm = [x for x in r if x.get("small")]; rs = [x for x in r if x.get("reasoning") == "heavy"]
    pf = " ".join(f"{acc([x for x in r if x['fold']==f]):.2f}" for f in folds)
    print(f"{m:10s} {len(r):5d} {acc(r):6.3f}  [{lo:.3f},{hi:.3f}]  {PAPER.get(m, float('nan')):6.3f}  {acc(sm):.3f}(n{len(sm)})  {acc(rs):.3f}(n{len(rs)})  {pf}")
print("\n== Paired differences (pp), CI over logs, W/L = questions A right & B wrong / reverse ==")
pairs = [("p1", "p0"), ("p1", "p1_blank"), ("p2", "p1"), ("p3_det", "p1"), ("p3_det", "p2"), ("p3_raw", "p2"), ("p3_oracle", "p3_det")]
for a, b in pairs:
    if a in M and b in M:
        for name, filt in (("all", lambda r: True), ("small", lambda r: bool(r.get("small"))), ("reasoning", lambda r: r.get("reasoning") == "heavy")):
            d = pdiff(M[a], M[b], filt)
            if d: print(f"{a:10s}-{b:9s} {name:9s} {d[0]*100:+5.1f}pp CI[{d[1]*100:+5.1f},{d[2]*100:+5.1f}] W{d[3]} L{d[4]} p={d[5]:.3f} n={d[6]}")
print("\n== Per template (accuracy) ==")
ms = [m for m in ("p0", "p1", "p2", "p3_det") if m in M]
tp = collections.defaultdict(list)
for q, r in M["p1"].items(): tp[r["template"]].append(q)
print(f"{'template':28s} n   " + " ".join(f"{m:>6s}" for m in ms))
for t, qs in sorted(tp.items()):
    print(f"{t:28s} {len(qs):3d} " + " ".join(f"{sum(M[m][q]['correct'] for q in qs if q in M[m])/len(qs):6.2f}" for m in ms))
print("\n== Zoom slice (questions that received a zoom hint) ==")
if "p3_det" in M:
    h = [q for q, r in M["p3_det"].items() if r.get("hint")]
    print("n hinted", len(h), {m: round(sum(M[m][q]["correct"] for q in h if q in M[m]) / max(1, len(h)), 3) for m in ("p1", "p2", "p3_det", "p3_oracle") if m in M})
print("\nPaper reference (Table, 4B model): zero-shot 0.420, fine-tuned 0.692, blank-image fine-tuned 0.587. Your question set differs, so treat as a sanity range only.")
