import math
import os
import random
from collections import defaultdict

import numpy as np

from .common import read_jsonl


def cluster_ci(rows, key=lambda r: r["correct"], b=1000, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r.get("log", r["scene_id"])].append(key(r))
    scenes = list(by)
    if not scenes:
        return (float("nan"),) * 3
    rng = random.Random(seed)
    vals = np.array([v for s in scenes for v in by[s]], dtype=float)
    stats = []
    for _ in range(b):
        samp = [by[rng.choice(scenes)] for _ in scenes]
        flat = [v for g in samp for v in g]
        stats.append(sum(flat) / len(flat))
    return float(vals.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar_p(a, b):
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def summarize(rows):
    out = {"total": cluster_ci(rows)}
    for name, sel in (
        ("perception", lambda r: r["reasoning"] == "perception"),
        ("reasoning", lambda r: r["reasoning"] == "heavy"),
        ("small_attr", lambda r: r["small"]),
    ):
        sub = [r for r in rows if sel(r)]
        if sub:
            out[name] = cluster_ci(sub)
    for t in sorted({r["template"] for r in rows}):
        sub = [r for r in rows if r["template"] == t]
        out["tpl:" + t] = (float(np.mean([r["correct"] for r in sub])), len(sub))
    for lo, hi in ((0, 20), (20, 40), (40, 1e9)):
        sub = [r for r in rows if r.get("size_px") is not None and lo <= r["size_px"] < hi]
        if sub:
            out[f"size[{lo},{hi})"] = (float(np.mean([r["correct"] for r in sub])), len(sub))
    for d in sorted({r["depth"] for r in rows}):
        sub = [r for r in rows if r["depth"] == d]
        out[f"depth{d}"] = (float(np.mean([r["correct"] for r in sub])), len(sub))
    return out


def compare(path_a, path_b):
    a = {r["qid"]: r for r in read_jsonl(path_a)}
    b = {r["qid"]: r for r in read_jsonl(path_b)}
    ids = sorted(set(a) & set(b))
    xa = [a[i]["correct"] for i in ids]
    xb = [b[i]["correct"] for i in ids]
    return {"n": len(ids), "acc_a": float(np.mean(xa)), "acc_b": float(np.mean(xb)), "mcnemar_p": mcnemar_p(xa, xb)}


def compare_all(named_paths, plot_path=None):
    table = []
    summaries = {}
    for name, path in named_paths:
        rows = read_jsonl(path)
        s = summarize(rows)
        summaries[name] = s
        ious = [r["gt_iou"] for r in rows if "gt_iou" in r]
        chits = [r["center_hit"] for r in rows if "center_hit" in r]
        hints = sum(1 for r in rows if r.get("hint"))
        snapped = sum(1 for r in rows if r.get("snapped"))
        table.append({
            "model": name,
            "n": len(rows),
            "total": s["total"][0],
            "perception": s.get("perception", (float("nan"),))[0],
            "reasoning": s.get("reasoning", (float("nan"),))[0],
            "small_attr": s.get("small_attr", (float("nan"),))[0],
            "mean_iou": float(np.mean(ious)) if ious else float("nan"),
            "center_hit": float(np.mean(chits)) if chits else float("nan"),
            "hints": hints,
            "snapped": snapped,
        })
    print(f"{'Model':<12} {'N':>4} {'Total':>7} {'Perc':>7} {'Reas':>7} {'Small':>7} {'IoU':>6} {'Ctr':>6} {'Hints':>6} {'Snap':>5}")
    for r in table:
        fmt = lambda v: f"{v:.3f}" if not math.isnan(v) else "  -  "
        print(f"{r['model']:<12} {r['n']:>4} {fmt(r['total']):>7} {fmt(r['perception']):>7} {fmt(r['reasoning']):>7} {fmt(r['small_attr']):>7} {fmt(r['mean_iou']):>6} {fmt(r['center_hit']):>6} {r['hints']:>6} {r['snapped']:>5}")

    by_name = dict(named_paths)
    for pair in (("P3", "P1"), ("P3", "P2"), ("P2", "P1")):
        if pair[0] in by_name and pair[1] in by_name:
            print(f"{pair[0]} vs {pair[1]}:", compare(by_name[pair[0]], by_name[pair[1]]))

    if plot_path:
        plot_comparison(summaries, plot_path)
    return table


def plot_comparison(summaries, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(summaries.keys())
    metrics = [m for m in ("total", "perception", "reasoning", "small_attr") if any(m in summaries[k] for k in models)]
    labels = {"total": "Total", "perception": "Perception", "reasoning": "Reasoning", "small_attr": "Small Attr (Lights/Lines)"}
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, m in enumerate(models):
        means, lo, hi = [], [], []
        for k in metrics:
            v = summaries[m].get(k, (0.0, 0.0, 0.0))
            means.append(v[0])
            lo.append(max(0.0, v[0] - v[1]))
            hi.append(max(0.0, v[2] - v[0]))
        off = (i - (len(models) - 1) / 2) * w
        bars = ax.bar(x + off, means, w, yerr=[lo, hi], capsize=3, label=m)
        for b, val in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy (95% log-clustered CI)")
    ax.set_title("CRS Road Reasoning: P0 / P1 / P2 / P3 Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print("saved plot:", out_path)
