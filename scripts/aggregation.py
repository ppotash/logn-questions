#!/usr/bin/env python3
"""
Aggregation analysis: how well does each model survey its whole context?

A yes/no question that splits the candidate set evenly yields one full bit.
Deviation from a 50% answer rate therefore measures how badly the model
misjudged what fraction of its context carries the property it asked about --
which is a direct probe of aggregation over the full input, not retrieval from
it.

Round 1 is the cleanest instance: the questioner must survey all N documents
with no prior constraints and no viable-set re-derivation. Later rounds mix
that survey with the harder job of re-applying every earlier predicate. Split
by round to separate them.

    python scripts/aggregation.py
    python scripts/aggregation.py --round 1     # aggregation only
    python scripts/aggregation.py --late        # rounds 2+ only

Read-only, no API calls.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from math import log2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "raw"

SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def H(p: float) -> float:
    """Shannon entropy of a Bernoulli(p) answer, in bits."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * log2(p) + (1 - p) * log2(1 - p))


def corr(xs, ys) -> float:
    if len(xs) < 3:
        return float("nan")
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
    return num / den if den else float("nan")


def load(round_filter):
    """{(model, size): [answers]} plus {(model,size): (wins, games)}."""
    ans = defaultdict(list)
    rec = defaultdict(lambda: [0, 0])
    for f in glob.glob(str(RESULTS / "*" / "*" / "*.json")):
        try:
            g = json.load(open(f, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if g.get("aborted"):
            continue
        key = (g["model"], g["size"])
        rec[key][0] += int(g["won"]); rec[key][1] += 1
        for h in g["history"]:
            if round_filter == "first" and h["round"] != 1:
                continue
            if round_filter == "late" and h["round"] == 1:
                continue
            ans[key].append(h["answer"])
    return ans, rec


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=[1],
                    help="round 1 only: pure aggregation, no re-derivation")
    ap.add_argument("--late", action="store_true",
                    help="rounds 2+ only: aggregation plus re-derivation")
    args = ap.parse_args()

    which = "first" if args.round == 1 else ("late" if args.late else "all")
    label = {"first": "ROUND 1 ONLY (aggregation)",
             "late": "ROUNDS 2+ (aggregation + re-derivation)",
             "all": "ALL ROUNDS"}[which]

    ans, rec = load(which)
    if not ans:
        sys.exit("no completed games found")

    models = sorted({m for m, _ in ans})

    print(f"\n{label}")
    print("=" * 78)
    print(f"{'model':<18}" + "".join(f"{n:>7}" for n in SIZES))
    print("-" * 78)
    for m in models:
        cells = []
        for n in SIZES:
            a = ans.get((m, n), [])
            cells.append(f"{sum(a)/len(a):>6.0%}" if a else "     -")
        print(f"{m[:17]:<18}" + "".join(f"{c:>7}" for c in cells))
    print("(yes rate; 50% = perfectly balanced partition)")

    # drift with context length
    print(f"\n{'model':<18}{'r(|dev|, log2 N)':>18}{'small N':>10}"
          f"{'large N':>10}{'bits lost':>11}{'win rate':>10}")
    print("-" * 78)
    for m in models:
        xs, dev = [], []
        for n in SIZES:
            a = ans.get((m, n), [])
            if len(a) >= 4:
                xs.append(log2(n)); dev.append(abs(sum(a)/len(a) - .5))
        small = [x for n in SIZES if n <= 32 for x in ans.get((m, n), [])]
        large = [x for n in SIZES if n >= 256 for x in ans.get((m, n), [])]
        if not (small and large):
            continue
        ps = sum(small)/len(small); pl = sum(large)/len(large)
        w, gms = rec_totals(rec, m)
        print(f"{m[:17]:<18}{corr(xs, dev):>+18.2f}{ps:>10.0%}{pl:>10.0%}"
              f"{(H(ps)-H(pl))*10:>11.2f}{w/gms:>10.0%}")
    print("bits lost = 10 * [H(p_small) - H(p_large)], i.e. information forgone")
    print("over ten rounds at N=1024 relative to the model's own small-N calibration.")

    # overall information budget
    print(f"\n{'model':<18}{'yes rate':>10}{'H(p) bits':>11}"
          f"{'bits @ N=1024':>15}{'survivors':>11}{'win rate':>10}")
    print("-" * 78)
    for m in models:
        a = [x for n in SIZES for x in ans.get((m, n), [])]
        if not a:
            continue
        p = sum(a)/len(a); h = H(p)
        w, gms = rec_totals(rec, m)
        print(f"{m[:17]:<18}{p:>10.0%}{h:>11.3f}{h*10:>15.2f}"
              f"{1024*2**(-10*h):>11.2f}{w/gms:>10.0%}")
    print("10 bits are needed to isolate 1 of 1024; survivors = N * 2^(-R*H).")
    return 0


def rec_totals(rec, model):
    w = sum(v[0] for k, v in rec.items() if k[0] == model)
    g = sum(v[1] for k, v in rec.items() if k[0] == model)
    return w, max(1, g)


if __name__ == "__main__":
    raise SystemExit(main())