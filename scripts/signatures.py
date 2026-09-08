#!/usr/bin/env python3
"""
Signature analysis.

With R = log2(N) yes/no questions over N documents, perfect play is a
bijection: every document receives a distinct R-bit answer signature. Two
targets producing the same signature are structurally indistinguishable under
that question tree, and the final guess becomes a lottery rather than a
deduction.

Signature uniqueness is therefore an upper bound on winnable games that is
computable from the logs alone, with no judge model and no further API spend.
Where win rate mixes question quality, answerer agreement, and resolution
luck, this isolates one of them.

    python scripts/signatures.py
    python scripts/signatures.py --size 16      # detail for one size
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "raw"


def load():
    games = []
    for f in glob.glob(str(RESULTS / "*" / "*" / "*.json")):
        try:
            g = json.load(open(f, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if g.get("aborted"):
            continue
        games.append(g)
    return games


def sig(g) -> tuple:
    return tuple(h["answer"] for h in g["history"])


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, help="show per-signature detail")
    args = ap.parse_args()

    games = load()
    if not games:
        sys.exit("no completed games under results/raw/")

    by = defaultdict(list)
    for g in games:
        by[(g["model"], g["size"])].append(g)

    if args.size:
        print(f"\nsignature detail at N={args.size}")
        print("=" * 70)
        for (m, s), gs in sorted(by.items()):
            if s != args.size:
                continue
            c = Counter(sig(g) for g in gs)
            won = {sig(g): 0 for g in gs}
            for g in gs:
                won[sig(g)] += int(g["won"])
            print(f"\n{m}  ({sum(g['won'] for g in gs)}/{len(gs)} won)")
            for s_, n in c.most_common():
                bits = "".join("T" if b else "F" for b in s_)
                flag = "  <- collapsed" if n > 1 else ""
                print(f"   {bits:<12} x{n}  won {won[s_]}{flag}")
        return 0

    # main table
    print(f"\n{'model':<18}{'N':>6}{'games':>7}{'uniq':>6}{'won':>5}"
          f"{'collapsed':>11}{'yes rate':>10}")
    print("-" * 72)
    rows = []
    for (m, s), gs in sorted(by.items(), key=lambda x: (x[0][0], x[0][1])):
        sigs = [sig(g) for g in gs]
        uniq = len(set(sigs))
        c = Counter(sigs)
        collapsed = sum(n for n in c.values() if n > 1)
        won = sum(g["won"] for g in gs)
        ans = Counter(h["answer"] for g in gs for h in g["history"])
        yes = ans[True] / max(1, ans[True] + ans[False])
        rows.append((m, s, len(gs), uniq, won, collapsed, yes))
        print(f"{m[:17]:<18}{s:>6}{len(gs):>7}{uniq:>6}{won:>5}"
              f"{collapsed:>11}{yes:>9.0%}")

    # does uniqueness predict wins better than chance?
    print("\n" + "=" * 72)
    print("per model, summed over sizes:")
    agg = defaultdict(lambda: [0, 0, 0, 0])
    for m, s, n, uniq, won, coll, _ in rows:
        a = agg[m]
        a[0] += n; a[1] += uniq; a[2] += won; a[3] += coll
    print(f"{'model':<18}{'games':>7}{'uniq sigs':>11}{'won':>6}"
          f"{'in collapsed':>14}{'win rate':>10}")
    for m, (n, uniq, won, coll) in sorted(agg.items()):
        print(f"{m[:17]:<18}{n:>7}{uniq:>11}{won:>6}{coll:>14}{won/n:>9.0%}")

    # correlation between uniqueness fraction and win rate, across all cells
    xs = [u / n for _, _, n, u, _, _, _ in rows]
    ys = [w / n for _, _, n, _, w, _, _ in rows]
    mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
    print(f"\ncorrelation(signature uniqueness, win rate) over "
          f"{len(rows)} model-size cells: r = {num/den:+.2f}")

    # win rate inside vs outside collapsed signatures
    inside = [g for gs in by.values() for g in gs
              if Counter(sig(x) for x in gs)[sig(g)] > 1]
    outside = [g for gs in by.values() for g in gs
               if Counter(sig(x) for x in gs)[sig(g)] == 1]
    def rate(v): return (sum(g["won"] for g in v), len(v))
    iw, ino = rate(inside); ow, ono = rate(outside)
    print(f"\ngames whose signature was shared:  {iw}/{ino} = "
          f"{iw/max(1,ino):.0%} won")
    print(f"games with a unique signature:     {ow}/{ono} = "
          f"{ow/max(1,ono):.0%} won")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())