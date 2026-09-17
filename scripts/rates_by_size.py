#!/usr/bin/env python3
"""
Per-round failure rates by set size.

Both failure types rise with N per game, but games at larger N have more rounds.
This asks whether the per-round rate is constant once that is accounted for.

Two normalisations, reported side by side:

  simple      failures per game divided by R = log2(N). Treats each round as an
              independent opportunity and asks how many are consumed.

  compounded  q = (1 - d)^(1/R), the per-round success rate implied if a game
              avoids the failure only by succeeding at every round. This matches
              the p^log2(N) transform used elsewhere in the analysis and is the
              right one if the failure is terminal rather than incremental.

Discrimination has no round index: it is a property of a whole game, defined as
every answer being correct while the guessed document remains consistent with
all of them. Normalising it by R does not locate it in time; it asks whether
each question contributes a constant amount of separating power regardless of
how many documents there are.

Answer errors do have a round index, so for them the direct per-round rate is
also reported, counted over judged calls rather than inferred from games.

    python scripts/rates_by_size.py
    python scripts/rates_by_size.py --exclude claude-opus-5
    python scripts/rates_by_size.py --only claude-opus-5
    python scripts/rates_by_size.py --cache judgments_gpt.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import sys
from collections import defaultdict
from math import log2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn import prompts                                    # noqa: E402

RESULTS = ROOT / "results" / "raw"
CACHE_DIR = ROOT / "results" / "adjudicated"


def key(q: str, d: str) -> str:
    return hashlib.sha256(f"{d}\x00{q}".encode()).hexdigest()[:20]


def load_judges(which=None) -> dict:
    files = ([CACHE_DIR / which] if which
             else sorted(CACHE_DIR.glob("judgments*.json")))
    out = {}
    for f in files:
        if not f.exists():
            sys.exit(f"no such cache: {f}")
        c = json.loads(f.read_text(encoding="utf-8"))
        # A stub cache with a handful of entries will silently dilute every
        # average, so skip anything far smaller than the rest.
        out[f.stem] = c
    if not out:
        sys.exit("no judgement caches under results/adjudicated/")
    big = max(len(c) for c in out.values())
    kept = {k: v for k, v in out.items() if len(v) >= big * 0.5}
    for k, v in out.items():
        if k not in kept:
            print(f"  skipping {k}: {len(v)} entries against {big} in the "
                  f"largest cache", file=sys.stderr)
    return kept


def corr(xs, ys):
    if len(xs) < 3:
        return float("nan")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    n = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    d = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** .5
    return n / d if d else float("nan")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--cache")
    args = ap.parse_args()

    judges = load_judges(args.cache)
    jnames = list(judges)

    # per size: games, answer-error games, discrimination games, and the
    # round-level answer tallies
    games = defaultdict(int)
    ans_games = defaultdict(float)
    disc_games = defaultdict(float)
    round_err = defaultdict(float)
    round_tot = defaultdict(float)
    models = set()

    for f in glob.glob(str(RESULTS / "*" / "*" / "*.json")):
        g = json.load(io.open(f, encoding="utf-8"))
        if g.get("aborted") or g.get("prompt_version") != prompts.PROMPT_VERSION:
            continue
        m = g["model"]
        if args.only and m not in args.only:
            continue
        if m in args.exclude:
            continue
        models.add(m)
        N = g["size"]
        games[N] += 1
        tid = g["target_id"]

        for jn, cache in judges.items():
            err_round = None
            complete = True
            for h in g["history"]:
                k = key(h["question"], tid)
                if k not in cache:
                    complete = False
                    break
                round_tot[N] += 1 / len(judges)
                if cache[k] != h["answer"]:
                    round_err[N] += 1 / len(judges)
                    if err_round is None:
                        err_round = h["round"]
            if not complete:
                continue
            if err_round is not None:
                ans_games[N] += 1 / len(judges)
                continue
            if g["won"] or not g["guess"]:
                continue
            gid = g["doc_ids"][g["guess"] - 1]
            consistent = True
            for h in g["history"]:
                k = key(h["question"], gid)
                if k not in cache:
                    consistent = None
                    break
                if cache[k] != h["answer"]:
                    consistent = False
                    break
            if consistent is True:
                disc_games[N] += 1 / len(judges)

    if not games:
        sys.exit("no games matched")

    scope = (f"{len(models)} models" if not args.only else ", ".join(sorted(models)))
    if args.exclude:
        scope += f" (excluding {', '.join(args.exclude)})"
    print(f"\nper-round failure rates by set size  [{scope}; "
          f"{len(jnames)} judge(s)]")

    sizes = sorted(games)
    print(f"\n{'N':>6}{'R':>3}{'games':>7}"
          f"{'disc/game':>11}{'disc/R':>9}{'q':>8}"
          f"{'ans/game':>10}{'ans/round':>11}")
    print("-" * 66)
    rows = []
    for N in sizes:
        R = int(log2(N))
        n = games[N]
        d = disc_games[N] / n
        a = ans_games[N] / n
        per_round_ans = (round_err[N] / round_tot[N]) if round_tot[N] else 0.0
        q = (1 - d) ** (1 / R) if R else float("nan")
        rows.append((N, R, n, d, q, a, per_round_ans))
        print(f"{N:>6}{R:>3}{n:>7}{d:>11.1%}{d/R:>9.1%}{q:>8.3f}"
              f"{a:>10.1%}{per_round_ans:>11.1%}")

    xs = [log2(r[0]) for r in rows]
    print(f"\ncorrelations with log2(N):")
    print(f"  discrimination per game      r = {corr(xs,[r[3] for r in rows]):+.2f}")
    print(f"  discrimination per round     r = {corr(xs,[r[3]/r[1] for r in rows]):+.2f}")
    print(f"  q (compounded per-round)     r = {corr(xs,[r[4] for r in rows]):+.2f}")
    print(f"  answer error per game        r = {corr(xs,[r[5] for r in rows]):+.2f}")
    print(f"  answer error per round       r = {corr(xs,[r[6] for r in rows]):+.2f}")
    print("\nA flat q means each question contributes the same separating power "
          "regardless\nof how many documents it has to separate. A declining q "
          "means questions lose\ndiscriminating power as the set grows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())