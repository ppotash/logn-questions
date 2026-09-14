#!/usr/bin/env python3
"""
Token expenditure by round index.

Two reasons this is worth looking at.

The questioner's job changes shape across a game. Round 1 requires a global
survey of all N documents with no prior constraints. Round 10 requires
re-deriving which two documents survive nine predicates, then separating them.
If one of those is harder, the model should spend more on it.

And §5.2 of the paper treats every round as an equivalent trial with the same
success probability p. If token spend varies systematically by round, the
rounds are not equivalent in effort even if they are equivalent in outcome.

Columns: visible is output minus reasoning, i.e. what the parser sees.
Providers differ in whether reasoning is reported inside completion_tokens;
logn.providers normalises this, so output_tokens is always the billable total
and reasoning_tokens is the portion of it spent thinking.

    python scripts/tokens_by_round.py
    python scripts/tokens_by_round.py --balanced        # N=1024 only
    python scripts/tokens_by_round.py --role questioner --by-model
    python scripts/tokens_by_round.py --size 64
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn import prompts                                    # noqa: E402

RESULTS = ROOT / "results" / "raw"


def mean(v):
    return sum(v) / len(v) if v else 0.0


def median(v):
    if not v:
        return 0.0
    w = sorted(v)
    n = len(w)
    return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2


def bar(x, hi, width=22):
    if hi <= 0:
        return "." * width
    n = int(round(min(1.0, x / hi) * width))
    return "#" * n + "." * (width - n)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", default="questioner",
                    choices=["questioner", "answerer", "guesser", "all"])
    ap.add_argument("--model")
    ap.add_argument("--size", type=int)
    ap.add_argument("--balanced", action="store_true",
                    help="shorthand for --size 1024, so every round index "
                         "comes from the same games")
    ap.add_argument("--by-model", action="store_true")
    ap.add_argument("--repairs", action="store_true",
                    help="include repair retries; excluded by default because "
                         "they double-count a round")
    args = ap.parse_args()

    if args.balanced:
        args.size = 1024

    # calls[round] = list of (output, reasoning)
    calls = defaultdict(list)
    per_model = defaultdict(list)
    trunc = defaultdict(int)

    for f in glob.glob(str(RESULTS / "*" / "*" / "*.json")):
        g = json.load(io.open(f, encoding="utf-8"))
        if g.get("aborted"):
            continue
        if g.get("prompt_version") != prompts.PROMPT_VERSION:
            continue
        if args.model and g["model"] != args.model:
            continue
        if args.size and g["size"] != args.size:
            continue
        for c in g["calls"]:
            if args.role != "all" and c["role"] != args.role:
                continue
            if c.get("repaired") and not args.repairs:
                continue
            u = c.get("usage", {})
            out = u.get("output_tokens", 0) or 0
            rea = u.get("reasoning_tokens", 0) or 0
            # The guesser is logged as round 0; place it after the last round.
            r = c["round"] if c["round"] else 99
            calls[r].append((out, rea))
            per_model[(g["model"], r)].append((out, rea))
            if c.get("truncated"):
                trunc[r] += 1

    if not calls:
        sys.exit("no matching calls; check --role/--model/--size")

    scope = [args.role]
    if args.model:
        scope.append(args.model)
    scope.append(f"N={args.size}" if args.size else "all sizes")
    print(f"\ntoken output by round  [{'; '.join(scope)}]")
    print("visible = output - reasoning, i.e. what the parser sees\n")

    rounds = sorted(calls)
    peak = max(mean([o for o, _ in calls[r]]) for r in rounds)

    print(f"{'round':>6}{'calls':>7}{'output':>10}{'reasoning':>11}"
          f"{'visible':>9}{'reas %':>8}{'median':>9}  ")
    print("-" * 84)
    for r in rounds:
        v = calls[r]
        o = mean([x for x, _ in v])
        rr = mean([y for _, y in v])
        label = "guess" if r == 99 else str(r)
        share = rr / o if o else 0
        print(f"{label:>6}{len(v):>7}{o:>10,.0f}{rr:>11,.0f}{o-rr:>9,.0f}"
              f"{share:>8.0%}{median([x for x, _ in v]):>9,.0f}  "
              f"{bar(o, peak)}")

    allv = [x for r in rounds for x in calls[r]]
    print("-" * 84)
    print(f"{'all':>6}{len(allv):>7}{mean([x for x,_ in allv]):>10,.0f}"
          f"{mean([y for _,y in allv]):>11,.0f}"
          f"{mean([x-y for x,y in allv]):>9,.0f}")
    if trunc:
        print("truncated calls by round: "
              + ", ".join(f"R{r}:{n}" for r, n in sorted(trunc.items())))

    # trend over the numbered rounds only (exclude the guess call)
    num = [r for r in rounds if r != 99]
    if len(num) > 2:
        xs = [float(r) for r in num]
        ys = [mean([x for x, _ in calls[r]]) for r in num]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den_x = sum((x - mx) ** 2 for x in xs)
        num_ = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (den_x * sum((y - my) ** 2 for y in ys)) ** .5
        print(f"\noutput vs round index: r = {num_/den:+.2f}, "
              f"slope = {num_/den_x:+,.0f} tokens per round")
        print(f"round 1 = {ys[0]:,.0f}, round {num[-1]} = {ys[-1]:,.0f}, "
              f"ratio {ys[-1]/ys[0] if ys[0] else 0:.2f}x")

    if args.by_model:
        models = sorted({m for m, _ in per_model})
        print(f"\nmean output tokens per call\n")
        print(f"{'model':<18}" + "".join(
            f"{('guess' if r==99 else r):>8}" for r in rounds))
        print("-" * (18 + 8 * len(rounds)))
        for m in models:
            cells = []
            for r in rounds:
                v = per_model[(m, r)]
                cells.append(f"{mean([x for x,_ in v]):>8,.0f}" if v
                             else f"{'-':>8}")
            print(f"{m[:17]:<18}" + "".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())