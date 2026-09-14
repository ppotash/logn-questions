#!/usr/bin/env python3
"""
Answer agreement by round index.

The compounding model in the paper assumes each round succeeds independently
with probability p. If that holds, agreement between the answerer and the
judges should be flat across round index: round 8 should be no worse than
round 1. A downward trend would mean errors cluster late, which breaks
independence and would need explaining -- for instance, that questions get
harder to adjudicate once the candidate set is small and the questioner starts
asking about fine distinctions.

Two confounds to keep in mind when reading the output. Round r only exists for
games with N >= 2^r, so high round indices are drawn only from large-N games;
the --balanced view restricts to N=1024 so every round index comes from the
same games. And a game that ends early (aborted, or a repair failure)
contributes fewer rounds, though in this dataset that is rare.

    python scripts/by_round.py
    python scripts/by_round.py --balanced          # N=1024 only
    python scripts/by_round.py --model claude-opus-5
    python scripts/by_round.py --cache judgments_gpt.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import sys
from collections import defaultdict
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
        name = f.stem.replace("judgments_", "").replace("judgments", "default")
        out[name] = json.loads(f.read_text(encoding="utf-8"))
    if not out:
        sys.exit("no judgement caches under results/adjudicated/")
    return out


def bar(frac: float, width: int = 24) -> str:
    """Simple ASCII bar, scaled 0.80-1.00 so differences are visible."""
    lo = 0.80
    f = max(0.0, (frac - lo) / (1 - lo))
    n = int(round(f * width))
    return "#" * n + "." * (width - n)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="restrict to one model")
    ap.add_argument("--size", type=int, help="restrict to one N")
    ap.add_argument("--balanced", action="store_true",
                    help="shorthand for --size 1024: every round index then "
                         "comes from the same set of games")
    ap.add_argument("--cache", help="one cache filename; default is all of them")
    ap.add_argument("--by-model", action="store_true",
                    help="break the table down per model")
    args = ap.parse_args()

    if args.balanced:
        args.size = 1024

    judges = load_judges(args.cache)
    jnames = list(judges)

    # agree[round] = [n_agree, n_total]; same keyed by (model, round)
    agree = defaultdict(lambda: [0, 0])
    per_model = defaultdict(lambda: [0, 0])
    yes_rate = defaultdict(lambda: [0, 0])
    skipped_unjudged = 0

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
        tid = g["target_id"]
        for h in g["history"]:
            r = h["round"]
            k = key(h["question"], tid)
            verdicts = [c[k] for c in judges.values() if k in c]
            if not verdicts:
                skipped_unjudged += 1
                continue
            # One observation per (call, judge): a round judged by three
            # judges contributes three trials.
            for v in verdicts:
                agree[r][0] += int(v == h["answer"])
                agree[r][1] += 1
                per_model[(g["model"], r)][0] += int(v == h["answer"])
                per_model[(g["model"], r)][1] += 1
            yes_rate[r][0] += int(h["answer"])
            yes_rate[r][1] += 1

    scope = []
    if args.model:
        scope.append(args.model)
    scope.append(f"N={args.size}" if args.size else "all sizes")
    scope.append(f"{len(jnames)} judge(s): {', '.join(jnames)}")
    print(f"\nanswer agreement by round index  [{'; '.join(scope)}]")
    if skipped_unjudged:
        print(f"({skipped_unjudged} calls had no judgement and were skipped)")

    print(f"\n{'round':>6}{'agree':>8}{'trials':>8}{'rate':>8}{'yes rate':>10}  ")
    print("-" * 66)
    rounds = sorted(agree)
    for r in rounds:
        a, n = agree[r]
        y, yn = yes_rate[r]
        print(f"{r:>6}{a:>8}{n:>8}{a/n:>8.3f}{y/yn:>10.0%}  {bar(a/n)}")

    tot_a = sum(agree[r][0] for r in rounds)
    tot_n = sum(agree[r][1] for r in rounds)
    print("-" * 66)
    print(f"{'all':>6}{tot_a:>8}{tot_n:>8}{tot_a/tot_n:>8.3f}")
    print("bar scaled 0.80-1.00")

    # trend
    if len(rounds) > 2:
        xs = [float(r) for r in rounds]
        ys = [agree[r][0] / agree[r][1] for r in rounds]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs)
               * sum((y - my) ** 2 for y in ys)) ** .5
        r_ = num / den if den else float("nan")
        slope = num / sum((x - mx) ** 2 for x in xs)
        print(f"\ncorrelation(agreement, round index) r = {r_:+.2f}, "
              f"slope = {slope:+.4f} per round")
        print("flat is what the independence assumption in the p^log2(N) fit "
              "requires;\na negative slope means errors cluster in later "
              "rounds.")

    if args.by_model:
        models = sorted({m for m, _ in per_model})
        print(f"\n{'model':<18}" + "".join(f"{r:>7}" for r in rounds))
        print("-" * (18 + 7 * len(rounds)))
        for m in models:
            cells = []
            for r in rounds:
                a, n = per_model[(m, r)]
                cells.append(f"{a/n:>7.2f}" if n >= 8 else f"{'-':>7}")
            print(f"{m[:17]:<18}" + "".join(cells))
        print("blank where fewer than 8 trials")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())