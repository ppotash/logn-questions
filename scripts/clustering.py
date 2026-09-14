#!/usr/bin/env python3
"""
Do answer errors cluster within a game?

The marginal agreement rate by round index says whether reliability decays with
horizon position. It cannot say whether errors are independent. A wrong answer
at round 4 might make round 5 more likely to go wrong, and the marginal rates
would look identical either way.

This computes the transition probabilities directly:

    P(error at r+1 | error at r)   against   P(error at r+1 | correct at r)

and tests the difference with a within-game permutation. The permutation holds
each game's length and error count fixed and shuffles only the positions, so a
game that simply contains more errors cannot create apparent clustering. That
is the right null: the question is whether errors bunch, not whether some games
are worse.

A confound that cannot be removed from this data. Once an answer eliminates the
target, every later question is chosen to separate documents that are not the
target. Those questions were never selected to be decidable about the target,
so they may be harder to adjudicate for reasons that have nothing to do with
the answerer getting worse. Observed clustering is therefore consistent with
two readings: the answerer degrades after an error, or the questions do. The
--pre-only view restricts to rounds before the first error, where this
confound does not apply.

Each call's verdict is the majority across available judges.

    python scripts/clustering.py
    python scripts/clustering.py --model claude-opus-5
    python scripts/clustering.py --balanced          # N=1024 only
    python scripts/clustering.py --trials 20000
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import random
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
        if f.exists():
            name = f.stem.replace("judgments_", "").replace("judgments", "default")
            out[name] = json.loads(f.read_text(encoding="utf-8"))
    if not out:
        sys.exit("no judgement caches under results/adjudicated/")
    return out


def sequences(judges, model=None, size=None):
    """One error sequence per game: [False, False, True, ...] by round.

    True means the majority of judges disagreed with the answer given.
    Games with any unjudged round are dropped, so every sequence is complete.
    """
    seqs = []
    dropped = 0
    for f in glob.glob(str(RESULTS / "*" / "*" / "*.json")):
        g = json.load(io.open(f, encoding="utf-8"))
        if g.get("aborted") or g.get("prompt_version") != prompts.PROMPT_VERSION:
            continue
        if model and g["model"] != model:
            continue
        if size and g["size"] != size:
            continue
        tid, seq, ok = g["target_id"], [], True
        for h in g["history"]:
            k = key(h["question"], tid)
            v = [c[k] for c in judges.values() if k in c]
            if not v:
                ok = False
                break
            majority = sum(1 for x in v if x == h["answer"]) * 2 > len(v)
            seq.append(not majority)
        if ok and len(seq) >= 2:
            seqs.append((g["model"], g["size"], seq))
        elif not ok:
            dropped += 1
    return seqs, dropped


def transitions(seqs):
    """Counts of (prev state -> error) over adjacent round pairs."""
    after_err = [0, 0]      # [errors, trials] given previous was an error
    after_ok = [0, 0]
    for _, _, s in seqs:
        for i in range(len(s) - 1):
            tgt = after_err if s[i] else after_ok
            tgt[0] += int(s[i + 1])
            tgt[1] += 1
    return after_err, after_ok


def permute(seqs, trials, rng):
    """Null: shuffle error positions within each game, keeping the count."""
    obs_e, obs_o = transitions(seqs)
    if not obs_e[1] or not obs_o[1]:
        return None
    obs = obs_e[0] / obs_e[1] - obs_o[0] / obs_o[1]
    ge = 0
    shuffled = [(m, n, list(s)) for m, n, s in seqs]
    for _ in range(trials):
        for _, _, s in shuffled:
            rng.shuffle(s)
        e, o = transitions(shuffled)
        if e[1] and o[1] and (e[0] / e[1] - o[0] / o[1]) >= obs:
            ge += 1
    return obs, (ge + 1) / (trials + 1)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--size", type=int)
    ap.add_argument("--balanced", action="store_true",
                    help="shorthand for --size 1024")
    ap.add_argument("--cache", help="one judgement cache; default is all")
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--pre-only", action="store_true",
                    help="truncate each game after its first error, removing "
                         "the post-elimination confound")
    ap.add_argument("--by-model", action="store_true")
    args = ap.parse_args()

    if args.balanced:
        args.size = 1024

    judges = load_judges(args.cache)
    seqs, dropped = sequences(judges, args.model, args.size)
    if not seqs:
        sys.exit("no complete judged games matched")

    if args.pre_only:
        cut = []
        for m, n, s in seqs:
            if True in s:
                s = s[:s.index(True) + 1]
            if len(s) >= 2:
                cut.append((m, n, s))
        seqs = cut

    rng = random.Random(0)
    scope = [args.model or "all models",
             f"N={args.size}" if args.size else "all sizes",
             f"{len(judges)} judge(s), majority verdict"]
    if args.pre_only:
        scope.append("truncated at first error")
    print(f"\nerror clustering  [{'; '.join(scope)}]")
    if dropped:
        print(f"({dropped} games dropped for unjudged rounds)")

    rounds = sum(len(s) for _, _, s in seqs)
    errs = sum(sum(s) for _, _, s in seqs)
    print(f"\n{len(seqs)} games, {rounds} judged rounds, {errs} errors "
          f"({errs/rounds:.1%} marginal error rate)")

    e, o = transitions(seqs)
    print(f"\n{'previous round':<20}{'errors':>9}{'trials':>9}{'P(error)':>11}")
    print("-" * 50)
    print(f"{'was an error':<20}{e[0]:>9}{e[1]:>9}"
          + (f"{e[0]/e[1]:>11.3f}" if e[1] else f"{'-':>11}"))
    print(f"{'was correct':<20}{o[0]:>9}{o[1]:>9}"
          + (f"{o[0]/o[1]:>11.3f}" if o[1] else f"{'-':>11}"))

    res = permute(seqs, args.trials, rng)
    if res:
        diff, p = res
        print(f"\ndifference = {diff:+.3f}")
        print(f"permutation test, {args.trials:,} shuffles within games: "
              f"p = {p:.4f}")
        if p > 0.05:
            print("no evidence that errors cluster; consistent with the "
                  "independence\nassumption in the p^log2(N) fit.")
        else:
            print("errors cluster more than chance. Note the confound above: "
                  "questions\nasked after the target is eliminated were never "
                  "selected to be\ndecidable about it.")

    # how many games have 2+ errors, versus what independence would predict
    multi = sum(1 for _, _, s in seqs if sum(s) >= 2)
    exp_multi = 0.0
    pbar = errs / rounds if rounds else 0
    for _, _, s in seqs:
        n = len(s)
        p0 = (1 - pbar) ** n
        p1 = n * pbar * (1 - pbar) ** (n - 1)
        exp_multi += 1 - p0 - p1
    print(f"\ngames with 2+ errors: {multi} observed, {exp_multi:.1f} expected "
          f"under independence")

    if args.by_model:
        print(f"\n{'model':<18}{'games':>7}{'err rate':>10}"
              f"{'P(e|e)':>9}{'P(e|ok)':>9}")
        print("-" * 55)
        for m in sorted({m for m, _, _ in seqs}):
            sub = [x for x in seqs if x[0] == m]
            e2, o2 = transitions(sub)
            r2 = sum(len(s) for _, _, s in sub)
            x2 = sum(sum(s) for _, _, s in sub)
            print(f"{m[:17]:<18}{len(sub):>7}{x2/r2:>10.1%}"
                  + (f"{e2[0]/e2[1]:>9.2f}" if e2[1] else f"{'-':>9}")
                  + (f"{o2[0]/o2[1]:>9.2f}" if o2[1] else f"{'-':>9}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())