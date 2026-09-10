#!/usr/bin/env python3
"""
Decompose losses into answer, discrimination, and prediction errors.

Win/loss compresses three distinct failures into one bit. Separating them needs
to know, for a given question, whether a given document satisfies it -- but not
for all N documents. Two suffice: the target, and whatever the model guessed.

  ANSWER ERROR         an independent adjudication of (question, target)
                       disagrees with what the answerer replied. The target was
                       eliminated by a wrong answer and every later round was
                       spent on a set that could not contain it. The round of
                       first disagreement is the round of death.

  DISCRIMINATION       every answer was correct, but the guessed document is
                       ALSO consistent with all of them. The questions never
                       separated the two and the final guess was a lottery.

  PREDICTION ERROR     every answer was correct and the guessed document is
                       inconsistent with the evidence the model itself
                       received. It had enough information and chose wrong.

  WIN+ERR              won despite an answer error: the questioner nominally
                       eliminated the target and guessed it anyway. Still an
                       agreement failure, but not a loss. Binning these as
                       ANSWER ERROR inflates the loss count.

A caveat that cannot be engineered away: for vague questions there may be no
fact of the matter, so a judge disagreeing with the answerer is not proof the
answerer erred. Both are estimates of the same underlying agreement quantity.
A judge that is also one of the evaluated models is biased toward agreeing with
itself, so run several judges and compare with --all-judges.

    python scripts/errors.py --judge gemini-3.8-flash --dry-run
    python scripts/errors.py --judge gemini-3.8-flash --workers 12
    python scripts/errors.py --report
    python scripts/errors.py --report --cache judgments_gpt.json
    python scripts/errors.py --all-judges

Judgements cache under results/adjudicated/ and are reused, so the script is
resumable and re-running costs nothing.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn.game import load_corpus                              # noqa: E402
from logn.providers.base import ProviderError                  # noqa: E402
from logn.runner import build_client, load_config              # noqa: E402

RESULTS = ROOT / "results" / "raw"
CACHE_DIR = ROOT / "results" / "adjudicated"
CACHE = CACHE_DIR / "judgments.json"      # default; --cache overrides

# Mean rounds per game across the size mix: 4 games at R=2, plus 8 games each
# at R=3..10. Converts games-containing-an-error into a per-round rate.
MEAN_ROUNDS = 6.24

CATS = ["win", "win+err", "answer", "discrimination", "prediction",
        "no guess", "unjudged"]

JUDGE_SYSTEM = """\
You judge whether a yes/no question is true of a document's subject.

Answer with exactly one word: Yes or No.

Use ordinary world knowledge; the document will not always state the answer \
outright. If a question contrasts the subject with an alternative ("X rather \
than Y"), ignore the contrast and judge only whether the subject fits X. If a \
question is vague or only partly applicable, judge whether it is a fair \
description of this subject overall. One of Yes or No always describes the \
document better than the other: choose it. Never reply with anything else."""

JUDGE_TAIL = """\
Document:
{document}

Question: {question}"""


# --------------------------------------------------------------------------
# cache and inputs
# --------------------------------------------------------------------------

def key(question: str, doc_id: str) -> str:
    return hashlib.sha256(f"{doc_id}\x00{question}".encode()).hexdigest()[:20]


def load_cache(path: Path | None = None) -> dict:
    path = path or CACHE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(c: dict, path: Path | None = None) -> None:
    path = path or CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c), encoding="utf-8")


def games():
    for f in sorted(glob.glob(str(RESULTS / "*" / "*" / "*.json"))):
        try:
            g = json.load(open(f, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not g.get("aborted"):
            yield g


def needed(pool) -> list[tuple[str, str]]:
    """(question, doc_id) pairs to adjudicate: the target every round, and the
    guessed document on losses."""
    want = set()
    for g in games():
        for h in g["history"]:
            want.add((h["question"], g["target_id"]))
        if not g["won"] and g["guess"]:
            gid = g["doc_ids"][g["guess"] - 1]
            for h in g["history"]:
                want.add((h["question"], gid))
    return sorted(want)


# --------------------------------------------------------------------------
# adjudication
# --------------------------------------------------------------------------

def adjudicate(client, pool, pairs, cache, limit=None, workers=8,
               effort="low", max_tokens=2000, path=None):
    """Judge pairs concurrently.

    Each judgement is independent, so this parallelises cleanly. httpx.Client
    is thread-safe, so one client is shared. The cache is guarded by a lock and
    flushed periodically, which makes the run resumable: Ctrl-C keeps
    everything already judged.
    """
    from logn import prompts

    todo = [x for x in pairs if key(*x) not in cache][:limit or None]
    if not todo:
        return 0

    lock = threading.Lock()
    state = {"done": 0, "failed": 0, "t0": time.perf_counter()}

    def judge(pair):
        q, did = pair
        d = pool[did]
        pr = prompts.Prompt(
            system=JUDGE_SYSTEM, cacheable="",
            tail=JUDGE_TAIL.format(document=f"{d['title']}\n{d['text']}",
                                   question=q))
        kw = {"max_tokens": max_tokens, "temperature": 0.0}
        try:
            import inspect
            params = inspect.signature(client.complete).parameters
            if "effort" in params and effort:
                kw["effort"] = effort
            c = client.complete(pr, **kw)
        except ProviderError as e:
            return pair, None, str(e)[:70]
        r = prompts.parse_answer(c.text)
        return pair, (bool(r.value) if r.ok else None), None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(judge, p_): p_ for p_ in todo}
        try:
            for fut in as_completed(futures):
                pair, val, err = fut.result()
                with lock:
                    if val is not None:
                        cache[key(*pair)] = val
                    else:
                        state["failed"] += 1
                    state["done"] += 1
                    n = state["done"]
                    if n % 100 == 0:
                        save_cache(cache, path)
                        el = time.perf_counter() - state["t0"]
                        rate = n / el
                        left = (len(todo) - n) / rate if rate else 0
                        print(f"  {n}/{len(todo)} judged  {rate:.1f}/s  "
                              f"~{left/60:.0f} min left"
                              + (f"  ({state['failed']} failed)"
                                 if state["failed"] else ""), flush=True)
        except KeyboardInterrupt:
            print("\n  interrupted; saving what is done", file=sys.stderr)
            for f in futures:
                f.cancel()

    save_cache(cache, path)
    return state["done"]


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify(g, pool, cache):
    """Returns (category, round_of_death or None). Categories are CATS."""
    tid = g["target_id"]

    # Did any answer contradict an independent adjudication of the target?
    # Record the round but do not return yet: the game may still be won.
    err_round = None
    for h in g["history"]:
        k = key(h["question"], tid)
        if k not in cache:
            return ("unjudged", None)
        if cache[k] != h["answer"]:
            err_round = h["round"]
            break

    if g["won"]:
        return ("win+err", err_round) if err_round else ("win", None)
    if err_round:
        return ("answer", err_round)
    if not g["guess"]:
        return ("no guess", None)

    # Answers all correct and the game still lost. Is the guessed document
    # distinguishable from the target under these questions?
    gid = g["doc_ids"][g["guess"] - 1]
    for h in g["history"]:
        k = key(h["question"], gid)
        if k not in cache:
            return ("unjudged", None)
        if cache[k] != h["answer"]:
            return ("prediction", None)
    return ("discrimination", None)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(pool, cache, label=""):
    by_model = defaultdict(Counter)
    by_size = defaultdict(Counter)
    death = defaultdict(list)
    for g in games():
        cat, rd = classify(g, pool, cache)
        by_model[g["model"]][cat] += 1
        by_size[g["size"]][cat] += 1
        if cat in ("answer", "win+err") and rd:
            death[g["model"]].append(rd / g["rounds"])

    if label:
        print(f"\n=== {label} ===")
    print(f"\n{'model':<18}" + "".join(f"{c[:7]:>9}" for c in CATS)
          + f"{'ans err':>9}")
    print("-" * (18 + 9 * len(CATS) + 9))
    for m in sorted(by_model):
        c = by_model[m]
        n = sum(c.values())
        # A game won despite an answer error is still an agreement failure.
        rate = (c["answer"] + c["win+err"]) / n
        print(f"{m[:17]:<18}" + "".join(f"{c[k]:>9}" for k in CATS)
              + f"{rate:>9.0%}")

    print(f"\n{'N':<18}" + "".join(f"{c[:7]:>9}" for c in CATS))
    print("-" * (18 + 9 * len(CATS)))
    for n_ in sorted(by_size):
        c = by_size[n_]
        print(f"{n_:<18}" + "".join(f"{c[k]:>9}" for k in CATS))

    print("\nwhere the target dies, as a fraction of the round budget:")
    for m in sorted(death):
        v = death[m]
        if v:
            print(f"  {m[:17]:<18}n={len(v):<4} mean {sum(v)/len(v):.2f} "
                  f"(0 = first round, 1 = last)")

    tot = Counter()
    for c in by_model.values():
        tot.update(c)
    losses = sum(tot[k] for k in ("answer", "discrimination", "prediction"))
    if losses:
        print(f"\nlosses by cause ({losses} total):")
        for k in ("answer", "discrimination", "prediction"):
            print(f"  {k:<16}{tot[k]:>5}  {tot[k]/losses:>5.0%}")
    if tot["win+err"]:
        print(f"\n{tot['win+err']} games won despite an answer error "
              f"(counted as wins, not losses)")


def report_all(pool, cache_dir):
    """Compare every judgement cache side by side.

    Per-model numbers vary by judge for two reasons: genuine disagreement, and
    self-preference where the judge is also the evaluated model. Printing them
    together makes both visible; the 'others' column excludes the
    self-judgement where one exists.
    """
    caches = {}
    for f in sorted(Path(cache_dir).glob("judgments*.json")):
        name = f.stem.replace("judgments_", "").replace("judgments", "default")
        try:
            caches[name] = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"  skipping unreadable {f.name}", file=sys.stderr)
    if not caches:
        print("no judgement caches found under", cache_dir)
        return

    judges = list(caches)
    models = sorted({g["model"] for g in games()})
    print(f"{len(judges)} judges: {', '.join(judges)}")

    # map each judge to the evaluated model it is, if any
    self_of = {j: next((m for m in models if j.split("-")[0] in m), None)
               for j in judges}

    tallies = {j: defaultdict(Counter) for j in judges}
    sizes = {j: defaultdict(Counter) for j in judges}
    for g in games():
        for j, c in caches.items():
            cat, _ = classify(g, pool, c)
            tallies[j][g["model"]][cat] += 1
            sizes[j][g["size"]][cat] += 1

    print("\ngames (of 68) containing an answer error, by judge")
    print(f"{'model':<18}" + "".join(f"{j[:8]:>9}" for j in judges)
          + f"{'self':>7}{'others':>9}{'emp. p':>9}")
    print("-" * (18 + 9 * len(judges) + 25))
    for m in models:
        vals = {j: tallies[j][m]["answer"] + tallies[j][m]["win+err"]
                for j in judges}
        selfj = [j for j in judges if self_of[j] == m]
        others = [vals[j] for j in judges if self_of[j] != m]
        mo = sum(others) / len(others) if others else float("nan")
        p = (1 - mo / 68) ** (1 / MEAN_ROUNDS) if others else float("nan")
        sv = str(vals[selfj[0]]) if selfj else "--"
        print(f"{m[:17]:<18}" + "".join(f"{vals[j]:>9}" for j in judges)
              + f"{sv:>7}{mo:>9.1f}{p:>9.3f}")
    print(f"emp. p = (1 - err/68)^(1/{MEAN_ROUNDS}), using only judges that "
          "are not the model itself")

    print("\nlosses by cause, per judge")
    print(f"{'judge':<12}{'losses':>8}{'answer':>9}{'discrim':>9}{'predict':>9}")
    print("-" * 47)
    agg = Counter()
    for j in judges:
        tot = Counter()
        for c in tallies[j].values():
            tot.update(c)
        L = sum(tot[k] for k in ("answer", "discrimination", "prediction"))
        if not L:
            continue
        for k in ("answer", "discrimination", "prediction"):
            agg[k] += tot[k]
        agg["L"] += L
        print(f"{j[:11]:<12}{L:>8}{tot['answer']/L:>9.0%}"
              f"{tot['discrimination']/L:>9.0%}{tot['prediction']/L:>9.0%}")
    if agg["L"]:
        print(f"{'mean':<12}{agg['L']//len(judges):>8}"
              f"{agg['answer']/agg['L']:>9.0%}"
              f"{agg['discrimination']/agg['L']:>9.0%}"
              f"{agg['prediction']/agg['L']:>9.0%}")

    print(f"\nlosses by size, mean across {len(judges)} judges "
          f"(use for the stacked-bar figure)")
    print(f"{'N':>6}{'answer':>9}{'discrim':>9}{'predict':>9}{'total':>9}")
    print("-" * 42)
    for N in sorted(sizes[judges[0]]):
        a = sum(sizes[j][N]["answer"] for j in judges) / len(judges)
        d = sum(sizes[j][N]["discrimination"] for j in judges) / len(judges)
        pr = sum(sizes[j][N]["prediction"] for j in judges) / len(judges)
        print(f"{N:>6}{a:>9.1f}{d:>9.1f}{pr:>9.1f}{a+d+pr:>9.1f}")

    print("\ngames won despite an answer error")
    for j in judges:
        n = sum(c["win+err"] for c in tallies[j].values())
        by = {m: tallies[j][m]["win+err"] for m in models
              if tallies[j][m]["win+err"]}
        print(f"  {j:<10}{n:>3}  {by if by else ''}")

    if len(judges) > 1:
        print("\ninter-judge agreement on shared (question, document) pairs")
        for i, a in enumerate(judges):
            for b in judges[i + 1:]:
                k = set(caches[a]) & set(caches[b])
                if k:
                    ag = sum(caches[a][x] == caches[b][x] for x in k)
                    print(f"  {a:<9} vs {b:<9} {ag}/{len(k)} = {ag/len(k):.1%}")


# --------------------------------------------------------------------------

def main() -> int:
    global CACHE
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="gemini-3.8-flash",
                    help="model name from config/models.yaml")
    ap.add_argument("--cache",
                    help="cache filename under results/adjudicated/, "
                         "e.g. judgments_gpt.json")
    ap.add_argument("--limit", type=int,
                    help="stop after this many new judgements")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent judge calls; raise until rate-limited")
    ap.add_argument("--judge-effort", default="low",
                    help="reasoning level for the judge; the task is one yes/no "
                         "on one short document, so 'low' is usually enough")
    ap.add_argument("--dry-run", action="store_true",
                    help="count work, call nothing")
    ap.add_argument("--report", action="store_true",
                    help="report from one cache, no API calls")
    ap.add_argument("--all-judges", action="store_true",
                    help="compare every cache in results/adjudicated/ at once")
    args = ap.parse_args()

    if args.cache:
        CACHE = (Path(args.cache)
                 if ("/" in args.cache or "\\" in args.cache)
                 else CACHE_DIR / args.cache)

    pool, _ = load_corpus(ROOT)

    if args.all_judges:
        report_all(pool, CACHE_DIR)
        return 0

    cache = load_cache(CACHE)

    if args.report:
        if not cache:
            avail = [f.name for f in CACHE_DIR.glob("judgments*.json")]
            print(f"no judgements in {CACHE}. Available caches: {avail}",
                  file=sys.stderr)
            return 1
        report(pool, cache, label=CACHE.stem)
        return 0

    pairs = needed(pool)
    todo = [x for x in pairs if key(*x) not in cache]
    print(f"{len(pairs)} (question, document) pairs; {len(todo)} not yet judged "
          f"({len(cache)} cached in {CACHE.name})")
    if args.dry_run or not todo:
        if not args.dry_run and cache:
            report(pool, cache, label=CACHE.stem)
        return 0

    _, models = load_config(ROOT)
    spec = next((m for m in models if m.name == args.judge), None)
    if spec is None:
        sys.exit(f"no model {args.judge!r} in config")
    print(f"judging with {spec.name} at effort={args.judge_effort}, "
          f"{args.workers} workers")
    client = build_client(spec)
    try:
        adjudicate(client, pool, todo, cache, args.limit,
                   workers=args.workers, effort=args.judge_effort, path=CACHE)
    finally:
        if hasattr(client, "close"):
            client.close()
    report(pool, cache, label=CACHE.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())