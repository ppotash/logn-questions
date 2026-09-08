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

A caveat that cannot be engineered away: for vague questions there may be no
fact of the matter, so a judge disagreeing with the answerer is not proof the
answerer erred. Both are estimates of the same underlying agreement quantity.
Judge disagreement is reported as such, not as ground truth. Using a judge that
is also one of the evaluated models would bias toward agreement with itself, so
prefer a judge outside the study, or accept the bias and say so.

    python scripts/errors.py --judge gemini-3.8-flash --dry-run
    python scripts/errors.py --judge gemini-3.8-flash
    python scripts/errors.py --report            # from the cache, no API calls

Judgments are cached in results/adjudicated/judgments.json and reused, so the
script is resumable and re-running is free.
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
CACHE = ROOT / "results" / "adjudicated" / "judgments.json"

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


def key(question: str, doc_id: str) -> str:
    return hashlib.sha256(f"{doc_id}\x00{question}".encode()).hexdigest()[:20]


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(c: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c), encoding="utf-8")


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
    guessed document on losses where the target was not obviously eliminated."""
    want = set()
    for g in games():
        for h in g["history"]:
            want.add((h["question"], g["target_id"]))
        if not g["won"] and g["guess"]:
            gid = g["doc_ids"][g["guess"] - 1]
            for h in g["history"]:
                want.add((h["question"], gid))
    return sorted(want)


def adjudicate(client, pool, pairs, cache, limit=None, workers=8,
               effort="low", max_tokens=2000):
    """Judge pairs concurrently.

    Each judgment is independent, so this parallelises cleanly. httpx.Client is
    thread-safe, so one client is shared. The cache is guarded by a lock and
    flushed periodically, which also makes the run resumable: a Ctrl-C keeps
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
                        save_cache(cache)
                        el = time.perf_counter() - state["t0"]
                        rate = n / el
                        left = (len(todo) - n) / rate if rate else 0
                        print(f"  {n}/{len(todo)} judged  "
                              f"{rate:.1f}/s  ~{left/60:.0f} min left"
                              + (f"  ({state['failed']} failed)"
                                 if state["failed"] else ""),
                              flush=True)
        except KeyboardInterrupt:
            print("\n  interrupted; saving what is done", file=sys.stderr)
            for f in futures:
                f.cancel()

    save_cache(cache)
    return state["done"]


def classify(g, pool, cache):
    """Returns (category, round_of_death or None)."""
    tid = g["target_id"]
    for h in g["history"]:
        k = key(h["question"], tid)
        if k not in cache:
            return ("unjudged", None)
        if cache[k] != h["answer"]:
            return ("answer", h["round"])
    if g["won"]:
        return ("win", None)
    if not g["guess"]:
        return ("no guess", None)

    gid = g["doc_ids"][g["guess"] - 1]
    consistent = True
    for h in g["history"]:
        k = key(h["question"], gid)
        if k not in cache:
            return ("unjudged", None)
        if cache[k] != h["answer"]:
            consistent = False
            break
    return ("discrimination" if consistent else "prediction", None)


def report(pool, cache):
    by_model = defaultdict(Counter)
    by_size = defaultdict(Counter)
    death = defaultdict(list)
    for g in games():
        cat, rd = classify(g, pool, cache)
        by_model[g["model"]][cat] += 1
        by_size[g["size"]][cat] += 1
        if cat == "answer":
            death[g["model"]].append(rd / g["rounds"])

    cats = ["win", "answer", "discrimination", "prediction", "no guess", "unjudged"]
    print(f"\n{'model':<18}" + "".join(f"{c[:6]:>9}" for c in cats) + f"{'ans err/game':>14}")
    print("-" * 90)
    for m in sorted(by_model):
        c = by_model[m]; n = sum(c.values())
        rate = c["answer"] / n
        print(f"{m[:17]:<18}" + "".join(f"{c[k]:>9}" for k in cats)
              + f"{rate:>14.0%}")

    print(f"\n{'N':<18}" + "".join(f"{c[:6]:>9}" for c in cats))
    print("-" * 76)
    for n_ in sorted(by_size):
        c = by_size[n_]
        print(f"{n_:<18}" + "".join(f"{c[k]:>9}" for k in cats))

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


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", default="gemini-3.8-flash",
                    help="model name from config/models.yaml")
    ap.add_argument("--limit", type=int, help="stop after this many new judgments")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent judge calls; raise until rate-limited")
    ap.add_argument("--judge-effort", default="low",
                    help="reasoning level for the judge. The task is a single "
                         "yes/no on one short document, so 'low' is usually "
                         "enough and is several times faster.")
    ap.add_argument("--dry-run", action="store_true", help="count work, call nothing")
    ap.add_argument("--report", action="store_true", help="report from cache only")
    args = ap.parse_args()

    pool, _ = load_corpus(ROOT)
    cache = load_cache()

    if args.report:
        report(pool, cache)
        return 0

    pairs = needed(pool)
    todo = [x for x in pairs if key(*x) not in cache]
    print(f"{len(pairs)} (question, document) pairs; {len(todo)} not yet judged "
          f"({len(cache)} cached)")
    if args.dry_run or not todo:
        if not args.dry_run:
            report(pool, cache)
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
                   workers=args.workers, effort=args.judge_effort)
    finally:
        if hasattr(client, "close"):
            client.close()
    report(pool, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())