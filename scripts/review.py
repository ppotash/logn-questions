#!/usr/bin/env python3
"""
Read logged games.

    python scripts/review.py --model claude-opus-5 --size 32 --run 0
        Full transcript: target, every question and answer, the guess, and what
        the guessed document actually was.

    python scripts/review.py --model claude-opus-5 --size 32 --run 0 --docs
        Same, plus every document in the set, so you can adjudicate each
        question by hand. Only practical up to about N=32.

    python scripts/review.py --model claude-opus-5 --size 32 --run 0 --reasoning
        Include the model's thinking traces.

    python scripts/review.py --summary
        One line per game across everything logged, plus per-size aggregates.

    python scripts/review.py --losses
        Transcripts for every losing game, condensed.

    python scripts/review.py --questions --size 32
        Every question asked at one size, grouped by round. The fastest way to
        see whether a model is asking semantic questions, enumerating titles,
        or bisecting alphabetically.

Read-only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results" / "raw"
POOL_PATH = REPO_ROOT / "data" / "pool.jsonl"


def load_pool() -> dict:
    pool = {}
    if POOL_PATH.exists():
        with POOL_PATH.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                pool[d["id"]] = d
    return pool


def load_games(model=None, size=None, run=None) -> list[dict]:
    games = []
    for path in sorted(RESULTS.glob("*/*/run*.json")):
        parts = path.parts
        m, s = parts[-3], int(parts[-2])
        r = int(re.search(r"run(\d+)", path.name).group(1))
        if model and m != model:
            continue
        if size is not None and s != size:
            continue
        if run is not None and r != run:
            continue
        try:
            games.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skipping unreadable {path}", file=sys.stderr)
    return games


def wrap(text: str, indent: int = 7, width: int = 74) -> str:
    out, line = [], ""
    for w in str(text).split():
        if len(line) + len(w) > width:
            out.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    pad = " " * indent
    return f"\n{pad}".join(out)


# --------------------------------------------------------------------------

def classify(q: str) -> str:
    """Cheap strategy guess. Not a substitute for the judge pass in
    adjudicate.py, but enough to spot the obvious cases at a glance."""
    ql = q.lower()
    if q.count(",") >= 4 or "one of the following" in ql or "any of these" in ql:
        return "enumeration"
    if re.search(r"\b(begin|start|first letter|alphabet|before the letter|a[- ]m|n[- ]z)\b", ql):
        return "lexical"
    if re.search(r"\btitle\b", ql) and re.search(r"\b(letter|word|character)\b", ql):
        return "lexical"
    return "semantic"


def transcript(g: dict, pool: dict, show_docs: bool, show_reasoning: bool,
               condensed: bool = False) -> None:
    tgt = pool.get(g["target_id"], {})
    outcome = "WIN" if g["won"] else ("ABORT" if g["aborted"] else "LOSS")
    print("=" * 78)
    print(f"{g['model']}  N={g['size']}  run={g['run']}  {outcome}"
          f"   ${g['cost_usd']:.3f}  {g['duration_s']:.0f}s")
    print(f"target  [{g['target_number']}] {tgt.get('title', g['target_id'])}")
    if g["guess"] is not None and not g["won"]:
        gid = g["doc_ids"][g["guess"] - 1]
        print(f"guessed [{g['guess']}] {pool.get(gid, {}).get('title', gid)}")
    if g["aborted"]:
        print(f"aborted: {g['aborted'][:160]}")

    calls_by = defaultdict(list)
    for c in g["calls"]:
        calls_by[(c["role"], c["round"])].append(c)

    print()
    for h in g["history"]:
        rnd = h["round"]
        qc = calls_by[("questioner", rnd)]
        kind = classify(h["question"])
        modes = ",".join(c["parse_mode"] for c in qc)
        out = sum(c["usage"].get("output_tokens", 0) for c in qc)
        print(f"  R{rnd} [{kind:<11} {modes:<14} {out:>5}t out]")
        print(f"       {wrap(h['question'])}")
        print(f"       -> {'Yes' if h['answer'] else 'No'}")
        if show_reasoning:
            for c in qc:
                if c.get("reasoning_text"):
                    print(f"       ~ thinking: {wrap(c['reasoning_text'][:700], 9)}")
        print()

    if show_docs:
        print("  documents:")
        for i, did in enumerate(g["doc_ids"], 1):
            d = pool.get(did, {})
            mark = " <-- TARGET" if did == g["target_id"] else (
                " <-- GUESS" if g["guess"] == i else "")
            print(f"    [{i:>3}] {d.get('title', did)}{mark}")
            if not condensed:
                print(f"          {wrap(d.get('text', ''), 10)}")
        print()


def summary(games: list[dict]) -> None:
    if not games:
        print("no games logged yet")
        return
    by = defaultdict(list)
    for g in games:
        by[(g["model"], g["size"])].append(g)

    print(f"{'model':<16}{'N':>6}{'games':>7}{'won':>5}{'rate':>7}"
          f"{'$/game':>9}{'out tok':>9}{'sec':>6}{'abort':>7}")
    print("-" * 78)
    tot_cost = 0.0
    for (m, s) in sorted(by, key=lambda k: (k[0], k[1])):
        gs = by[(m, s)]
        won = sum(g["won"] for g in gs)
        cost = sum(g["cost_usd"] for g in gs)
        out = sum(g["usage_total"].get("output_tokens", 0) for g in gs) / len(gs)
        sec = sum(g["duration_s"] for g in gs) / len(gs)
        ab = sum(1 for g in gs if g["aborted"])
        tot_cost += cost
        print(f"{m:<16}{s:>6}{len(gs):>7}{won:>5}{won/len(gs):>7.0%}"
              f"{cost/len(gs):>9.3f}{out:>9,.0f}{sec:>6.0f}{ab:>7}")
    print("-" * 78)
    print(f"{len(games)} games, ${tot_cost:.2f} total")

    # Format compliance and strategy mix are cross-cutting, so report separately.
    modes = defaultdict(int)
    kinds = defaultdict(int)
    trunc = 0
    for g in games:
        for c in g["calls"]:
            modes[c["parse_mode"]] += 1
            if c.get("stop_reason") == "max_tokens":
                trunc += 1
        for h in g["history"]:
            kinds[classify(h["question"])] += 1
    total_calls = sum(modes.values())
    print("parse modes: " + "  ".join(f"{k}={v} ({v/total_calls:.0%})"
                                      for k, v in sorted(modes.items())))
    total_q = sum(kinds.values()) or 1
    print("question kinds: " + "  ".join(f"{k}={v} ({v/total_q:.0%})"
                                         for k, v in sorted(kinds.items())))
    if trunc:
        print(f"WARNING: {trunc} calls hit max_tokens (truncated mid-reply)")


def questions_view(games: list[dict]) -> None:
    by_round = defaultdict(list)
    for g in games:
        for h in g["history"]:
            by_round[h["round"]].append(
                (g["model"], g["run"], h["question"], h["answer"]))
    for rnd in sorted(by_round):
        print(f"\n--- round {rnd} ---")
        for model, run, q, a in by_round[rnd]:
            print(f"  [{classify(q):<11}] {model} run{run} "
                  f"({'Y' if a else 'N'}) {q}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model")
    ap.add_argument("--size", type=int)
    ap.add_argument("--run", type=int)
    ap.add_argument("--docs", action="store_true", help="print the document set too")
    ap.add_argument("--reasoning", action="store_true", help="include thinking traces")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--losses", action="store_true", help="only losing games")
    ap.add_argument("--questions", action="store_true",
                    help="every question, grouped by round")
    args = ap.parse_args()

    games = load_games(args.model, args.size, args.run)
    if not games:
        print("no matching games under results/raw/", file=sys.stderr)
        return 1
    pool = load_pool()

    if args.summary:
        summary(games)
        return 0
    if args.questions:
        questions_view(games)
        return 0
    if args.losses:
        games = [g for g in games if not g["won"]]
        if not games:
            print("no losses")
            return 0
        for g in games:
            transcript(g, pool, args.docs, args.reasoning, condensed=True)
        return 0

    for g in games:
        transcript(g, pool, args.docs, args.reasoning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())