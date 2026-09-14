#!/usr/bin/env python3
"""
List answer errors under the frozen prompt, for inspection.

An answer error is a logged (question, target, answer) triple where an
independent judge disagrees with the answer the answerer gave. This prints them
with the number of judges disagreeing, so the clearest cases sort first.

    python scripts/answer_errors.py --model claude-opus-5
    python scripts/answer_errors.py --model claude-opus-5 --unanimous
    python scripts/answer_errors.py                      # all models, summary
"""
from __future__ import annotations
import argparse, glob, hashlib, io, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from logn import prompts                                   # noqa: E402
from logn.game import load_corpus                          # noqa: E402

key = lambda q, d: hashlib.sha256(f"{d}\x00{q}".encode()).hexdigest()[:20]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--unanimous", action="store_true",
                    help="only where every judge disagrees with the answer")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    pool, _ = load_corpus(ROOT)
    judges = {}
    for f in (ROOT / "results" / "adjudicated").glob("judgments*.json"):
        judges[f.stem.replace("judgments_", "")] = json.loads(
            f.read_text(encoding="utf-8"))
    if not judges:
        sys.exit("no judgement caches found")

    rows, per_model = [], Counter()
    for f in glob.glob(str(ROOT / "results" / "raw" / "*" / "*" / "*.json")):
        g = json.load(io.open(f, encoding="utf-8"))
        if g.get("aborted") or g.get("prompt_version") != prompts.PROMPT_VERSION:
            continue
        if args.model and g["model"] != args.model:
            continue
        tid = g["target_id"]
        for h in g["history"]:
            k = key(h["question"], tid)
            v = [c[k] for c in judges.values() if k in c]
            if not v:
                continue
            dis = sum(1 for x in v if x != h["answer"])
            if dis and (not args.unanimous or dis == len(v)):
                rows.append((dis, len(v), g["model"], g["size"], h["round"],
                             h["answer"], tid, h["question"]))
                per_model[g["model"]] += 1

    rows.sort(key=lambda r: (-r[0], r[2]))
    print(f"{len(rows)} answer errors under prompt {prompts.PROMPT_VERSION}")
    for m, n in per_model.most_common():
        print(f"  {m:<18}{n}")
    print()
    for dis, tot, m, N, rd, ans, tid, q in rows[:args.limit]:
        d = pool[tid]
        print(f"[{dis}/{tot} judges disagree] {m}  N={N} R{rd}  "
              f"answered {'Yes' if ans else 'No'}")
        print(f"  Q: {q[:120]}")
        print(f"  target: {d['title']} -- {d['text'][:150]}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())