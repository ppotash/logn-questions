#!/usr/bin/env python3
"""Play one game end to end and report what went right or wrong.

Throwaway diagnostic, not part of the pipeline. It exists to answer five
questions before you commit to a full sweep:

  1. Does the QUESTION:/GUESS: format hold in practice?
  2. Does the answerer actually reply with one word?
  3. Does prompt caching engage, and by how much?
  4. What do reasoning tokens cost at this N?
  5. Does the model win at all?

Usage:
    python scripts/pilot.py                    # N=8, first target
    python scripts/pilot.py --size 64 --run 3
    python scripts/pilot.py --size 8 --repeat 2   # second game to test caching
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # works without pip install -e .

from logn import prompts                                        # noqa: E402
from logn.providers.anthropic import AnthropicClient            # noqa: E402
from logn.providers.base import Usage                           # noqa: E402

POOL = ROOT / "data" / "pool.jsonl"
MANIFEST = ROOT / "data" / "docsets" / "manifest.json"


def load(size: int, run: int):
    pool = {}
    with POOL.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pool[d["id"]] = d
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = man["sizes"][str(size)]
    usable = entry["max_usable_runs"]
    if run >= usable:
        sys.exit(f"N={size} has only {usable} usable runs (0-{usable - 1})")
    docs = [pool[i] for i in entry["doc_ids"]]
    target_id = entry["targets"][run]
    return docs, target_id, entry["doc_ids"].index(target_id) + 1, entry["rounds"]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--reasoning", type=int, default=0,
                    help="thinking budget; 0 disables extended thinking")
    ap.add_argument("--repeat", type=int, default=1,
                    help="play the same game N times; the 2nd+ shows cross-game caching")
    args = ap.parse_args()

    docs, target_id, target_no, rounds = load(args.size, args.run)
    target = next(d for d in docs if d["id"] == target_id)

    print(f"N={args.size}  rounds={rounds}  target=[{target_no}] {target['title']}")
    print(f"model={args.model}  reasoning={args.reasoning or 'off'}  "
          f"prompt_version={prompts.PROMPT_VERSION}\n")

    client = AnthropicClient(args.model, args.model,
                             reasoning_budget=args.reasoning)
    total = Usage()
    modes: list[str] = []

    for game in range(args.repeat):
        if args.repeat > 1:
            print(f"--- game {game + 1} of {args.repeat} ---")
        history: list[tuple[str, bool]] = []

        for rnd in range(1, rounds + 1):
            p = prompts.qbot_prompt(docs, history, rnd, rounds)
            c = client.complete(p, max_tokens=args.max_tokens, temperature=0.0)
            total += c.usage
            q = prompts.parse_question(c.text)
            modes.append(f"q:{q.mode}")
            if not q.ok:
                print(f"  R{rnd} QUESTION UNPARSEABLE: {c.text[:160]!r}")
                return 1

            ap_ = prompts.abot_prompt(target, q.value)
            ca = client.complete(ap_, max_tokens=16, temperature=0.0)
            total += ca.usage
            a = prompts.parse_answer(ca.text)
            modes.append(f"a:{a.mode}")
            if not a.ok:
                print(f"  R{rnd} ANSWER UNPARSEABLE: {ca.text[:160]!r}")
                return 1

            history.append((q.value, a.value))
            print(f"  R{rnd} [{q.mode:<6}] {q.value}")
            print(f"       -> {'Yes' if a.value else 'No'} "
                  f"({ca.text.strip()!r}, {a.mode})"
                  f"   cache_read={c.usage.cache_read_tokens}")

        gp = prompts.qbot_guess_prompt(docs, history, rounds)
        cg = client.complete(gp, max_tokens=args.max_tokens, temperature=0.0)
        total += cg.usage
        g = prompts.parse_guess(cg.text, len(docs))
        modes.append(f"g:{g.mode}")
        won = g.ok and g.value == target_no
        print(f"  GUESS [{g.mode}] {g.value}  actual {target_no}  "
              f"-> {'WIN' if won else 'LOSS'}\n")

    rates = {"input": 5.0, "cache_read": 0.5, "cache_write": 6.25, "output": 25.0}
    print(f"tokens: in={total.input_tokens:,} cache_read={total.cache_read_tokens:,} "
          f"cache_write={total.cache_write_tokens:,} out={total.output_tokens:,} "
          f"reasoning~{total.reasoning_tokens:,}")
    print(f"cost (at Opus rates): ${total.cost(rates):.4f}")
    strict = sum(1 for m in modes if m.endswith("strict"))
    print(f"format: {strict}/{len(modes)} strict")
    if total.cache_read_tokens == 0:
        print("no cache hits: expected below ~1024 tokens of documents "
              "(roughly N<8); investigate if N is larger")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())