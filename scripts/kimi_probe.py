#!/usr/bin/env python3
"""
Locate the document(s) triggering Moonshot's content filter.

Moonshot rejects the N=512 prompt with error 400 / content_filter. Measured
over ten identical requests the block rate was 10/10, but two earlier requests
with the same prompt passed -- so the filter is heavily biased toward blocking
without being fully deterministic.

That asymmetry dictates the search rule:

    BLOCKED  -> trusted immediately (false blocks appear rare)
    PASS     -> only trusted after --repeats consecutive passes

A single unconfirmed pass would send the bisection down the wrong half, so
passes are the expensive side. Blocked requests are not billed, and passing
requests use max_tokens=64, so the whole search costs cents.

    python scripts/kimi_probe.py                    # bisect the N=512 set
    python scripts/kimi_probe.py --size 1024        # or another size
    python scripts/kimi_probe.py --repeats 5        # stricter pass rule
    python scripts/kimi_probe.py --verify-only      # just measure block rate

If more than one trigger exists, removing the first reveals the next: the
script re-tests the full set with all found triggers removed and reports
whether it is clean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn import prompts                                    # noqa: E402
from logn.game import game_spec, load_corpus                 # noqa: E402
from logn.providers.base import ProviderError                # noqa: E402
from logn.providers.openai_compat import MoonshotClient      # noqa: E402

CALLS = 0


def is_blocked(client, docs, repeats: int, rounds: int) -> bool:
    """True if this document set trips the filter.

    Asymmetric by design: one block is conclusive, a pass must repeat.
    """
    global CALLS
    p = prompts.qbot_prompt(docs, [], 1, rounds)
    for _ in range(repeats):
        CALLS += 1
        try:
            client.complete(p, max_tokens=64, temperature=0.0)
        except ProviderError as e:
            if "content_filter" in str(e) or "high risk" in str(e):
                return True
            raise                      # a real error is not a filter hit
    return False


def bisect(client, docs, repeats, rounds, depth=0):
    """Narrow a blocked set to one document. Returns the document, or None if
    the block does not localise (which suggests an interaction rather than a
    single trigger)."""
    while len(docs) > 1:
        mid = len(docs) // 2
        left, right = docs[:mid], docs[mid:]
        if is_blocked(client, left, repeats, rounds):
            docs = left
        elif is_blocked(client, right, repeats, rounds):
            docs = right
        else:
            print(f"    neither half blocks at n={len(docs)}: the trigger is "
                  f"an interaction between documents, not a single one",
                  file=sys.stderr)
            return None
        print(f"    narrowed to {len(docs):>4}  ({CALLS} calls)", flush=True)
    return docs[0]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3,
                    help="consecutive passes required to call a set clean")
    ap.add_argument("--max-triggers", type=int, default=4,
                    help="give up after this many; more suggests a partial fix "
                         "is not worth it")
    ap.add_argument("--verify-only", action="store_true",
                    help="measure the block rate on the full set and exit")
    args = ap.parse_args()

    pool, man = load_corpus(ROOT)
    docs, _, rounds = game_spec(pool, man, args.size, 0)
    targets = set(man["sizes"][str(args.size)]["targets"])
    client = MoonshotClient("kimi", "kimi-k3")

    print(f"N={args.size}, {len(docs)} documents, pass rule = "
          f"{args.repeats} consecutive")

    if args.verify_only:
        blocks = 0
        p = prompts.qbot_prompt(docs, [], 1, rounds)
        for i in range(10):
            try:
                client.complete(p, max_tokens=64, temperature=0.0)
                print(f"  {i}: PASS")
            except ProviderError as e:
                hit = "content_filter" in str(e) or "high risk" in str(e)
                blocks += hit
                print(f"  {i}: {'BLOCKED' if hit else 'ERR ' + str(e)[:70]}")
        print(f"\nblocked {blocks}/10")
        return 0

    if not is_blocked(client, docs, args.repeats, rounds):
        print("full set is not blocked; nothing to find")
        return 0
    print("full set blocked, bisecting\n")

    found = []
    remaining = list(docs)
    while len(found) < args.max_triggers:
        print(f"  search {len(found)+1}:")
        hit = bisect(client, remaining, args.repeats, rounds)
        if hit is None:
            break
        pos = next(i for i, d in enumerate(docs) if d["id"] == hit["id"]) + 1
        is_target = hit["id"] in targets
        found.append((pos, hit, is_target))
        print(f"\n  TRIGGER {len(found)}: [{pos}] {hit['title']}  ({hit['id']})")
        print(f"    target? {'YES -- cannot substitute' if is_target else 'no'}")
        print(f"    {hit['text'][:300]}\n")

        remaining = [d for d in remaining if d["id"] != hit["id"]]
        if not is_blocked(client, remaining, args.repeats, rounds):
            print(f"  set is clean after removing {len(found)} document(s)")
            break
        print(f"  still blocked; searching for another\n")

    print(f"\n{'=' * 70}")
    print(f"{len(found)} trigger(s), {CALLS} API calls")
    if not found:
        print("no single document accounts for the block")
        return 0

    spare = next((i for i in pool if i not in {d["id"] for d in docs}), None)
    print("\nconfig/models.yaml, under the kimi-k3 entry:")
    print("    doc_substitutions:")
    for pos, d, tgt in found:
        note = "  # IS A TARGET -- substitution invalid" if tgt else ""
        print(f"      {d['id']}: {spare}{note}")
    if len(found) > 1:
        print(f"\n  Note: {len(found)} distinct triggers. Each needs its own "
              f"spare id; the one shown is a placeholder for the first.")
    if any(t for _, _, t in found):
        print("\n  At least one trigger is a target. Substituting it would "
              "change the game, not the document set. Report the size as "
              "blocked instead.")
    if hasattr(client, "close"):
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())