#!/usr/bin/env python3
"""
Inspect the documents that will be, or were, used in the experiment.

    python inspect_pool.py                       # 20 random docs from the 1024
    python inspect_pool.py --n 40                # more
    python inspect_pool.py --seed 7              # a reproducible sample
    python inspect_pool.py --size 128            # only the docs used at N=128

    python inspect_pool.py --size 8 --all --run 0
        Every document in the N=8 set, numbered exactly as the questioner saw
        it, with run 0's target marked. This is the view to read alongside a
        game transcript.

    python inspect_pool.py --short 15            # the thinnest docs
    python inspect_pool.py --check               # near-duplicate topic scan
    python inspect_pool.py --size 8 --titles     # one line per doc

Read-only. Never touches pool.jsonl or manifest.json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
POOL_PATH = REPO_ROOT / "data" / "pool.jsonl"
MANIFEST_PATH = REPO_ROOT / "data" / "docsets" / "manifest.json"

# Words too common to indicate two articles share a topic.
STOP = set("""a an the of in on at to for and or but is are was were be been being
by with from as that this it its his her their they he she we you i not no than
then there here which who whom whose what when where why how all any both each
few more most other some such only own same so too very can will just also into
over under after before during about between against through
first second new one two three years year known also used other""".split())


def load() -> tuple[dict[str, dict], dict]:
    if not (POOL_PATH.exists() and MANIFEST_PATH.exists()):
        sys.exit(f"missing artifacts under {REPO_ROOT / 'data'}; run build_corpus.py first")
    pool = {}
    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pool[d["id"]] = d
    return pool, json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: int(len(s.split()) * 1.33)


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP}


def show(pool, ids, sample_ids, count, target_id=None, header="") -> None:
    """ids: the full set, defining prompt numbering. sample_ids: what to print."""
    pos = {d: i for i, d in enumerate(ids)}
    print(f"\n{header or f'{len(sample_ids)} of {len(ids)} documents'}")
    print("=" * 78)
    for did in sample_ids:
        d = pool[did]
        n = pos[did] + 1                      # 1-based, matching the prompt
        mark = "   <-- TARGET" if did == target_id else ""
        body = f"{d['title']}. {d['text']}"
        print(f"\n[{n:>2}] {d['title']}{mark}")
        print(f"     id={did}  {count(body)} tokens")
        line = ""
        for word in d["text"].split():
            if len(line) + len(word) > 74:
                print(f"     {line}")
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            print(f"     {line}")


def show_titles(pool, ids, count, target_id=None) -> None:
    print(f"\n{len(ids)} documents, in prompt order")
    print("=" * 78)
    for i, did in enumerate(ids, 1):
        d = pool[did]
        mark = " <-- TARGET" if did == target_id else ""
        toks = count(f"{d['title']}. {d['text']}")
        print(f"[{i:>4}] {toks:>4}t  {d['title']}{mark}")


def check_dupes(pool, ids, threshold: float) -> None:
    """Crude near-duplicate scan: Jaccard overlap of content words."""
    words = {d: content_words(f"{pool[d]['title']} {pool[d]['text']}") for d in ids}
    hits = []
    idlist = list(ids)
    for i, a in enumerate(idlist):
        wa = words[a]
        if not wa:
            continue
        for b in idlist[i + 1:]:
            wb = words[b]
            inter = len(wa & wb)
            if inter < 4:
                continue
            j = inter / len(wa | wb)
            if j >= threshold:
                hits.append((j, a, b))
    hits.sort(reverse=True)

    print(f"\nnear-duplicate scan over {len(ids)} docs (Jaccard >= {threshold})")
    print("=" * 78)
    if not hits:
        print("no pairs above threshold - topically well spread")
    else:
        for j, a, b in hits[:25]:
            print(f"  {j:.2f}  {pool[a]['title']}")
            print(f"        {pool[b]['title']}")
        if len(hits) > 25:
            print(f"  ... and {len(hits) - 25} more")
        print("\nA handful of pairs is normal. Many pairs sharing one theme means "
              "questions can key on that theme; note it as a property of the set.")

    freq = Counter(w for d in ids for w in words[d])
    top = [(w, c) for w, c in freq.most_common(12) if c > len(ids) * 0.02]
    if top:
        print("\nmost common content words: "
              + ", ".join(f"{w} ({c})" for w, c in top))


def main() -> int:
    # Windows consoles default to cp1252 and will crash on accented titles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=1024,
                    help="which docset to inspect (default 1024)")
    ap.add_argument("--n", type=int, default=20, help="how many docs to sample")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed; omit for a different sample each time")
    ap.add_argument("--all", action="store_true",
                    help="show every doc in the set, in prompt order")
    ap.add_argument("--run", type=int, default=None,
                    help="mark the target for this run index (0-based)")
    ap.add_argument("--titles", action="store_true",
                    help="one line per document instead of full text")
    ap.add_argument("--short", type=int, metavar="K",
                    help="show the K shortest docs instead of sampling")
    ap.add_argument("--check", action="store_true", help="near-duplicate scan")
    ap.add_argument("--threshold", type=float, default=0.12,
                    help="Jaccard threshold for --check")
    args = ap.parse_args()

    pool, manifest = load()
    key = str(args.size)
    if key not in manifest["sizes"]:
        sys.exit(f"no docset for N={args.size}; "
                 f"have {sorted(manifest['sizes'], key=int)}")
    entry = manifest["sizes"][key]
    ids = entry["doc_ids"]
    count = tokenizer()

    target_id = None
    if args.run is not None:
        usable = entry["max_usable_runs"]
        if not (0 <= args.run < usable):
            sys.exit(f"N={args.size} has {usable} usable runs (0-{usable - 1})")
        target_id = entry["targets"][args.run]

    if args.check:
        check_dupes(pool, ids, args.threshold)
        return 0

    if args.titles:
        show_titles(pool, ids, count, target_id)
        return 0

    if args.all:
        chosen = ids
        header = (f"all {len(ids)} documents at N={args.size}, in prompt order"
                  + (f"  (run {args.run})" if args.run is not None else ""))
        if len(ids) > 64:
            print(f"note: {len(ids)} documents is a lot of output; "
                  "--titles gives one line each", file=sys.stderr)
    elif args.short:
        chosen = sorted(ids, key=lambda d: count(pool[d]["text"]))[:args.short]
        header = (f"{args.short} shortest of {len(ids)} documents - the ones most "
                  "likely to be content-free stubs")
    else:
        rng = random.Random(args.seed) if args.seed is not None else random.Random()
        chosen = rng.sample(ids, min(args.n, len(ids)))
        header = f"{len(chosen)} of {len(ids)} documents, sampled"

    show(pool, ids, chosen, count, target_id, header)

    if target_id is not None and target_id not in chosen:
        d = pool[target_id]
        print(f"\n(target for run {args.run} is [{ids.index(target_id) + 1}] "
              f"{d['title']}, not shown above; use --all to include it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())