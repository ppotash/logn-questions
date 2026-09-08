#!/usr/bin/env python3
"""
Build the frozen corpus for the log(N)-Questions experiment.

Two artifacts, both meant to be committed and never regenerated:

  data/pool.jsonl              candidate documents (title + lead paragraph),
                               randomly sampled from English Wikipedia
  data/docsets/manifest.json   nested doc sets for each N, plus target
                               assignments, with a hash of the pool

Usage:
    python build_corpus.py --mode shards --shards 8      # ~4 GB cache, fast
    python build_corpus.py --mode stream                 # 0 disk, slow
    python build_corpus.py --verify                      # re-check hashes only
    python build_corpus.py --force ...                   # DESTRUCTIVE, see below

--force rebuilds the pool from a fresh sample, changing every document in the
experiment. Only ever use it before the first API call has been made.

Requires: datasets, huggingface_hub. Optional: tiktoken (for token stats).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"

POOL_SIZE = 4000          # candidate pool; only 1024 are used, rest are slack
MAX_N = 1024
SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# The manifest always contains MAX_RUNS targets per size, generated in
# bit-reversed order so that any power-of-two prefix is evenly stratified.
# The runner takes targets[:min(runs, max_usable_runs)]. Scaling 8 -> 16 -> 64
# runs therefore needs no manifest change and invalidates no completed game.
MAX_RUNS = 64
RUNS_PLANNED = 8          # recorded for reference; the runner's config wins

SEED = 20260901           # change this and you have a different experiment

MIN_WORDS, MAX_WORDS = 60, 140   # ~80-190 tokens; keeps N=1024 under 200K ctx
MIN_SENT_DENSITY = 2.0           # sentence terminators per 100 words

# Paths are anchored to the repo root, not the working directory, so the script
# behaves identically whether you run it from the root or from scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DOCSETS_DIR = DATA_DIR / "docsets"
RESULTS_DIR = REPO_ROOT / "results"

POOL_PATH = DATA_DIR / "pool.jsonl"
MANIFEST_PATH = DOCSETS_DIR / "manifest.json"

# Used only to pre-create the results tree. The runner writes into
# results/raw/{model}/{N}/{run}.json and skips any file that already exists.
MODELS = (
    "claude-opus-5",
    "gpt-5.6-sol",
    "gemini-3.8-flash",
    "grok-4.6",
    "glm-5.2",
    "kimi-k3",
)

# --- rejection patterns ---------------------------------------------------
# Titles for articles that are indexes rather than subjects.
BAD_TITLE = re.compile(
    r"\((disambiguation)\)|^List of |^Index of |^Outline of |^Timeline of "
    r"|^\d{4} in |^Category:|^Template:",
    re.IGNORECASE,
)

# Disambiguation pages, name pages, and section dumps that are really lists.
# Note "may refer to" without the colon: real pages say "may refer to the
# following ..." as often as they say "may refer to:".
LISTY = re.compile(
    r"may refer to|refers to any of|following (?:family|list|people)"
    r"|is a surname|is a given name|redirects here"
    r"|^(?:Cast|Filmography|Discography|Personnel|Track listing|Squad)\b",
    re.IGNORECASE,
)

# Wikipedia unit/convert templates are stripped by the dump, leaving holes like
# "The channel is long" or "depth ranges from to ." Those sentences are
# unanswerable, so drop the document.
GAP = re.compile(
    r"\b(?:is|are|was|were|ranges|measures|stands|covers)\s+"
    r"(?:to|long|wide|deep|tall|high)\b"
    r"|\bfrom\s+to\b|\s\.",
)


# --------------------------------------------------------------------------
# document extraction
# --------------------------------------------------------------------------

def lead_paragraph(text: str) -> str | None:
    """The article's opening paragraph, or None.

    Deliberately does NOT scan forward for the first substantial block. Doing so
    falls through into the first section body when an article has a short or
    missing lead, which produces documents that open with a section header glued
    to the text ("Early life and amateur career Crocker was born in ...") and
    never state what the subject is. Rejecting the article outright is correct:
    there are millions more.
    """
    if not text:
        return None
    first = " ".join(text.split("\n\n", 2)[0].split())
    # Length is judged by rejection_reason, so "no lead at all" and "lead is
    # too short" stay distinguishable in the reject tally.
    return first or None


def sentence_density(para: str) -> float:
    """Sentence terminators per 100 words.

    Prose runs 3-7. Cast lists, disambiguation pages, and other enumerations
    score at or near 0, which is the cleanest single signal separating them.
    """
    words = len(para.split())
    if not words:
        return 0.0
    return len(re.findall(r"[.!?](?:\s|$)", para)) / words * 100


def rejection_reason(title: str, para: str) -> str | None:
    """None if the document is usable, else a short reason for reporting."""
    if BAD_TITLE.search(title):
        return "title"
    if LISTY.search(para):
        return "list-like"
    if GAP.search(para):
        return "template gap"
    n = len(para.split())
    if n < MIN_WORDS:
        return "too short"
    if n > MAX_WORDS:
        return "too long"
    if sentence_density(para) < MIN_SENT_DENSITY:
        return "not prose"
    # Heavy parenthetical clutter is usually IPA / native-script / date soup,
    # which inflates token count without adding anything askable.
    if para.count("(") > 5:
        return "parenthetical clutter"
    return None


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def iter_articles(mode: str, n_shards: int, rng: random.Random):
    """Yield (title, text) from Wikipedia, either streamed or from random shards."""
    from datasets import load_dataset

    if mode == "stream":
        print("streaming full dataset (no disk use, expect this to take a while)",
              file=sys.stderr)
        ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
        for rec in ds:
            yield rec["title"], rec["text"]
        return

    from huggingface_hub import hf_hub_download, list_repo_files

    files = sorted(
        f for f in list_repo_files(DATASET, repo_type="dataset")
        if f.startswith(f"{CONFIG}/") and f.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet shards found for {DATASET}/{CONFIG}")

    # Random shards, not the first k. Shards are ordered by page id, so taking
    # a prefix would bias toward older/larger articles.
    chosen = rng.sample(files, k=min(n_shards, len(files)))
    print(f"using {len(chosen)}/{len(files)} shards", file=sys.stderr)

    paths = [hf_hub_download(DATASET, filename=f, repo_type="dataset") for f in chosen]
    ds = load_dataset("parquet", data_files=paths, split="train")
    for rec in ds:
        yield rec["title"], rec["text"]


def build_pool(mode: str, n_shards: int) -> list[dict]:
    """Reservoir-sample POOL_SIZE acceptable documents in a single pass."""
    rng = random.Random(SEED)
    reservoir: list[dict] = []
    kept = 0
    seen_titles: set[str] = set()
    rejects: Counter[str] = Counter()

    for i, (title, text) in enumerate(iter_articles(mode, n_shards, rng)):
        if i and i % 100_000 == 0:
            print(f"  scanned {i:,} articles, {kept:,} usable", file=sys.stderr)

        if title in seen_titles:
            rejects["duplicate title"] += 1
            continue

        para = lead_paragraph(text or "")
        if para is None:
            rejects["no usable lead"] += 1
            continue

        reason = rejection_reason(title, para)
        if reason is not None:
            rejects[reason] += 1
            continue

        seen_titles.add(title)
        kept += 1
        doc = {"title": title, "text": para}

        # Classic reservoir: every usable doc has an equal chance of ending up
        # in the pool, without knowing the corpus size in advance.
        if len(reservoir) < POOL_SIZE:
            reservoir.append(doc)
        else:
            j = rng.randrange(kept)
            if j < POOL_SIZE:
                reservoir[j] = doc

    total = kept + sum(rejects.values())
    print(f"\nscanned {total:,} articles, {kept:,} usable "
          f"({kept / max(1, total):.1%})", file=sys.stderr)
    for reason, n in rejects.most_common():
        print(f"    rejected {n:>8,}  {reason}", file=sys.stderr)

    if len(reservoir) < MAX_N:
        raise RuntimeError(
            f"only found {len(reservoir)} usable docs, need at least {MAX_N}. "
            "Raise --shards."
        )

    # Stable ids assigned after sampling, so the pool file is order-independent.
    reservoir.sort(key=lambda d: d["title"])
    for k, doc in enumerate(reservoir):
        doc["id"] = f"w{k:05d}"

    return reservoir


# --------------------------------------------------------------------------
# doc sets
# --------------------------------------------------------------------------

def _bitrev(i: int, bits: int) -> int:
    """Reverse the low `bits` bits of i."""
    r = 0
    for _ in range(bits):
        r = (r << 1) | (i & 1)
        i >>= 1
    return r


def build_manifest(pool: list[dict], pool_sha: str) -> dict:
    """Nested doc sets + a prefix-extensible target sequence per size.

    Nesting matters: docset(512) is a strict subset of docset(1024), so a change
    in accuracy between sizes is attributable to N rather than to one doc set
    happening to be easier than another.

    Targets are emitted in bit-reversed order over MAX_RUNS buckets. Any
    power-of-two prefix is evenly stratified across the document set, so
    targets[:8] covers one document per octile, targets[:16] one per sixteenth,
    and the first is a strict prefix of the second. Each size draws from its own
    RNG, so adding sizes or changing the run count never perturbs another size.
    """
    order = [d["id"] for d in pool]
    random.Random(SEED + 1).shuffle(order)      # dedicated RNG: shuffle only
    order = order[:MAX_N]

    bits = MAX_RUNS.bit_length() - 1
    sizes = {}
    for N in SIZES:
        ids = order[:N]
        rng = random.Random(f"{SEED}:targets:{N}")   # independent per size

        targets = []
        for run in range(MAX_RUNS):
            bucket = _bitrev(run, bits)
            lo = bucket * N // MAX_RUNS
            hi = max(lo + 1, (bucket + 1) * N // MAX_RUNS)
            targets.append(ids[rng.randrange(lo, min(hi, N))])

        # A target is never reused: there are only N documents, so a size with
        # N < MAX_RUNS simply runs fewer games. Bit-reversal guarantees
        # targets[:min(MAX_RUNS, N)] are distinct for power-of-two N.
        usable = min(MAX_RUNS, N)
        assert len(set(targets[:usable])) == usable, (
            f"N={N}: first {usable} targets are not distinct"
        )

        sizes[str(N)] = {
            "doc_ids": ids,
            "targets": targets,
            "rounds": N.bit_length() - 1,          # log2(N)
            "max_usable_runs": usable,
        }

    return {
        "seed": SEED,
        "dataset": f"{DATASET}:{CONFIG}",
        "pool_sha256": pool_sha,
        "pool_size": len(pool),
        "max_runs": MAX_RUNS,
        "runs_planned": RUNS_PLANNED,
        "target_order": "bit-reversed; targets[:2^k] is stratified into 2^k strata",
        "sizes": sizes,
    }


# --------------------------------------------------------------------------
# io + reporting
# --------------------------------------------------------------------------

def ensure_dirs() -> None:
    """Create every directory the pipeline writes to, idempotently."""
    made = []
    targets = [DATA_DIR, RAW_DIR, DOCSETS_DIR,
               RESULTS_DIR / "adjudicated", RESULTS_DIR / "tables"]
    for model in MODELS:
        for N in SIZES:
            targets.append(RESULTS_DIR / "raw" / model / str(N))

    for d in targets:
        if not d.exists():
            made.append(d)
        d.mkdir(parents=True, exist_ok=True)

    # The HF shard cache and any dump files land here; they are large and
    # regenerable, so keep them out of git.
    gitignore = RAW_DIR / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")

    if made:
        print(f"created {len(made)} directories under {REPO_ROOT}", file=sys.stderr)


def write_pool(pool: list[dict]) -> str:
    h = hashlib.sha256()
    with POOL_PATH.open("w", encoding="utf-8") as f:
        for doc in pool:
            line = json.dumps(doc, ensure_ascii=False, sort_keys=True)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def load_pool() -> tuple[list[dict], str]:
    h = hashlib.sha256()
    pool = []
    with POOL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            h.update(line.encode("utf-8"))
            pool.append(json.loads(line))
    return pool, h.hexdigest()


def token_report(pool: list[dict], manifest: dict) -> None:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        count = lambda s: len(enc.encode(s))
        basis = "tiktoken cl100k_base"
    except Exception:
        count = lambda s: int(len(s.split()) * 1.33)
        basis = "estimated (1.33 x words); pip install tiktoken for exact"

    by_id = {d["id"]: d for d in pool}
    used = manifest["sizes"][str(MAX_N)]["doc_ids"]
    per_doc = [count(f"{by_id[i]['title']}. {by_id[i]['text']}") for i in used]
    per_doc.sort()
    mean = sum(per_doc) / len(per_doc)

    print(f"\ntoken stats over the {MAX_N} docs actually used  [{basis}]")
    print(f"  mean {mean:.0f}   p50 {per_doc[len(per_doc)//2]}   "
          f"p95 {per_doc[int(len(per_doc)*0.95)]}   max {per_doc[-1]}")

    biggest = sum(per_doc) + 200
    print(f"  full N=1024 doc block: ~{biggest:,} tokens")
    if biggest > 200_000:
        print("  WARNING: over 200K. Gemini and Grok double their rate for the "
              "entire request at that point. Tighten MAX_WORDS.")
    elif biggest > 170_000:
        print("  NOTE: close to the 200K long-context threshold; little headroom.")
    else:
        print("  comfortably under the 200K long-context threshold.")


def game_report(manifest: dict) -> None:
    print(f"\ngames per model at runs = {RUNS_PLANNED}:")
    total = 0
    for N in SIZES:
        n = min(RUNS_PLANNED, manifest["sizes"][str(N)]["max_usable_runs"])
        total += n
        note = "  (capped: only N documents available)" if n < RUNS_PLANNED else ""
        print(f"  N={N:<5} {n:>3} games{note}")
    print(f"  total {total} games per model, {total * len(MODELS)} across all models")


def main() -> int:
    # Windows consoles default to cp1252 and will crash on accented titles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["shards", "stream"], default="shards",
                    help="shards: download k random parquet shards (~500MB each). "
                         "stream: no disk, full pass over the dataset.")
    ap.add_argument("--shards", type=int, default=8,
                    help="number of random shards when --mode shards")
    ap.add_argument("--force", action="store_true",
                    help="rebuild the pool from scratch; changes every document")
    ap.add_argument("--verify", action="store_true",
                    help="check existing artifacts and exit")
    args = ap.parse_args()

    ensure_dirs()

    if args.verify:
        if not (POOL_PATH.exists() and MANIFEST_PATH.exists()):
            print("artifacts missing", file=sys.stderr)
            return 1
        pool, sha = load_pool()
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        ok = sha == manifest["pool_sha256"]
        print(f"pool {len(pool)} docs, sha {sha[:16]}...  "
              f"manifest {'MATCHES' if ok else 'MISMATCH'}")
        if ok:
            game_report(manifest)
            token_report(pool, manifest)
        return 0 if ok else 1

    if POOL_PATH.exists() and not args.force:
        print(f"{POOL_PATH} exists; reusing it (--force to rebuild).\n"
              "Rebuilding changes which documents every model sees, so only do "
              "it before any runs have happened.", file=sys.stderr)
        pool, sha = load_pool()
    else:
        pool = build_pool(args.mode, args.shards)
        sha = write_pool(pool)
        print(f"wrote {POOL_PATH} ({len(pool)} docs, sha {sha[:16]}...)")

    manifest = build_manifest(pool, sha)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {MANIFEST_PATH} ({len(SIZES)} sizes x up to {MAX_RUNS} targets; "
          f"runner uses targets[:min(runs, max_usable_runs)])")

    game_report(manifest)
    token_report(pool, manifest)
    print("\nCommit both files. Every model must read the same manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())