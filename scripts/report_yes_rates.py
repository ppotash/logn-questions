#!/usr/bin/env python3
"""
Analyze how model behavior changes after the first factual disagreement.
This script compares 'Yes' rates when the model is accurate vs. after it errs.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn.game import load_corpus

RESULTS = ROOT / "results" / "raw"
CACHE_DIR = ROOT / "results" / "adjudicated"
DEFAULT_CACHE = CACHE_DIR / "judgments.json"

def key(question: str, doc_id: str) -> str:
    """Unique key for a (question, document) pair."""
    return hashlib.sha256(f"{doc_id}\x00{question}".encode()).hexdigest()[:20]

def load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def games():
    """Generator for all valid game JSONs."""
    for f in sorted(glob.glob(str(RESULTS / "*" / "*" / "*.json"))):
        try:
            with open(f, encoding="utf-8") as j:
                g = json.load(j)
            if not g.get("aborted"):
                yield g
        except (json.JSONDecodeError, OSError):
            continue

def get_error_round(g, cache):
    """
    Returns the round number of the first disagreement between 
    the answerer and the judge regarding the target document.
    Returns None if they always agree.
    """
    tid = g["target_id"]
    for h in g["history"]:
        k = key(h["question"], tid)
        # If we haven't adjudicated this pair, we can't determine the error round
        if k not in cache:
            return "unjudged"
        if cache[k] != h["answer"]:
            return h["round"]
    return None

def report_yes_rates(cache):
    """
    Calculates the percentage of 'Yes' answers before and after 
    the model makes its first factual error.
    """
    # stats[model] = {before_yes, before_total, after_yes, after_total}
    stats = defaultdict(lambda: Counter())
    
    for g in games():
        err_round = get_error_round(g, cache)
        
        # Skip games that aren't fully adjudicated or had zero errors
        if err_round == "unjudged" or err_round is None:
            continue
            
        m = g["model"]
        for h in g["history"]:
            is_yes = 1 if h["answer"] else 0
            
            if h["round"] < err_round:
                # Phases where the answerer and judge agreed
                stats[m]["before_yes"] += is_yes
                stats[m]["before_total"] += 1
            else:
                # Phases from the first error until the end of the game
                stats[m]["after_yes"] += is_yes
                stats[m]["after_total"] += 1

    print("\nYES-RATE ANALYSIS: BEHAVIORAL SHIFT AFTER FIRST ERROR")
    print("Definitions:")
    print("  Before: Rounds where the model and judge agreed on the target ('Alive' state)")
    print("  After:  Rounds from the first disagreement until game end ('Hallucinated' state)")
    print("  n=...:  Total number of rounds (questions) analyzed in this state")
    
    header = f"\n{'Model':<18} {'Before Error':>18} {'After Error':>18} {'Change':>10}"
    print(header)
    print("-" * len(header))
    
    pooled = Counter()
    
    for m in sorted(stats.keys()):
        s = stats[m]
        pooled.update(s)
        
        b_rate = s["before_yes"] / s["before_total"] if s["before_total"] else 0
        a_rate = s["after_yes"] / s["after_total"] if s["after_total"] else 0
        delta = a_rate - b_rate
        
        print(f"{m[:17]:<18} "
              f"{b_rate:>7.1%} (n={s['before_total']:<3}) "
              f"{a_rate:>7.1%} (n={s['after_total']:<3}) "
              f"{delta:>+9.1%}")
              
    # Calculate aggregate across all models
    if pooled["before_total"] > 0:
        pb_rate = pooled["before_yes"] / pooled["before_total"]
        pa_rate = pooled["after_yes"] / pooled["after_total"]
        print("-" * len(header))
        print(f"{'POOLED TOTAL':<18} "
              f"{pb_rate:>7.1%} (n={pooled['before_total']:<3}) "
              f"{pa_rate:>7.1%} (n={pooled['after_total']:<3}) "
              f"{pa_rate - pb_rate:>+9.1%}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", help="Filename of the judge's cache")
    args = parser.parse_args()

    cache_path = Path(args.cache) if args.cache else DEFAULT_CACHE
    if not cache_path.exists() and "/" not in str(cache_path):
        cache_path = CACHE_DIR / cache_path

    if not cache_path.exists():
        print(f"Error: Cache file not found at {cache_path}")
        return 1

    cache = load_cache(cache_path)
    report_yes_rates(cache)
    return 0

if __name__ == "__main__":
    sys.exit(main())