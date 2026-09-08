#!/usr/bin/env python3
"""Run games.

    python scripts/run.py --dry-run                      # show the plan
    python scripts/run.py --models claude-opus-5 --sizes 8 64
    python scripts/run.py --budget 25                    # stop at $25
    python scripts/run.py                                # everything in config

Safe to interrupt and re-run: completed games are skipped, aborted ones retried.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from logn import runner                                  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="model names from config/models.yaml; default all")
    ap.add_argument("--sizes", nargs="*", type=int, default=None,
                    help="document set sizes; default all")
    ap.add_argument("--runs", type=int, default=None,
                    help="runs per size; default experiment.yaml")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="parallel (model, size) groups; default experiment.yaml")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="stop once cumulative cost reaches this many USD")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be played, call nothing")
    args = ap.parse_args()

    return runner.execute(
        ROOT, runs=args.runs, sizes=args.sizes, only_models=args.models,
        concurrency=args.concurrency, budget=args.budget, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())