#!/usr/bin/env python3
"""
Scaffold the logn-questions project layout.

Run once from the repo root:

    python setup_project.py
    python setup_project.py --dry-run     # show what would happen, write nothing

Idempotent and non-destructive: every file is written only if absent. Existing
work -- build_corpus.py, inspect_pool.py, pool.jsonl, manifest.json, anything
under results/ -- is never touched. Re-run it any time; it will only fill gaps.

Afterwards:

    pip install -e .

which puts `logn` on the import path so scripts/ can `import logn.prompts`
from any working directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Models the results tree is pre-created for. Kept in sync with config/models.yaml
# by hand; a stale entry only leaves an empty directory, which is harmless.
MODELS = [
    "claude-opus-5",
    "gpt-5.6-sol",
    "gemini-3.8-flash",
    "grok-4.6",
    "glm-5.2",
    "kimi-k3",
]
SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

DIRS = [
    "logn",
    "logn/providers",
    "scripts",
    "config",
    "data",
    "data/raw",
    "data/docsets",
    "results",
    "results/raw",
    "results/adjudicated",
    "results/tables",
    "tests",
]
DIRS += [f"results/raw/{m}/{n}" for m in MODELS for n in SIZES]


# --------------------------------------------------------------------------
# file contents
# --------------------------------------------------------------------------

PYPROJECT = """\
[project]
name = "logn"
version = "0.1.0"
description = "log(N)-Questions over Wikipedia abstracts, across frontier models"
requires-python = ">=3.10"
dependencies = [
    "datasets",
    "huggingface_hub",
    "tiktoken",
    "pyyaml",
    "httpx",
]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["logn*"]
"""

GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/
.pytest_cache/
.env
.env.*
!.env.example

# Large regenerable inputs. pool.jsonl and manifest.json are NOT ignored:
# they are the frozen artifacts that make the experiment reproducible.
data/raw/*
!data/raw/.gitignore
"""

ENV_EXAMPLE = """\
# Copy to .env and fill in. .env is gitignored.
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
XAI_API_KEY=
ZAI_API_KEY=
MOONSHOT_API_KEY=
"""

EXPERIMENT_YAML = """\
# Experiment-level settings. The manifest holds up to 64 targets per size;
# `runs` selects how many of them to actually play. Raising it later re-uses
# every completed game, because targets[:8] is a prefix of targets[:16].
runs: 8

# Rounds are log2(N) by definition. A multiplier > 1 gives the questioner a
# larger budget, as a deliberate manipulation rather than the default.
round_budget_multiplier: 1

# Cap on model output. Note this INCLUDES reasoning tokens on most APIs, so a
# small value with reasoning enabled can return an empty completion. Set
# reasoning_budget separately per model in models.yaml.
max_output_tokens: 4000

temperature: 0.0

# One reformat attempt on an unparseable reply, then the game is scored lost.
# Unlimited retries would hide format-compliance differences between models.
max_repair_attempts: 1

# Store prompts by reference (doc set id + history) rather than verbatim.
# Verbatim logging costs ~620 MB across the sweep and is fully reconstructible
# from the manifest plus prompts.py.
log_prompts_verbatim: false
log_reasoning_traces: true

concurrency: 4
"""

MODELS_YAML = """\
# One frontier model per provider. `id` is the exact API model string; record
# whatever the API echoes back in the result file too, since aliases move.
#
# Rates are USD per million tokens, for cost reporting only.

models:
  - name: claude-opus-5
    provider: anthropic
    id: claude-opus-5
    rates: {input: 5.00, cache_read: 0.50, cache_write: 6.25, output: 25.00}
    reasoning_budget: 2000

  - name: gpt-5.6-sol
    provider: openai
    id: gpt-5.6-sol
    rates: {input: 5.00, cache_read: 0.50, cache_write: 5.00, output: 30.00}
    reasoning_effort: medium

  - name: gemini-3.1-pro
    provider: gemini
    id: gemini-3.1-pro
    rates: {input: 2.00, cache_read: 0.20, cache_write: 2.00, output: 12.00}
    reasoning_budget: 2000

  - name: grok-4.6
    provider: xai
    id: grok-4.6
    rates: {input: 2.00, cache_read: 0.50, cache_write: 2.00, output: 6.00}

  - name: glm-5.2
    provider: zai
    id: glm-5.2
    rates: {input: 1.40, cache_read: 0.26, cache_write: 1.40, output: 4.40}

  - name: kimi-k3
    provider: moonshot
    id: kimi-k3
    rates: {input: 3.00, cache_read: 0.30, cache_write: 3.00, output: 15.00}
"""

INIT_PY = '''\
"""log(N)-Questions over sentences, played by frontier models."""

__version__ = "0.1.0"
'''

PROVIDERS_INIT = '''\
"""Provider adapters.

Each adapter normalises one vendor's API to the LLMClient protocol in base.py.
Adapters handle transport and usage accounting only -- never prompt wording,
which lives in logn.prompts so that all models see identical text.
"""
'''

STUBS = {
    "logn/providers/base.py": '''\
"""Provider-agnostic client protocol.

Implementations translate a logn.prompts.Prompt into one vendor's request
format, place the cache breakpoint at the end of Prompt.cacheable, and
normalise the response into Completion.

Not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    model_id: str = ""          # as echoed by the API, not as requested
    latency_s: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMClient(Protocol):
    name: str

    def complete(self, prompt, max_tokens: int, temperature: float) -> Completion:
        ...
''',
    "logn/game.py": '''\
"""One game of log(N)-Questions, provider-agnostic.

Drives the question/answer loop and the final guess against any LLMClient.
Contains no metric computation: scoring and partition adjudication happen
offline in logn.adjudicate so they can be redone without re-spending budget.

Not yet implemented.
"""
''',
    "logn/runner.py": '''\
"""Orchestration across (model, N, run).

Resumable by construction: skips any triple whose result file already exists,
so re-running after a rate-limit or a crash costs nothing.

Not yet implemented.
"""
''',
    "logn/adjudicate.py": '''\
"""Offline scoring.

For each logged question, computes the true partition it induces over that
round's viable set, then derives split quality and round-of-death -- the round
at which the target left the questioner's viable set. Final accuracy alone
compresses that into one bit and discards the diagnosis.

Not yet implemented.
"""
''',
    "logn/analyze.py": '''\
"""Tables and plots from adjudicated results.

Not yet implemented.
"""
''',
    "scripts/run.py": '''\
#!/usr/bin/env python3
"""CLI entry point for running games. Thin wrapper over logn.runner.

Not yet implemented.
"""
''',
    "scripts/adjudicate.py": '''\
#!/usr/bin/env python3
"""CLI entry point for offline scoring. Thin wrapper over logn.adjudicate.

Not yet implemented.
"""
''',
    "scripts/analyze.py": '''\
#!/usr/bin/env python3
"""CLI entry point for tables and plots. Thin wrapper over logn.analyze.

Not yet implemented.
"""
''',
    "tests/test_prompts.py": '''\
"""Parser contract tests. Run with: pytest"""

import pytest

logn_prompts = pytest.importorskip("logn.prompts")


def test_strict_question():
    r = logn_prompts.parse_question("reasoning\\nQUESTION: Is it music?")
    assert r.mode == "strict" and r.value == "Is it music?"


def test_loose_question():
    r = logn_prompts.parse_question("I think:\\nIs it music?")
    assert r.mode == "loose"


def test_failed_question():
    assert logn_prompts.parse_question("no idea").mode == "failed"


def test_guess_out_of_range_is_not_clamped():
    assert logn_prompts.parse_guess("GUESS: 99", 4).mode == "failed"


def test_answers():
    assert logn_prompts.parse_answer("Yes").value is True
    assert logn_prompts.parse_answer("no.").value is False
    assert logn_prompts.parse_answer("unclear").mode == "failed"


def test_cache_prefix_is_round_invariant():
    docs = [{"title": f"D{i}", "text": "body"} for i in range(4)]
    a = logn_prompts.qbot_prompt(docs, [], 1, 2)
    b = logn_prompts.qbot_prompt(docs, [("q?", True)], 2, 2)
    assert a.cacheable == b.cacheable      # or caching silently stops working
''',
}

README = """\
# log(N)-Questions across frontier models

A questioner sees N Wikipedia lead paragraphs and must identify a secretly
chosen target using log2(N) yes/no questions. An answerer sees only the target
and the question, and replies Yes or No. Both roles run on the same provider.
After Peter Potash and Kaheer Suleman, *Playing log(N)-Questions over
Sentences* (arXiv:1908.04660), scaled from 4 documents to 1024 and from trained
agents to frontier models.

## Layout

    logn/            importable package: prompts, providers, game, scoring
    scripts/         CLI entry points
    config/          experiment.yaml, models.yaml
    data/            pool.jsonl and docsets/manifest.json -- frozen, committed
    results/raw/     one JSON per game, append-only, resumable

## Setup

    pip install -e .
    copy .env.example .env        # then fill in keys

## Build the corpus (once)

    python scripts/build_corpus.py --mode shards --shards 8
    python scripts/build_corpus.py --verify
    python scripts/inspect_pool.py --short 15

`data/pool.jsonl` and `data/docsets/manifest.json` are the frozen artifacts that
make the experiment reproducible. Commit both. Never re-run with `--force`
after any API call has been made: it resamples the pool and changes every
document in the experiment.

## Design notes

**Nested doc sets.** docset(512) is a strict subset of docset(1024), so an
accuracy change between sizes is attributable to N rather than to one set being
easier.

**Prefix-extensible targets.** The manifest holds 64 targets per size in
bit-reversed order, so targets[:8] is a prefix of targets[:16] and both are
evenly stratified. Raising `runs` re-uses every completed game.

**No target is reused.** A size with N < runs simply plays fewer games: N=4
plays 4.

**Cache-aligned prompts.** Every prompt splits into system / cacheable / tail,
with documents in the cacheable part. The document block is byte-identical
across all rounds of a game. Getting this ordering wrong costs roughly 4x.

**The answerer never sees document numbers.** Otherwise the game collapses into
integer bisection. It does see titles, which leaves the enumeration and
lexical-bisection strategies genuinely available -- observing whether models
climb that ladder as N grows is part of the point.

**Scoring is offline.** Adjudication computes the true partition per question
and the round at which the target left the viable set, so metrics can be redone
without re-spending the budget.
"""


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be created, write nothing")
    args = ap.parse_args()

    files: dict[str, str] = {
        "pyproject.toml": PYPROJECT,
        ".gitignore": GITIGNORE,
        ".env.example": ENV_EXAMPLE,
        "README.md": README,
        "config/experiment.yaml": EXPERIMENT_YAML,
        "config/models.yaml": MODELS_YAML,
        "logn/__init__.py": INIT_PY,
        "logn/providers/__init__.py": PROVIDERS_INIT,
        "data/raw/.gitignore": "*\n!.gitignore\n",
        **STUBS,
    }

    made_dirs, made_files, skipped = [], [], []

    for rel in DIRS:
        p = ROOT / rel
        if not p.exists():
            made_dirs.append(rel)
            if not args.dry_run:
                p.mkdir(parents=True, exist_ok=True)

    for rel, body in files.items():
        p = ROOT / rel
        if p.exists():
            skipped.append(rel)
            continue
        made_files.append(rel)
        if not args.dry_run:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")

    tag = "would create" if args.dry_run else "created"
    print(f"{tag} {len(made_dirs)} directories")
    print(f"{tag} {len(made_files)} files:")
    for rel in sorted(made_files):
        print(f"    {rel}")
    if skipped:
        print(f"\nleft alone ({len(skipped)} already present):")
        for rel in sorted(skipped):
            print(f"    {rel}")

    if args.dry_run:
        return 0

    print("\nNext:")
    print("    pip install -e .")
    print("    copy .env.example .env")
    print("  then drop prompts.py into logn/ and run: pytest")

    missing = [f for f in ("scripts/build_corpus.py", "data/pool.jsonl",
                           "data/docsets/manifest.json") if not (ROOT / f).exists()]
    if missing:
        print("\nnote: expected existing artifacts not found: " + ", ".join(missing))
        print("      run this from the repo root, not from scripts/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())