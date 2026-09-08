"""Orchestration across (model, N, run).

Resumable by construction: any triple whose result file exists and is complete
is skipped, so re-running after a rate limit, a crash, or a closed laptop costs
nothing. Aborted games are not complete and will be retried.

Ordering is deliberate. Games are grouped by (model, size) and played
consecutively, because every game at one size shares the same document prefix.
Running them back to back keeps that prefix inside the provider's cache TTL,
which is worth roughly 4x on the bill. Sizes run largest-first within a model so
that a run interrupted early still has the expensive, most informative data
points; small sizes are cheap to fill in later.

Concurrency is per (model, size) group rather than global: one worker per group
keeps each group's cache prefix hot, while different groups proceed in
parallel.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import keys, prompts
from .game import (SCHEMA_VERSION, game_spec, load_corpus, play_game,
                   result_path, write_atomic)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    provider: str
    id: str
    rates: dict = field(default_factory=dict)
    # Reasoning knobs. Which one applies depends on the provider; see
    # config/models.yaml for the mapping.
    effort: str = "high"              # Anthropic adaptive thinking, questioner
    answerer_effort: str = "medium"   # the answerer's job is smaller
    answerer_thinking: bool = True    # starving it biases answers toward "No"
    reasoning_effort: str = ""        # OpenAI
    reasoning_budget: int = 0         # legacy Anthropic, Gemini
    # Per-provider document swaps, {from_id: to_id}. Needed when a provider's
    # content filter rejects a document that the frozen corpus contains.
    # Substituting preserves N and the round budget; dropping the document
    # would change log2(N) and make the size non-comparable.
    doc_substitutions: dict = field(default_factory=dict)


def load_config(repo_root: Path) -> tuple[dict, list[ModelSpec]]:
    import yaml
    root = Path(repo_root)
    exp = yaml.safe_load((root / "config" / "experiment.yaml").read_text())
    raw = yaml.safe_load((root / "config" / "models.yaml").read_text())
    models = [
        ModelSpec(
            name=m["name"], provider=m["provider"], id=m["id"],
            rates=m.get("rates", {}),
            effort=m.get("effort", "high") or "high",
            answerer_effort=m.get("answerer_effort", "medium") or "medium",
            answerer_thinking=m.get("answerer_thinking", True),
            reasoning_effort=m.get("reasoning_effort", "") or "",
            reasoning_budget=m.get("reasoning_budget", 0) or 0,
            doc_substitutions=dict(m.get("doc_substitutions", {}) or {}),
        )
        for m in raw["models"]
    ]
    return exp, models


def build_client(spec: ModelSpec):
    """Import adapters lazily so a missing SDK for one provider does not stop
    a run that does not use it."""
    if spec.provider == "anthropic":
        from .providers.anthropic import AnthropicClient
        return AnthropicClient(spec.name, spec.id, effort=spec.effort,
                               reasoning_budget=spec.reasoning_budget)
    if spec.provider == "gemini":
        from .providers.gemini import GeminiClient
        return GeminiClient(spec.name, spec.id, effort=spec.effort,
                            reasoning_budget=spec.reasoning_budget)
    if spec.provider in ("openai", "xai", "zai", "moonshot", "deepseek"):
        from .providers import openai_compat as oc
        cls = {"openai": oc.OpenAIClient, "xai": oc.XAIClient,
               "zai": oc.ZAIClient, "moonshot": oc.MoonshotClient, "deepseek": oc.DeepSeekClient}[spec.provider]
        return cls(spec.name, spec.id, effort=spec.effort,
                   reasoning_effort=spec.reasoning_effort)

    raise NotImplementedError(
        f"no adapter for provider {spec.provider!r} yet "
        f"(model {spec.name}). Implement logn/providers/{spec.provider}.py."
    )


# --------------------------------------------------------------------------
# work planning
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    model: str
    size: int
    run: int


def game_state(path: Path) -> str:
    """One of: missing, unreadable, aborted, stale, done.

    "stale" means the game was played under a different prompt or schema
    version. Those must be replayed: a results tree that silently mixes
    conditions is worse than one with gaps, because the mixture is invisible
    at analysis time.
    """
    if not path.exists():
        return "missing"
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "unreadable"
    if d.get("aborted"):
        return "aborted"
    if d.get("prompt_version") != prompts.PROMPT_VERSION:
        return "stale"
    if d.get("schema_version") != SCHEMA_VERSION:
        return "stale"
    return "done"


def is_complete(path: Path) -> bool:
    return game_state(path) == "done"


def plan(manifest: dict, models: list[ModelSpec], runs: int,
         results_root: Path, sizes: list[int] | None = None,
         only_models: list[str] | None = None) -> tuple[list[Task], int]:
    """Returns (todo, already_done). Largest sizes first.

    Also records a breakdown on plan.last_states so the caller can report why
    games are being replayed.
    """
    all_sizes = sorted((int(k) for k in manifest["sizes"]), reverse=True)
    use_sizes = [s for s in all_sizes if sizes is None or s in sizes]

    todo, done = [], 0
    states: Counter[str] = Counter()
    for m in models:
        if only_models and m.name not in only_models:
            continue
        for size in use_sizes:
            usable = manifest["sizes"][str(size)]["max_usable_runs"]
            for run in range(min(runs, usable)):
                st = game_state(result_path(results_root, m.name, size, run))
                states[st] += 1
                if st == "done":
                    done += 1
                else:
                    todo.append(Task(m.name, size, run))
    plan.last_states = states
    return todo, done


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

class Ledger:
    """Thread-safe running totals, for progress and the budget guard."""

    def __init__(self, budget: float = 0.0):
        self._lock = threading.Lock()
        self._print_lock = threading.Lock()
        self.cost = 0.0
        self.games = 0
        self.wins = 0
        self.aborted = 0
        self.truncated = 0
        self.budget = budget
        self.stop = threading.Event()

    def add(self, res) -> None:
        with self._lock:
            self.cost += res.cost_usd
            self.games += 1
            self.wins += int(res.won)
            self.aborted += int(bool(res.aborted))
            self.truncated += sum(
                1 for c in res.calls if c.get("stop_reason") == "max_tokens")
            if self.budget and self.cost >= self.budget:
                self.stop.set()

    def line(self) -> str:
        with self._lock:
            return (f"{self.games} games  {self.wins} won  "
                    f"{self.aborted} aborted  ${self.cost:.2f}")

    def report(self, text: str) -> None:
        """Print one whole line at a time. Without this, concurrent groups
        interleave mid-line and the log becomes unreadable."""
        with self._print_lock:
            print(text, flush=True)


def run_group(spec: ModelSpec, size: int, runs: list[int], *, pool, manifest,
              exp: dict, results_root: Path, ledger: Ledger) -> None:
    """All runs for one (model, size). Sequential, to keep the cache prefix hot."""
    client = None
    try:
        for run in sorted(runs):
            if ledger.stop.is_set():
                return
            if client is None:
                client = build_client(spec)

            docs, target_id, rounds = game_spec(pool, manifest, size, run)
            subs = {}
            if spec.doc_substitutions:
                swapped = []
                for d in docs:
                    to = spec.doc_substitutions.get(d["id"])
                    if to and to in pool:
                        subs[d["id"]] = to
                        swapped.append(pool[to])
                    else:
                        swapped.append(d)
                docs = swapped
                if target_id in subs:
                    raise RuntimeError(
                        f"{spec.name}: cannot substitute target {target_id}; "
                        "that would change the game rather than the set")

            res = play_game(
                client, docs, target_id, rounds,
                size=size, run=run, pool_sha=manifest["pool_sha256"],
                max_output_tokens=exp.get("max_output_tokens", 16000),
                temperature=exp.get("temperature", 0.0),
                max_repair_attempts=exp.get("max_repair_attempts", 1),
                log_reasoning=exp.get("log_reasoning_traces", True),
                questioner_effort=spec.effort,
                answerer_effort=spec.answerer_effort,
                answerer_thinking=spec.answerer_thinking,
                answerer_max_tokens=exp.get("answerer_max_tokens"),
                rates=spec.rates,
            )
            # Record any substitution in the result file, so a reader can see
            # that this arm did not run on the identical frozen set.
            res.settings["doc_substitutions"] = subs
            write_atomic(result_path(results_root, spec.name, size, run),
                         res.to_json())
            ledger.add(res)

            flag = "WIN " if res.won else ("ABORT" if res.aborted else "loss")
            cr = res.usage_total.get("cache_read_tokens", 0)
            cut = sum(1 for c in res.calls if c.get("stop_reason") == "max_tokens")
            note = f"  TRUNCATED x{cut}" if cut else ""
            ledger.report(f"  {spec.name} N={size:<5} run={run:<2} {flag} "
                          f"${res.cost_usd:.3f} cache_read={cr:,} "
                          f"{res.duration_s:.0f}s   [{ledger.line()}]{note}")
    finally:
        if client is not None and hasattr(client, "close"):
            client.close()


def execute(repo_root: Path, *, runs: int | None = None,
            sizes: list[int] | None = None, only_models: list[str] | None = None,
            concurrency: int | None = None, budget: float = 0.0,
            dry_run: bool = False) -> int:
    root = Path(repo_root)
    exp, models = load_config(root)
    pool, manifest = load_corpus(root)          # verifies the pool hash
    results_root = root / "results"
    runs = runs if runs is not None else exp.get("runs", 8)
    conc = concurrency if concurrency is not None else exp.get("concurrency", 4)

    selected = [m for m in models if not only_models or m.name in only_models]
    if not selected:
        known = ", ".join(m.name for m in models)
        print(f"no models selected; known: {known}", file=sys.stderr)
        return 1

    # Fail before a long run, not forty minutes into it.
    if not dry_run:
        keys.check_keys({m.provider for m in selected})
        for m in selected:
            try:
                c = build_client(m)
            except NotImplementedError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            if hasattr(c, "close"):
                c.close()

    todo, done = plan(manifest, selected, runs, results_root, sizes, only_models)
    st = getattr(plan, "last_states", {})
    extra = "  ".join(f"{v} {k}" for k, v in sorted(st.items())
                      if k not in ("done", "missing") and v)
    print(f"{len(todo)} games to play, {done} already complete, "
          f"prompt cache grouped by (model, size)")
    if extra:
        print(f"  will be replayed: {extra}")
    if st.get("stale"):
        print(f"  (stale = played under a different prompt_version; "
              f"current is {prompts.PROMPT_VERSION})")
    if budget:
        print(f"budget guard: stop after ${budget:.2f}")
    if not todo:
        return 0

    # Group by (model, size); each group is one sequential unit of work.
    groups: dict[tuple[str, int], list[int]] = {}
    for t in todo:
        groups.setdefault((t.model, t.size), []).append(t.run)
    by_name = {m.name: m for m in selected}

    if dry_run:
        # Synchronous, so the plan reads in order instead of interleaving.
        for (mname, size) in sorted(groups, key=lambda k: (k[0], -k[1])):
            rl = sorted(groups[(mname, size)])
            spec = by_name[mname]
            print(f"  {mname:<16} N={size:<5} runs {rl[0]}-{rl[-1]} "
                  f"({len(rl)} games)")
        return 0

    ledger = Ledger(budget)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futures = {
            ex.submit(run_group, by_name[mname], size, rl, pool=pool,
                      manifest=manifest, exp=exp, results_root=results_root,
                      ledger=ledger): (mname, size)
            for (mname, size), rl in groups.items()
        }
        for fut in as_completed(futures):
            mname, size = futures[fut]
            try:
                fut.result()
            except Exception as e:                       # noqa: BLE001
                print(f"  group {mname} N={size} failed: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)

    print(f"\n{ledger.line()}  in {(time.perf_counter() - t0) / 60:.1f} min")
    if ledger.stop.is_set():
        print("stopped early: budget reached. Re-run to continue.")
    if ledger.aborted:
        print(f"{ledger.aborted} aborted games will be retried on the next run.")
    if ledger.truncated:
        print(f"WARNING: {ledger.truncated} calls hit max_tokens and were cut "
              f"off mid-reply. Raise max_output_tokens in experiment.yaml; "
              f"with adaptive thinking the cap covers reasoning too.")
    return 0