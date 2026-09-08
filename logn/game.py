"""One game of log(N)-Questions.

Provider-agnostic: drives the question/answer loop and the final guess against
any LLMClient. Both roles run on the same client, per the experiment design.

The two roles get separate reasoning settings. They are different jobs: the
questioner partitions a large document set and benefits from as much reasoning
as you are willing to buy; the answerer makes one factual judgement about one
short document. The answerer's setting matters more than it looks, because
questioner/answerer agreement compounds as p^log2(N) -- at N=1024, p=0.90 caps
you at 35% while p=0.97 gives 74%.

This module computes no metrics beyond win/loss. Split quality, round-of-death,
and strategy classification are derived offline in logn.adjudicate, so they can
be recomputed without re-spending the API budget. The job here is to capture
everything those passes will need.

Prompts are stored by reference (doc set + history), never verbatim: they
reconstruct exactly from the manifest plus logn.prompts.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .providers.base import Completion, ProviderError, Usage

SCHEMA_VERSION = 2

# Headroom for the answerer. It emits one word, but with reasoning enabled the
# cap covers thinking too.
ANSWER_TOKENS_NO_THINKING = 64
ANSWER_TOKENS_THINKING = 4000

# Vendors disagree on the name AND the case of the truncation signal:
#   Anthropic  "max_tokens"
#   Gemini     "MAX_TOKENS"
#   OpenAI/xAI "length"
# Matching a single spelling meant the guard only ever fired for Anthropic.
# Gemini truncated 96 times across a full arm; the loose parser then rescued a
# rhetorical question out of the cut-off reasoning and played it as a real
# move. Compare case-insensitively against every known spelling.
TRUNCATED_REASONS = {"max_tokens", "maxtokens", "max_output_tokens",
                     "max-tokens", "length", "max_completion_tokens"}

# Normal completion. Anything outside these two sets is unrecognised and gets
# warned about once, because a silently-unhandled terminal reason is exactly
# how the truncation bug survived a full arm.
NORMAL_REASONS = {"stop", "end_turn", "eos", "complete", "finish_reason_stop",
                  "stop_sequence", "tool_use", ""}

_warned_reasons: set[str] = set()


def classify_stop(reason: str) -> tuple[bool, bool]:
    """Returns (truncated, recognised)."""
    r = str(reason or "").strip().lower()
    if r in TRUNCATED_REASONS:
        return True, True
    if r in NORMAL_REASONS:
        return False, True
    return False, False


# --------------------------------------------------------------------------
# result records
# --------------------------------------------------------------------------

@dataclass
class CallRecord:
    """One API call, with everything adjudication or debugging might need."""
    role: str                      # "questioner" | "answerer" | "guesser"
    round: int                     # 1-based; 0 for the final guess
    text: str = ""                 # raw response, unparsed
    reasoning_text: str = ""
    parsed: Any = None             # question str / bool / int
    parse_mode: str = ""           # strict | loose | failed
    truncated: bool = False        # hit the output cap; the reply is incomplete
    repaired: bool = False         # a reformat attempt was needed
    effort: str = ""               # reasoning level actually requested
    thinking: bool = True
    usage: dict = field(default_factory=dict)
    model_id: str = ""
    stop_reason: str = ""
    latency_s: float = 0.0
    attempts: int = 1


@dataclass
class GameResult:
    schema_version: int
    prompt_version: str
    pool_sha256: str
    model: str
    provider: str
    model_id_requested: str
    size: int
    run: int
    rounds: int
    target_id: str
    target_number: int             # 1-based position in the prompt
    doc_ids: list[str]
    history: list[dict] = field(default_factory=list)   # {question, answer}
    guess: int | None = None
    won: bool = False
    aborted: str = ""              # non-empty if the game did not finish
    calls: list[dict] = field(default_factory=list)
    usage_total: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    settings: dict = field(default_factory=dict)
    started_at: float = 0.0
    duration_s: float = 0.0
    host: str = field(default_factory=platform.node)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def result_path(results_root: Path, model: str, size: int, run: int) -> Path:
    return Path(results_root) / "raw" / model / str(size) / f"run{run:02d}.json"


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file and replace, so a crash mid-write cannot leave a
    truncated file that the runner would then skip as 'already done'."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _accepts(client, name: str) -> bool:
    """Whether this adapter's complete() takes a given keyword. Lets game.py
    stay provider-agnostic while using richer controls where they exist."""
    try:
        return name in inspect.signature(client.complete).parameters
    except (TypeError, ValueError):
        return False


def _record(role: str, rnd: int, c: Completion, parsed, mode: str,
            repaired: bool, effort: str, thinking: bool,
            truncated: bool) -> CallRecord:
    return CallRecord(
        role=role, round=rnd, text=c.text, reasoning_text=c.reasoning_text,
        parsed=parsed, parse_mode=mode, truncated=truncated, repaired=repaired,
        effort=effort or "", thinking=thinking,
        usage=asdict(c.usage), model_id=c.model_id, stop_reason=c.stop_reason,
        latency_s=round(c.latency_s, 3), attempts=c.attempts,
    )


# --------------------------------------------------------------------------
# the game
# --------------------------------------------------------------------------

def play_game(client, docs: list[dict], target_id: str, rounds: int, *,
              size: int, run: int, pool_sha: str,
              max_output_tokens: int = 60000, temperature: float = 0.0,
              max_repair_attempts: int = 1,
              log_reasoning: bool = True,
              questioner_effort: str | None = None,
              answerer_effort: str | None = None,
              answerer_thinking: bool = True,
              answerer_max_tokens: int | None = None,
              rates: dict | None = None) -> GameResult:
    """Play one complete game and return a fully populated result.

    A single reformat attempt is allowed per call. Beyond that the game is
    marked aborted rather than retried indefinitely: unlimited retries would
    turn a format-compliance failure into a hidden cost and hide a real
    difference between models.
    """
    target = next(d for d in docs if d["id"] == target_id)
    target_no = next(i for i, d in enumerate(docs, 1) if d["id"] == target_id)

    supports_thinking = _accepts(client, "thinking")
    supports_effort = _accepts(client, "effort")

    if answerer_max_tokens is None:
        answerer_max_tokens = (ANSWER_TOKENS_NO_THINKING
                               if (answerer_thinking is False and supports_thinking)
                               else ANSWER_TOKENS_THINKING)

    res = GameResult(
        schema_version=SCHEMA_VERSION,
        prompt_version=prompts.PROMPT_VERSION,
        pool_sha256=pool_sha,
        model=client.name,
        provider=client.provider,
        model_id_requested=client.model_id,
        size=size, run=run, rounds=rounds,
        target_id=target_id, target_number=target_no,
        doc_ids=[d["id"] for d in docs],
        settings={
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "max_repair_attempts": max_repair_attempts,
            "questioner_effort": questioner_effort or getattr(client, "effort", None),
            "answerer_effort": answerer_effort or getattr(client, "effort", None),
            "answerer_thinking": answerer_thinking,
            "answerer_max_tokens": answerer_max_tokens,
            "supports_effort": supports_effort,
            "supports_thinking_toggle": supports_thinking,
            "send_temperature": getattr(client, "send_temperature", None),
            "reasoning_budget": getattr(client, "reasoning_budget", None),
            "dropped_params": list(getattr(client, "dropped_params", []) or []),
        },
        started_at=time.time(),
    )
    total = Usage()
    t0 = time.perf_counter()

    def invoke(prompt, max_tokens: int, thinking: bool,
               effort: str | None) -> Completion:
        kw = {"max_tokens": max_tokens, "temperature": temperature}
        if supports_thinking:
            kw["thinking"] = thinking
        if supports_effort and effort:
            kw["effort"] = effort
        return client.complete(prompt, **kw)

    def call(prompt, max_tokens, role, rnd, parse_fn, repair_msg,
             thinking: bool = True, effort: str | None = None):
        """One call plus at most `max_repair_attempts` reformat attempts."""
        nonlocal total
        repaired = False
        for attempt in range(max_repair_attempts + 1):
            p = prompt if attempt == 0 else prompts.Prompt(
                system=prompt.system, cacheable=prompt.cacheable,
                tail=prompt.tail + "\n\n" + repair_msg)
            c = invoke(p, max_tokens, thinking, effort)
            if not log_reasoning:
                c.reasoning_text = ""
            total += c.usage

            r = parse_fn(c.text)
            truncated, recognised = classify_stop(c.stop_reason)
            if not recognised and c.stop_reason not in _warned_reasons:
                # A terminal reason nobody anticipated. Say so loudly: an
                # unhandled one is how a whole arm got corrupted once.
                _warned_reasons.add(c.stop_reason)
                print(f"WARNING: unrecognised stop_reason {c.stop_reason!r} from "
                      f"{client.name}. If it means truncation, add it to "
                      f"TRUNCATED_REASONS in logn/game.py.",
                      file=sys.stderr, flush=True)
            if truncated:
                # A cut-off reply can still satisfy the QUESTION: regex, and
                # the loose fallback will happily lift a rhetorical question
                # out of abandoned reasoning. Truncation is a failure whatever
                # the parser found.
                r = prompts.Parsed(None, "failed")

            res.calls.append(asdict(_record(
                role, rnd, c, r.value, r.mode, repaired,
                effort or getattr(client, "effort", ""), thinking, truncated)))
            if r.ok:
                return r
            repaired = True
        return r        # last failed attempt

    history: list[tuple[str, bool]] = []
    try:
        for rnd in range(1, rounds + 1):
            q = call(prompts.qbot_prompt(docs, history, rnd, rounds),
                     max_output_tokens, "questioner", rnd,
                     prompts.parse_question, prompts.REPAIR_QUESTION,
                     thinking=True, effort=questioner_effort)
            if not q.ok:
                res.aborted = _why("question", rnd, res)
                break

            a = call(prompts.abot_prompt(target, q.value),
                     answerer_max_tokens, "answerer", rnd,
                     prompts.parse_answer, prompts.REPAIR_ANSWER,
                     thinking=answerer_thinking, effort=answerer_effort)
            if not a.ok:
                res.aborted = _why("answer", rnd, res)
                break

            history.append((q.value, a.value))
            res.history.append({"round": rnd, "question": q.value,
                                "answer": bool(a.value)})
        else:
            g = call(prompts.qbot_guess_prompt(docs, history, rounds),
                     max_output_tokens, "guesser", 0,
                     lambda t: prompts.parse_guess(t, len(docs)),
                     prompts.REPAIR_GUESS,
                     thinking=True, effort=questioner_effort)
            if g.ok:
                res.guess = g.value
                res.won = g.value == target_no
            else:
                res.aborted = _why("guess", 0, res)

    except ProviderError as e:
        # Record the partial game rather than losing it. The runner can retry
        # the triple later; an aborted file is not treated as complete.
        res.aborted = f"provider error: {e}"

    res.usage_total = asdict(total)
    res.cost_usd = round(total.cost(rates or {}), 6)
    res.duration_s = round(time.perf_counter() - t0, 2)
    return res


def _why(what: str, rnd: int, res: GameResult) -> str:
    """Distinguish 'the model would not follow the format' from 'the reply was
    cut off'. They call for different fixes: one is a model property, the other
    means max_output_tokens is too low."""
    if res.calls and res.calls[-1].get("truncated"):
        return (f"{what} truncated at max_tokens (round {rnd}); "
                f"raise max_output_tokens in experiment.yaml")
    return f"unparseable {what} at round {rnd}"


# --------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------

def load_corpus(repo_root: Path):
    """Returns (pool_by_id, manifest). Verifies the pool hash matches."""
    import hashlib
    pool_path = Path(repo_root) / "data" / "pool.jsonl"
    manifest_path = Path(repo_root) / "data" / "docsets" / "manifest.json"

    h = hashlib.sha256()
    pool = {}
    with pool_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            h.update(line.encode("utf-8"))
            d = json.loads(line)
            pool[d["id"]] = d

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if h.hexdigest() != manifest["pool_sha256"]:
        raise RuntimeError(
            "pool.jsonl does not match manifest.json. The corpus changed after "
            "the manifest was built; results would not be comparable. "
            "Re-run build_corpus.py, or restore the committed pool."
        )
    return pool, manifest


def game_spec(pool: dict, manifest: dict, size: int, run: int):
    """Returns (docs_in_prompt_order, target_id, rounds)."""
    entry = manifest["sizes"][str(size)]
    usable = entry["max_usable_runs"]
    if not (0 <= run < usable):
        raise ValueError(f"N={size} has {usable} usable runs (0-{usable - 1})")
    docs = [pool[i] for i in entry["doc_ids"]]
    return docs, entry["targets"][run], entry["rounds"]