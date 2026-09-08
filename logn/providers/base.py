"""Provider-agnostic client contract.

Adapters translate a logn.prompts.Prompt into one vendor's request format and
normalise the response into a Completion. They handle transport, caching
mechanics, and usage accounting -- never prompt wording, which lives in
logn.prompts so every model sees identical text.

The cache boundary is the load-bearing part of this interface. Prompt.cacheable
holds the document block, which is byte-identical across every round of a game.
Adapters must mark it as cacheable in whatever way their vendor supports.
Getting this wrong costs roughly 4x and produces no other symptom, so
Usage.cache_read_tokens is worth watching on the first few calls.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class Usage:
    """Normalised token accounting.

    Vendors disagree on whether cached reads are included in the input count.
    Adapters must normalise so that these four fields are disjoint:
    input_tokens excludes cache_read_tokens and cache_write_tokens.
    """
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0      # 0 where the vendor does not report it

    def cost(self, rates: dict) -> float:
        """USD, given per-million rates from config/models.yaml."""
        return (
            self.input_tokens * rates.get("input", 0.0)
            + self.cache_read_tokens * rates.get("cache_read", 0.0)
            + self.cache_write_tokens * rates.get("cache_write", rates.get("input", 0.0))
            + self.output_tokens * rates.get("output", 0.0)
        ) / 1e6

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(
            (self.input_tokens, self.cache_read_tokens, self.cache_write_tokens,
             self.output_tokens, self.reasoning_tokens),
            (other.input_tokens, other.cache_read_tokens, other.cache_write_tokens,
             other.output_tokens, other.reasoning_tokens))))


@dataclass
class Completion:
    text: str
    usage: Usage = field(default_factory=Usage)
    model_id: str = ""             # as echoed by the API, not as requested
    stop_reason: str = ""
    latency_s: float = 0.0
    attempts: int = 1              # >1 means transient failures were retried
    reasoning_text: str = ""       # kept separate; only if the vendor exposes it

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = asdict(self.usage)
        return d


class ProviderError(RuntimeError):
    """Non-retryable. The request was rejected and will be rejected again."""


class TransientError(RuntimeError):
    """Retryable: rate limit, overload, timeout, 5xx."""


# --------------------------------------------------------------------------
# protocol
# --------------------------------------------------------------------------

class LLMClient(Protocol):
    name: str          # config name, e.g. "claude-opus-5"
    provider: str      # key in logn.keys.ENV_VAR
    model_id: str      # exact API model string

    def complete(self, prompt, *, max_tokens: int, temperature: float) -> Completion:
        """Issue one request. Raises ProviderError on permanent failure."""
        ...


# --------------------------------------------------------------------------
# shared retry
# --------------------------------------------------------------------------

def with_retries(fn, *, max_attempts: int = 5, base_delay: float = 2.0,
                 max_delay: float = 60.0) -> Completion:
    """Exponential backoff with jitter on TransientError.

    Jitter matters here: the runner fans out concurrent requests, and
    unjittered backoff makes them retry in lockstep and re-trigger the same
    rate limit.
    """
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            c = fn()
            c.attempts = attempt
            return c
        except TransientError as e:
            last = e
            if attempt == max_attempts:
                break
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            time.sleep(delay * (0.5 + random.random()))
    raise ProviderError(f"giving up after {max_attempts} attempts: {last}") from last


def is_transient_status(status: int) -> bool:
    return status == 429 or status == 408 or status >= 500


def redact(headers: dict) -> dict:
    """Never let a key reach a log or a result file."""
    out = {}
    for k, v in headers.items():
        out[k] = "<redacted>" if any(
            s in k.lower() for s in ("key", "auth", "token")) else v
    return out


# --------------------------------------------------------------------------
# test double
# --------------------------------------------------------------------------

class ScriptedClient:
    """Returns canned replies. For testing game logic without spending money."""

    name = "scripted"
    provider = "none"
    model_id = "scripted"

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[Any] = []

    def complete(self, prompt, *, max_tokens: int, temperature: float) -> Completion:
        self.calls.append(prompt)
        if not self._replies:
            raise ProviderError("ScriptedClient ran out of replies")
        text = self._replies.pop(0)
        return Completion(text=text, model_id="scripted",
                          usage=Usage(input_tokens=len(prompt.as_text()) // 4,
                                      output_tokens=len(text) // 4))