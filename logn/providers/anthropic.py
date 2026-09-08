"""Anthropic adapter.

Caching is explicit here: a `cache_control` marker on the last content block of
the prefix tells the API where the reusable segment ends. We put it at the end
of Prompt.cacheable, which holds the document block.

Thinking has two incompatible contracts across the model line:

  adaptive  Opus 5, Sonnet 5, Fable 5 and later. thinking={"type":"adaptive"}
            plus output_config={"effort": ...}. Token budgets are ignored.
            Reasoning is ON BY DEFAULT: a request that configures nothing still
            thinks. Disabling requires effort <= "high".
  budget    Opus 4.1 and earlier. thinking={"type":"enabled","budget_tokens":N}.

Rather than hardcode which model uses which, the adapter starts in `adaptive`
and switches on the specific 400 the API returns. Same for `temperature`, which
newer models reject outright.

`complete()` accepts a per-call `effort`, so the questioner and the answerer can
run at different levels within one game. They are different jobs: the questioner
partitions a large set, the answerer makes one factual judgement about one short
document.

Two things that surprise people:

  - The minimum cacheable prefix is 1024 tokens (2048 for Haiku). A pilot at
    N=4 is ~600 tokens and will report zero cache hits. That is correct.
  - max_tokens caps thinking AND visible output together. Sized too tightly, a
    reply truncates mid-answer; stop_reason will say "max_tokens".
"""

from __future__ import annotations

import time

import httpx

from ..keys import get_key
from .base import (Completion, ProviderError, TransientError, Usage,
                   is_transient_status, with_retries)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"

EFFORTS = ("low", "medium", "high", "xhigh", "max")
# Anthropic rejects disabled thinking above "high".
MAX_EFFORT_WHEN_DISABLED = "high"


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, name: str, model_id: str, *, effort: str = "medium",
                 reasoning_budget: int = 0, thinking_api: str = "adaptive",
                 timeout: float = 900.0):
        """effort applies to the adaptive API; reasoning_budget to the legacy one."""
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        self.name = name
        self.model_id = model_id
        self.effort = effort                      # default when a call says nothing
        self.reasoning_budget = reasoning_budget
        self.thinking_api = thinking_api          # "adaptive" | "budget"
        # Newer models reject `temperature` outright. Discovered from the first
        # 400 rather than hardcoded per model.
        self.send_temperature = True
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "x-api-key": get_key("anthropic"),
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
        )

    # ----------------------------------------------------------------
    def _body(self, prompt, max_tokens: int, temperature: float,
              thinking: bool, effort: str | None) -> dict:
        eff = effort or self.effort
        if eff not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {eff!r}")

        # The document block is its own content block with a cache breakpoint;
        # the tail follows uncached. A stable prefix is the whole point.
        content = []
        if prompt.cacheable:
            content.append({
                "type": "text",
                "text": prompt.cacheable,
                "cache_control": {"type": "ephemeral"},
            })
        content.append({"type": "text", "text": prompt.tail})

        body = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": content}],
        }

        if self.thinking_api == "adaptive":
            if thinking:
                body["thinking"] = {"type": "adaptive"}
            else:
                body["thinking"] = {"type": "disabled"}
                # Disabled thinking is only permitted up to "high".
                if EFFORTS.index(eff) > EFFORTS.index(MAX_EFFORT_WHEN_DISABLED):
                    eff = MAX_EFFORT_WHEN_DISABLED
                if self.send_temperature:
                    body["temperature"] = temperature
            body["output_config"] = {"effort": eff}
        else:
            if thinking and self.reasoning_budget:
                if max_tokens <= self.reasoning_budget:
                    raise ProviderError(
                        f"max_tokens ({max_tokens}) must exceed reasoning_budget "
                        f"({self.reasoning_budget}); the reply would be empty"
                    )
                body["thinking"] = {"type": "enabled",
                                    "budget_tokens": self.reasoning_budget}
            elif self.send_temperature:
                body["temperature"] = temperature
        return body

    # ----------------------------------------------------------------
    def _once(self, body: dict) -> Completion:
        t0 = time.perf_counter()
        try:
            r = self._client.post(API_URL, json=body)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport: {e}") from e
        elapsed = time.perf_counter() - t0

        if r.status_code != 200:
            detail = r.text[:500]
            if is_transient_status(r.status_code):
                raise TransientError(f"{r.status_code}: {detail}")
            raise ProviderError(f"{r.status_code}: {detail}")

        data = r.json()
        # content is an array of typed blocks. With thinking on, a thinking
        # block precedes the text block; reading content[0] is a common bug.
        text_parts, think_parts = [], []
        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "thinking":
                think_parts.append(block.get("thinking", ""))

        u = data.get("usage", {})
        thinking_text = "\n".join(think_parts)
        return Completion(
            text="\n".join(text_parts).strip(),
            reasoning_text=thinking_text,
            usage=Usage(
                input_tokens=u.get("input_tokens", 0),
                cache_read_tokens=u.get("cache_read_input_tokens", 0),
                cache_write_tokens=u.get("cache_creation_input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                # Not broken out by the API; approximated from the blocks so
                # the figure is comparable across providers.
                reasoning_tokens=len(thinking_text) // 4 if thinking_text else 0,
            ),
            model_id=data.get("model", self.model_id),
            stop_reason=data.get("stop_reason", ""),
            latency_s=elapsed,
        )

    # ----------------------------------------------------------------
    def _adapt(self, msg: str) -> bool:
        """Reconfigure in response to a contract mismatch. True if retrying
        is worth it. Each branch fires at most once per session."""
        if "thinking.type.enabled" in msg and self.thinking_api != "adaptive":
            self.thinking_api = "adaptive"
            return True
        if ("thinking.type.adaptive" in msg or "output_config" in msg) \
                and self.thinking_api != "budget":
            self.thinking_api = "budget"
            return True
        if "thinking.type.disabled" in msg:
            # This model cannot switch thinking off. Leave it on.
            self.thinking_api = "adaptive"
            self._no_disable = True
            return True
        if "temperature" in msg and self.send_temperature:
            self.send_temperature = False
            return True
        return False

    def complete(self, prompt, *, max_tokens: int, temperature: float,
                 thinking: bool = True, effort: str | None = None) -> Completion:
        if thinking is False and getattr(self, "_no_disable", False):
            thinking = True
        for _ in range(3):        # at most a couple of contract corrections
            try:
                return with_retries(lambda: self._once(
                    self._body(prompt, max_tokens, temperature, thinking, effort)))
            except ProviderError as e:
                if not self._adapt(str(e)):
                    raise
                if getattr(self, "_no_disable", False):
                    thinking = True
        raise ProviderError("could not find a working thinking configuration")

    def close(self) -> None:
        self._client.close()