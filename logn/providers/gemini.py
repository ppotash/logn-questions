"""Gemini adapter.

Google's generateContent API differs from the others in three ways that matter:

  - the system prompt is a separate `systemInstruction`, not a message
  - reasoning is a string enum, `thinkingLevel`, not a token budget. The
    integer `thinkingBudget` is deprecated for Gemini 3+ and is ignored when
    a level is present. Sending a budget alone on a Gemini 3 model silently
    gets you the model's default, which is HIGH.
  - caching is implicit above a model-dependent threshold. There is no
    breakpoint to mark; a stable prefix is matched automatically and reported
    as usageMetadata.cachedContentTokenCount. Observed not to engage at all on
    3.8 Flash at ~8K tokens, so do not count on it.

Sampling parameters (temperature, top_p, top_k) are ignored by the backend on
Gemini 3+ and are not sent. Determinism is controlled through thinkingLevel
instead. This means Gemini, like Opus 5 and GPT-5.6, cannot be pinned to
temperature 0 -- worth stating in any writeup rather than implying otherwise.

Thinking tokens are billed as output and reported separately as
thoughtsTokenCount, so output_tokens sums both and reasoning_tokens carries the
thinking portion.
"""

from __future__ import annotations

import time

import httpx

from ..keys import get_key
from .base import (Completion, ProviderError, TransientError, Usage,
                   is_transient_status, with_retries)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# game.py speaks Anthropic's five-level effort scale; Gemini has three.
# MINIMAL exists on some models but is rejected by 3.8 Flash, so it is never
# sent -- the answerer runs at its configured level instead.
EFFORT_TO_LEVEL = {"low": "LOW", "medium": "MEDIUM", "high": "HIGH",
                   "xhigh": "HIGH", "max": "HIGH"}

# Fallback for older models that still take an integer budget, used only if
# the API rejects thinkingLevel.
EFFORT_TO_BUDGET = {"low": 2048, "medium": 8192, "high": 16384,
                    "xhigh": 24576, "max": 32768}


class GeminiClient:
    provider = "gemini"

    def __init__(self, name: str, model_id: str, *, effort: str = "high",
                 reasoning_budget: int = 0, timeout: float = 900.0):
        self.name = name
        self.model_id = model_id
        self.effort = effort
        self.reasoning_budget = reasoning_budget
        # "level" is correct for Gemini 3+; the adapter falls back to "budget"
        # only if the API rejects the enum.
        self.thinking_api = "level"
        self.dropped_params: list[str] = []
        self._key = get_key("gemini")
        self._client = httpx.Client(
            timeout=timeout, headers={"Content-Type": "application/json"})

    # ----------------------------------------------------------------
    def _thinking_config(self, thinking: bool, effort: str | None) -> dict | None:
        eff = effort or self.effort
        if self.thinking_api == "level":
            # There is no "off" switch on 3.8 Flash: MINIMAL is rejected. A
            # call that asks for no thinking gets the lowest level instead.
            level = "LOW" if not thinking else EFFORT_TO_LEVEL.get(eff, "HIGH")
            return {"thinkingLevel": level, "includeThoughts": True}
        budget = 0 if not thinking else (
            self.reasoning_budget or EFFORT_TO_BUDGET.get(eff, 16384))
        return {"thinkingBudget": budget, "includeThoughts": True}

    def _body(self, prompt, max_tokens: int, temperature: float,
              thinking: bool, effort: str | None) -> dict:
        gen: dict = {"maxOutputTokens": max_tokens}
        # temperature/top_p/top_k are ignored by Gemini 3+ and sending
        # deprecated fields risks a validation error, so they are omitted.
        if "thinkingConfig" not in self.dropped_params:
            cfg = self._thinking_config(thinking, effort)
            if cfg:
                gen["thinkingConfig"] = cfg
        return {
            "systemInstruction": {"parts": [{"text": prompt.system}]},
            # Documents first, history second: the stable prefix is what
            # implicit caching matches on, where it engages at all.
            "contents": [{"role": "user",
                          "parts": [{"text": prompt.as_text()}]}],
            "generationConfig": gen,
        }

    # ----------------------------------------------------------------
    def _once(self, body: dict) -> Completion:
        url = f"{API_ROOT}/{self.model_id}:generateContent?key={self._key}"
        t0 = time.perf_counter()
        try:
            r = self._client.post(url, json=body)
        except httpx.TimeoutException as e:
            raise TransientError(f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientError(f"transport: {e}") from e
        elapsed = time.perf_counter() - t0

        if r.status_code != 200:
            # The key travels in the query string; never let it reach a log.
            detail = r.text[:500].replace(self._key, "<redacted>")
            if is_transient_status(r.status_code):
                raise TransientError(f"{r.status_code}: {detail}")
            raise ProviderError(f"{r.status_code}: {detail}")

        data = r.json()
        cand = (data.get("candidates") or [{}])[0]
        text_parts, think_parts = [], []
        for part in (cand.get("content", {}) or {}).get("parts", []) or []:
            t = part.get("text", "")
            if not t:
                continue
            # Thought summaries are flagged; without the flag it is the answer.
            (think_parts if part.get("thought") else text_parts).append(t)

        u = data.get("usageMetadata", {}) or {}
        cached = u.get("cachedContentTokenCount", 0) or 0
        thoughts = u.get("thoughtsTokenCount", 0) or 0
        prompt_toks = u.get("promptTokenCount", 0) or 0

        return Completion(
            text="\n".join(text_parts).strip(),
            reasoning_text="\n".join(think_parts),
            usage=Usage(
                # promptTokenCount includes cached; keep the fields disjoint.
                input_tokens=max(0, prompt_toks - cached),
                cache_read_tokens=cached,
                cache_write_tokens=0,          # implicit caching: no write cost
                # Thinking is billed as output but reported separately.
                output_tokens=(u.get("candidatesTokenCount", 0) or 0) + thoughts,
                reasoning_tokens=thoughts,
            ),
            model_id=data.get("modelVersion", self.model_id),
            stop_reason=cand.get("finishReason", ""),
            latency_s=elapsed,
        )

    # ----------------------------------------------------------------
    def _adapt(self, msg: str) -> bool:
        """Recover from a contract mismatch. Each branch fires at most once."""
        low = msg.lower()
        if "thinkinglevel" in low and self.thinking_api == "level":
            # Older model that still wants an integer budget.
            self.thinking_api = "budget"
            return True
        if "thinkingconfig" in low or "thinking" in low:
            if "thinkingConfig" not in self.dropped_params:
                self.dropped_params.append("thinkingConfig")
                return True
        return False

    def complete(self, prompt, *, max_tokens: int, temperature: float,
                 thinking: bool = True, effort: str | None = None) -> Completion:
        for _ in range(3):
            try:
                return with_retries(lambda: self._once(
                    self._body(prompt, max_tokens, temperature, thinking, effort)))
            except ProviderError as e:
                if not self._adapt(str(e)):
                    raise
        raise ProviderError(
            f"{self.name}: no working thinking configuration; "
            f"api={self.thinking_api} dropped={self.dropped_params}")

    def close(self) -> None:
        self._client.close()