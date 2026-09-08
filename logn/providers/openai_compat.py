"""Adapters for providers speaking the OpenAI chat-completions API.

Four vendors share one implementation here: OpenAI, xAI (Grok), Z.ai (GLM) and
Moonshot (Kimi). They differ in base URL, in how reasoning is requested, and in
whether prompt caching needs to be asked for.

Caching is implicit on all four: send a stable prefix and the provider matches
it automatically, usually above a ~1024-token threshold. There is no breakpoint
to mark, so Prompt.cacheable and Prompt.tail are simply concatenated in that
order. Cached tokens come back in usage.prompt_tokens_details.cached_tokens and
are billed at a discount, so Usage subtracts them from input_tokens to keep the
fields disjoint.

Reasoning parameters are the fragile part. These models postdate this code's
last verification, so rather than hardcoding what each accepts, the adapter
sends its best guess and drops any parameter the API rejects with a 400,
recording that it did so in `dropped_params`. Check that field on the first run
of a new model: an empty list means everything was accepted.
"""

from __future__ import annotations

import re
import time

import httpx

from ..keys import get_key
from .base import (Completion, ProviderError, TransientError, Usage,
                   is_transient_status, with_retries)

# Anthropic's effort scale is the one game.py speaks. Map it per vendor.
EFFORT_TO_OPENAI = {"low": "low", "medium": "medium", "high": "high",
                    "xhigh": "high", "max": "high"}


class OpenAICompatClient:
    """Base for chat-completions providers. Subclasses set the class attrs."""

    provider = ""
    base_url = ""
    reasoning_style = "none"      # "effort" | "thinking_flag" | "none"
    # Whether the vendor counts reasoning tokens inside completion_tokens.
    # OpenAI does; xAI reports them alongside instead, which understates cost
    # by ~90% on a reasoning-heavy workload if taken at face value. The symptom
    # is reasoning_tokens > completion_tokens in a response.
    reasoning_in_completion = True
    # Vendors differ in which effort levels exist. Override per subclass.
    EFFORT_MAP = EFFORT_TO_OPENAI

    def __init__(self, name: str, model_id: str, *, effort: str = "high",
                 reasoning_effort: str = "", timeout: float = 900.0):
        self.name = name
        self.model_id = model_id
        self.effort = effort
        self.reasoning_effort = reasoning_effort or EFFORT_TO_OPENAI.get(effort, "high")
        self.dropped_params: list[str] = []
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {get_key(self.provider)}",
                "Content-Type": "application/json",
            },
        )

    # ----------------------------------------------------------------
    def _reasoning_fields(self, thinking: bool, effort: str | None) -> dict:
        """Vendor-specific reasoning knobs, before any get dropped."""
        eff = effort or self.effort
        if self.reasoning_style == "effort":
            return {"reasoning_effort": self.EFFORT_MAP.get(eff, "high")
                    if thinking else self.EFFORT_MAP.get("low", "low")}
        if self.reasoning_style == "thinking_flag":
            # GLM takes an effort level alongside the flag, and defaults to
            # "max" if omitted -- which would run this arm harder than every
            # other. Levels are low/high/max only: no medium, so the answerer's
            # medium maps down to low.
            level = {"low": "low", "medium": "low", "high": "high",
                     "xhigh": "max", "max": "max"}.get(eff, "high")
            return {"thinking": {"type": "enabled" if thinking else "disabled"},
                    "reasoning_effort": level}

    def _body(self, prompt, max_tokens: int, temperature: float,
              thinking: bool, effort: str | None) -> dict:
        # Documents first, history second: the stable prefix is what implicit
        # caching matches on.
        body = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.as_text()},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body.update(self._reasoning_fields(thinking, effort))
        for p in self.dropped_params:
            body.pop(p, None)
        return body

    # ----------------------------------------------------------------
    def _once(self, body: dict) -> Completion:
        t0 = time.perf_counter()
        try:
            r = self._client.post(self.base_url, json=body)
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
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        text = msg.get("content") or ""
        # Several of these expose the chain of thought on a side field.
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "")

        u = data.get("usage", {}) or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                  or u.get("cached_tokens") or 0)
        reasoning_toks = ((u.get("completion_tokens_details") or {})
                          .get("reasoning_tokens")
                          or u.get("reasoning_tokens") or 0)
        prompt_toks = u.get("prompt_tokens", 0)
        completion = u.get("completion_tokens", 0)
        # Billing is on visible + reasoning either way, so normalise to the
        # total regardless of how the vendor chose to report it.
        total_out = (completion if self.reasoning_in_completion
                     else completion + reasoning_toks)

        return Completion(
            text=(text or "").strip(),
            reasoning_text=reasoning or "",
            usage=Usage(
                # prompt_tokens includes cached; keep the fields disjoint.
                input_tokens=max(0, prompt_toks - cached),
                cache_read_tokens=cached,
                cache_write_tokens=0,          # implicit caching: no write cost
                output_tokens=total_out,
                reasoning_tokens=reasoning_toks or (len(reasoning) // 4
                                                    if reasoning else 0),
            ),
            model_id=data.get("model", self.model_id),
            stop_reason=choice.get("finish_reason", ""),
            latency_s=elapsed,
        )

    # ----------------------------------------------------------------
    _PARAM_RE = re.compile(
        r"'([a-z_.]+)'|\"([a-z_.]+)\"|Unrecognized request argument[:\s]+([a-z_.]+)",
        re.IGNORECASE)

    def _adapt(self, msg: str) -> bool:
        """Drop a parameter the API rejected, so the run continues instead of
        dying on a vendor's naming difference. Returns True if worth retrying."""
        candidates = ("reasoning_effort", "thinking", "temperature", "max_tokens")
        low = msg.lower()
        for p in candidates:
            if p in low and p not in self.dropped_params:
                # max_tokens is required by most; only drop if named explicitly
                # as unsupported, not merely mentioned.
                if p == "max_tokens" and "unsupported" not in low \
                        and "unrecognized" not in low:
                    continue
                self.dropped_params.append(p)
                return True
        return False

    def complete(self, prompt, *, max_tokens: int, temperature: float,
                 thinking: bool = True, effort: str | None = None) -> Completion:
        for _ in range(4):
            try:
                return with_retries(lambda: self._once(
                    self._body(prompt, max_tokens, temperature, thinking, effort)))
            except ProviderError as e:
                if not self._adapt(str(e)):
                    raise
        raise ProviderError(
            f"{self.name}: no working parameter set; dropped {self.dropped_params}")

    def close(self) -> None:
        self._client.close()


# --------------------------------------------------------------------------
# vendors
# --------------------------------------------------------------------------

class OpenAIClient(OpenAICompatClient):
    provider = "openai"
    base_url = "https://api.openai.com/v1/chat/completions"
    reasoning_style = "effort"

    def _body(self, prompt, max_tokens, temperature, thinking, effort):
        body = super()._body(prompt, max_tokens, temperature, thinking, effort)
        # Reasoning models use max_completion_tokens and reject max_tokens;
        # the cap covers reasoning tokens as well as visible output.
        if "max_tokens" in body and "max_tokens" not in self.dropped_params:
            body["max_completion_tokens"] = body.pop("max_tokens")
        return body


class XAIClient(OpenAICompatClient):
    provider = "xai"
    base_url = "https://api.x.ai/v1/chat/completions"
    reasoning_style = "effort"
    # Observed: completion_tokens=308 with reasoning_tokens=3853 on one call,
    # so reasoning sits outside the completion count here.
    reasoning_in_completion = False


class ZAIClient(OpenAICompatClient):
    provider = "zai"
    base_url = "https://api.z.ai/api/paas/v4/chat/completions"
    reasoning_style = "thinking_flag"
    # Unverified. Check the first game: if reasoning_tokens > output_tokens,
    # set this False.
    reasoning_in_completion = True


class MoonshotClient(OpenAICompatClient):
    provider = "moonshot"
    base_url = "https://api.moonshot.ai/v1/chat/completions"
    # K3 thinks unconditionally; the level is set by a top-level
    # reasoning_effort and DEFAULTS TO "max", so omitting it would run this
    # arm harder than the others. Only low/high/max exist -- no medium, so the
    # answerer maps down to low, as with GLM.
    reasoning_style = "effort"
    EFFORT_MAP = {"low": "low", "medium": "low", "high": "high",
                  "xhigh": "max", "max": "max"}
    reasoning_in_completion = True   # unverified; check the first game

class DeepSeekClient(OpenAICompatClient):
    provider = "deepseek"
    base_url = "https://api.deepseek.com/chat/completions"
    # V4 takes both the thinking flag and reasoning_effort, and thinks by
    # default -- same shape as GLM. Levels are undocumented beyond "high";
    # the self-healing path drops the parameter if a level is rejected.
    reasoning_style = "thinking_flag"
    reasoning_in_completion = True   # unverified; check the first game