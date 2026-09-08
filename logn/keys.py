"""API key resolution.

Keys come from the environment. A .env file in the repo root fills any gaps, so
you don't have to export six variables in every shell. Real environment
variables win, which means a one-off `set ANTHROPIC_API_KEY=...` overrides the
file without editing it.

Two rules the adapters must hold to:
  - keys never enter a result file (log request bodies, never headers)
  - keys never enter a traceback (hold the client, don't pass the key around)
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # dotenv is optional; plain env vars still work
    pass


ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "zai": "ZAI_API_KEY",
    "moonshot": "MOONSHOT_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}


def get_key(provider: str) -> str:
    try:
        var = ENV_VAR[provider]
    except KeyError:
        raise KeyError(f"unknown provider {provider!r}; "
                       f"known: {', '.join(sorted(ENV_VAR))}") from None
    key = os.environ.get(var, "").strip()
    if not key:
        raise RuntimeError(
            f"{var} is not set. Add it to {REPO_ROOT / '.env'} or export it. "
            "Copy .env.example to .env if you haven't."
        )
    return key


def check_keys(providers) -> None:
    """Fail before a long run, not forty minutes into it.

    The runner should call this against every provider it is about to use,
    before issuing the first request.
    """
    missing = [ENV_VAR[p] for p in providers
               if not os.environ.get(ENV_VAR.get(p, ""), "").strip()]
    if missing:
        raise SystemExit("missing API keys: " + ", ".join(sorted(missing)))


def available() -> list[str]:
    """Providers with a key present. Useful for deciding what a pilot can run."""
    return [p for p, var in ENV_VAR.items() if os.environ.get(var, "").strip()]