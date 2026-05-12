"""
omnirca/llm_client.py — Shared Azure OpenAI client singleton.

Used by:
  - omnirca/agent/react_agent.py   (ReAct loop)
  - omnirca/tools/rca_tools.py     (explain_evidence)

Credentials are loaded from a .env file (via python-dotenv) or from
environment variables set directly in the shell.  Copy .env.example to
.env and fill in your values before running.

Required environment variables:
  AZURE_OPENAI_KEY      — API key
  AZURE_OPENAI_ENDPOINT — https://your-resource.cognitiveservices.azure.com/
  AZURE_DEPLOYMENT      — deployment name (e.g. gpt-4o-2)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AzureOpenAI
from omnirca.logger import get_logger

# Load .env from the omnirca package root (one level up from this file's dir
# if installed as a package, or the same dir if run directly).
load_dotenv(Path(__file__).parent / ".env")

_log = get_logger("omnirca.llm")

_API_VERSION = "2024-08-01-preview"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Required environment variable '{name}' is not set. "
            "Copy .env.example to .env and fill in your credentials."
        )
    return value


@lru_cache(maxsize=1)
def get_client() -> AzureOpenAI:
    """Return the cached AzureOpenAI client (initialised once per process)."""
    return AzureOpenAI(
        api_key        = _require_env("AZURE_OPENAI_KEY"),
        api_version    = _API_VERSION,
        azure_endpoint = _require_env("AZURE_OPENAI_ENDPOINT"),
        timeout        = 120,
        max_retries    = 3,
    )


def chat(
    messages:     list[dict],
    tools:        list[dict] | None = None,
    tool_choice:  str | dict = "auto",
    temperature:  float = 0.0,
    max_tokens:   int = 2048,
) -> Any:
    """
    Thin wrapper around client.chat.completions.create().

    Returns the raw ChatCompletion response object.
    Keeps the function signature stable so callers don't need to know the
    deployment name or API version.
    """
    client = get_client()
    kwargs: dict[str, Any] = dict(
        model       = _require_env("AZURE_DEPLOYMENT"),
        messages    = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )
    if tools:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = tool_choice

    tool_names = [t["function"]["name"] for t in (tools or [])]
    _log.info(
        "LLM request — %d messages | tools=[%s]",
        len(messages),
        ", ".join(tool_names) if tool_names else "none",
    )

    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    if msg.tool_calls:
        called = [tc.function.name for tc in msg.tool_calls]
        _log.info("LLM response — tool_calls: %s", called)
    else:
        snippet = (msg.content or "")[:150].replace("\n", " ")
        _log.info("LLM response — text: %s", snippet)

    return response


def simple_chat(user_prompt: str, system_prompt: str = "", temperature: float = 0.3) -> str:
    """
    Convenience helper for single-turn text-only LLM calls.
    Returns the assistant's text string directly.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    _log.info("LLM simple_chat — prompt: %s", user_prompt[:100].replace("\n", " "))
    response = chat(messages, temperature=temperature)
    result = response.choices[0].message.content or ""
    _log.info("LLM simple_chat response — %s", result[:150].replace("\n", " "))
    return result
