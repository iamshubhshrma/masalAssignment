"""Shared LLM plumbing: one JSON/text interface over Groq and Google Gemini.

Both are free-tier providers. Every call goes through `engine_chain()`, so if the
primary provider rate-limits or 503s the next one answers instead of the request
failing. Structured calls use each provider's native constrained decoding
(Groq strict `json_schema`, Gemini `responseSchema`) so we parse JSON, not prose.

Schemas avoid nullable/union types -- unknown values come back as "" and are
normalised to None here -- which keeps a single schema valid for both providers.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from .config import Settings

log = logging.getLogger("masal.llm")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


class LLMError(RuntimeError):
    """Every configured provider failed. `attempts` lists what each one said."""

    def __init__(self, message: str, attempts: list[str] | None = None):
        super().__init__(message)
        self.attempts = attempts or []


# --------------------------------------------------------------------------- #
# Provider calls
# --------------------------------------------------------------------------- #
async def _groq(
    system: str,
    messages: list[dict[str, str]],
    settings: Settings,
    http: httpx.AsyncClient,
    schema: dict[str, Any] | None,
    max_tokens: int,
) -> str:
    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set")

    body: dict[str, Any] = {
        "model": settings.groq_model,
        "temperature": 0.2 if schema else 0.4,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}, *messages],
    }
    if schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "result", "strict": True, "schema": schema},
        }

    last = ""
    for attempt in range(2):
        resp = await http.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json=body,
        )
        if resp.status_code < 400:
            content = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                raise LLMError("Groq returned no content")
            return content

        last = f"groq {resp.status_code}: {resp.text[:200]}"
        # Strict mode occasionally drops a field, and the shared tier 429s.
        retryable = resp.status_code == 429 or (
            resp.status_code == 400 and "json_validate_failed" in resp.text
        )
        if attempt == 0 and retryable:
            await asyncio.sleep(1.0)
            continue
        raise LLMError(last)
    raise LLMError(last)


# Gemini's responseSchema accepts only a subset of JSON Schema. Anything else --
# notably `additionalProperties`, which Groq's strict mode *requires* -- is a hard
# 400, so the shared schema is translated rather than duplicated.
_GEMINI_ALLOWED = {
    "type", "format", "description", "nullable", "enum",
    "items", "properties", "required", "propertyOrdering", "minItems", "maxItems",
}


def gemini_schema(node: Any) -> Any:
    """Strip keys Gemini rejects, recursively."""
    if isinstance(node, list):
        return [gemini_schema(n) for n in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for k, v in node.items():
        if k not in _GEMINI_ALLOWED:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {pk: gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = gemini_schema(v)
        else:
            out[k] = v
    return out


async def _gemini(
    system: str,
    messages: list[dict[str, str]],
    settings: Settings,
    http: httpx.AsyncClient,
    schema: dict[str, Any] | None,
    max_tokens: int,
) -> str:
    if not settings.google_api_key:
        raise LLMError("GOOGLE_API_KEY is not set")

    contents = [
        {"role": "model" if m["role"] == "assistant" else "user",
         "parts": [{"text": m["content"]}]}
        for m in messages
    ]
    gen: dict[str, Any] = {
        "temperature": 0.2 if schema else 0.4,
        "maxOutputTokens": max_tokens,
    }
    if schema:
        # Gemini 2.5 Flash thinks by default and those tokens come out of the
        # same budget, which silently truncates the JSON mid-string. Structured
        # extraction does not need it.
        gen["thinkingConfig"] = {"thinkingBudget": 0}
    if schema:
        gen["responseMimeType"] = "application/json"
        gen["responseSchema"] = gemini_schema(schema)

    resp = await http.post(
        GEMINI_URL.format(model=settings.gemini_model),
        params={"key": settings.google_api_key},
        headers={"Content-Type": "application/json"},
        json={
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": gen,
        },
    )
    if resp.status_code >= 400:
        raise LLMError(f"gemini {resp.status_code}: {resp.text[:200]}")

    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMError(f"gemini returned no candidates: {str(data)[:160]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    text = next((p["text"] for p in reversed(parts) if p.get("text")), None)
    if not text:
        raise LLMError("gemini returned no text part")
    return text


_PROVIDERS = {"groq": _groq, "gemini": _gemini}


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #
async def call_text(
    system: str,
    messages: list[dict[str, str]],
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    max_tokens: int = 900,
) -> tuple[str, str]:
    """Free-text completion. Returns (text, engine)."""
    text, engine, _ = await _run(system, messages, settings, client, None, max_tokens)
    return text, engine


async def call_json(
    system: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any],
    settings: Settings,
    client: httpx.AsyncClient | None = None,
    max_tokens: int = 2000,
) -> tuple[dict[str, Any], str]:
    """Schema-constrained completion. Returns (parsed dict, engine)."""
    text, engine, _ = await _run(system, messages, settings, client, schema, max_tokens)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{engine} returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError(f"{engine} returned {type(payload).__name__}, expected an object")
    return payload, engine


async def _run(
    system: str,
    messages: list[dict[str, str]],
    settings: Settings,
    client: httpx.AsyncClient | None,
    schema: dict[str, Any] | None,
    max_tokens: int,
) -> tuple[str, str, list[str]]:
    chain = [e for e in settings.engine_chain() if e in _PROVIDERS]
    if not chain:
        raise LLMError(
            "No AI provider configured. Set GROQ_API_KEY or GOOGLE_API_KEY in .env."
        )

    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    attempts: list[str] = []
    try:
        for name in chain:
            try:
                text = await _PROVIDERS[name](system, messages, settings, http, schema, max_tokens)
                model = settings.groq_model if name == "groq" else settings.gemini_model
                return text, f"{name}:{model}", attempts
            except LLMError as exc:
                log.warning("provider %s failed: %s", name, exc)
                attempts.append(str(exc))
            except httpx.RequestError as exc:
                log.warning("provider %s unreachable: %s", name, exc)
                attempts.append(f"{name} unreachable: {exc}")
        raise LLMError("every AI provider failed: " + " | ".join(attempts), attempts)
    finally:
        if own:
            await http.aclose()


# --------------------------------------------------------------------------- #
# Schema helpers
# --------------------------------------------------------------------------- #
def strict_object(properties: dict[str, Any]) -> dict[str, Any]:
    """Object schema valid for Groq strict mode and Gemini alike."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def blanks_to_none(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for k, v in list(out.items()):
        if isinstance(v, str) and not v.strip():
            out[k] = None
    return out
