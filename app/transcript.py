"""Transcript normalisation.

Ringg's call-details payload is inconsistent about how it carries the
conversation. The documented shape is an array of `{"bot": ...}` / `{"user": ...}`
objects under `transcription_url`, but the field name is literal: it can also be
an actual URL that must be fetched, a newline-delimited string, or a list using
`role`/`content` style keys. Older payloads use `transcript`.

`resolve_turns` accepts all of them so a transcript that exists in the dashboard
is never silently dropped.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from .models import Turn, parse_turns

log = logging.getLogger("masal.transcript")

# Checked in order; first one holding usable content wins.
TRANSCRIPT_KEYS = (
    "transcription_url",
    "transcript",
    "transcripts",
    "call_transcript",
    "conversation",
    "messages",
    "turns",
    "dialogue",
)


def _looks_like_url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().startswith(("http://", "https://"))


def _parse_payload(payload: Any) -> list[Turn]:
    """Parse an already-fetched transcript body: JSON array, JSON object, or text."""
    if isinstance(payload, (list, dict)):
        turns = parse_turns(payload if isinstance(payload, list) else [payload])
        if turns:
            return turns
        if isinstance(payload, dict):
            for key in TRANSCRIPT_KEYS:
                if key in payload:
                    inner = _parse_payload(payload[key])
                    if inner:
                        return inner
        return []

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return []
        # JSON delivered as a string
        if text[0] in "[{":
            try:
                return _parse_payload(json.loads(text))
            except json.JSONDecodeError:
                pass
        return parse_turns(text)
    return []


async def fetch_transcript_url(url: str, client: httpx.AsyncClient | None = None) -> list[Turn]:
    """Download a transcript that Ringg exposed as a URL."""
    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        resp = await http.get(url)
        if resp.status_code >= 400:
            log.warning("transcript url %s -> HTTP %s", url, resp.status_code)
            return []
        try:
            return _parse_payload(resp.json())
        except ValueError:
            return _parse_payload(resp.text)
    except httpx.RequestError as exc:
        log.warning("could not fetch transcript url: %s", exc)
        return []
    finally:
        if own:
            await http.aclose()


async def resolve_turns(
    details: dict[str, Any], client: httpx.AsyncClient | None = None
) -> tuple[list[Turn], str | None]:
    """Pull the conversation out of a call-details payload, whatever its shape.

    Returns (turns, note). `note` is a short diagnostic when nothing was found,
    naming the keys that were actually present.
    """
    if not isinstance(details, dict):
        return [], "call details were not an object"

    for key in TRANSCRIPT_KEYS:
        if key not in details:
            continue
        value = details[key]
        if value in (None, "", [], {}):
            continue

        if _looks_like_url(value):
            turns = await fetch_transcript_url(value.strip(), client)
            if turns:
                log.info("transcript fetched from url in `%s` (%d turns)", key, len(turns))
                return turns, None
            continue

        turns = _parse_payload(value)
        if turns:
            log.info("transcript read from `%s` (%d turns)", key, len(turns))
            return turns, None

    present = sorted(k for k, v in details.items() if v not in (None, "", [], {}))
    note = "no transcript field recognised; payload keys: " + ", ".join(present[:25])
    log.warning(note)
    return [], note
