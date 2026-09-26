"""Bulk triage -- paste the mess, get ranked leads.

Real inbound does not arrive as tidy form submissions. It is a WhatsApp export,
a portal email digest, a pasted spreadsheet column. This splits one raw blob
into individual enquiries and structures each into the same intake shape the
form produces, so the rest of the pipeline is unchanged.

Splitting and field extraction are one AI call; each resulting lead is then
analysed by the normal `analyze_lead` path.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import Settings
from .llm import call_json
from .models import LeadIntake

log = logging.getLogger("masal.triage")

SYSTEM = """You split a raw blob of inbound real-estate enquiries into individual leads.

The blob may be a WhatsApp export, forwarded emails, portal notifications, chat
logs, or rows pasted from a spreadsheet. Formatting is inconsistent and messy.

For each distinct person enquiring, produce one entry. Rules:
- One entry per PERSON, not per message. Merge several messages from the same
  person into a single entry, combining what they said into `message`.
- Copy values across only when the text actually states them. Use "" otherwise.
  Never infer a budget or timeline that is not written down.
- `message` should preserve the customer's own words -- that is what the
  analysis step reads. Strip timestamps, phone-UI noise and "forwarded" banners.
- `name`: use the name given. If there is genuinely none, use the phone number,
  or "Unknown" as a last resort. Never invent a name.
- `phone`: only if a phone number appears. Keep the country code, E.164 style.
- Skip anything that is not an enquiry: delivery notifications, OTPs, your own
  outgoing messages, group chatter.
- If the blob contains no real enquiry at all, return an empty list."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "leads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "location": {"type": "string"},
                    "property_requirement": {"type": "string"},
                    "budget": {"type": "string"},
                    "timeline": {"type": "string"},
                    "message": {"type": "string"},
                    "phone": {"type": "string"},
                    "source": {"type": "string",
                               "description": "Where it came from if stated, e.g. WhatsApp."},
                },
                "required": ["name", "location", "property_requirement", "budget",
                             "timeline", "message", "phone", "source"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["leads"],
    "additionalProperties": False,
}

MAX_LEADS = 25


async def triage_blob(
    text: str,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> list[LeadIntake]:
    """Split raw text into structured intakes. Raises LLMError if the AI fails."""
    payload, engine = await call_json(
        SYSTEM,
        [{"role": "user", "content": f"Raw inbound:\n\"\"\"\n{text.strip()}\n\"\"\"\n\nSplit into leads."}],
        SCHEMA,
        settings,
        client,
        max_tokens=4000,
    )
    rows = payload.get("leads")
    if not isinstance(rows, list):
        return []

    intakes: list[LeadIntake] = []
    for row in rows[:MAX_LEADS]:
        if not isinstance(row, dict):
            continue
        row = {k: ("" if v is None else str(v).strip()) for k, v in row.items()}
        if not row.get("name"):
            row["name"] = row.get("phone") or "Unknown"
        try:
            intakes.append(LeadIntake.model_validate(row))
        except ValueError as exc:
            # A bad phone must not lose the whole lead -- drop the field and keep it.
            log.info("triage row rejected (%s); retrying without phone", exc)
            row["phone"] = ""
            try:
                intakes.append(LeadIntake.model_validate(row))
            except ValueError:
                log.warning("triage row dropped entirely: %s", row)
    log.info("triage split %d lead(s) via %s", len(intakes), engine)
    return intakes
