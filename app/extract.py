"""Turn a call transcript into a structured ConfirmedLead.

Three engines behind one interface:
  groq   -- Groq chat completions, strict json_schema response format
  gemini -- Google Generative Language, responseSchema + JSON mime type
  rules  -- offline keyword heuristics, always available, no key

`extract_lead` walks Settings.engine_chain() and returns the first success, so a
provider outage or a missing key degrades instead of failing the request.

Both LLM schemas avoid nullable types -- unknown values come back as "" and are
normalised to None here. That keeps one schema valid for both providers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import httpx

from .config import Settings
from .models import ConfirmedLead

log = logging.getLogger("masal.extract")

SYSTEM_PROMPT = """You analyse transcripts of outbound phone calls made to confirm real-estate enquiries.

From the transcript, extract what the PROSPECT (the "user" turns) actually said. Rules:
- Judge only on the user's own words. Never infer interest from the agent's pitch.
- confirmed = true only when the user affirms a genuine, live interest in buying or renting.
  If they have already bought elsewhere, ask to be removed, deny enquiring, or never
  engaged (voicemail, silence, wrong number), confirmed = false.
- confirmed and interest_level must agree: hot/warm/cold all mean the user IS still a
  live lead, so those require confirmed = true. Use not_interested (or unknown, when
  nothing was said) for confirmed = false. A lukewarm "still looking, not urgent"
  prospect is a confirmed lead -- interest_level cold or warm, not not_interested.
- do_not_call = true if they ask not to be contacted again.
- Use "" for any field the user did not state. Never guess or invent a value.
- Quote budget, timeline and configuration in the user's own units (e.g. "1.4 crore", "3BHK").
- objections: short phrases for concerns they raised (e.g. "price too high", "location far").
- confidence: 0.0-1.0, how clearly the transcript supports your verdict.
- summary: one factual sentence a sales rep can act on."""

# Shared schema (no nullable types -> valid for Groq strict mode AND Gemini).
_PROPS: dict[str, Any] = {
    "confirmed": {"type": "boolean"},
    "interest_level": {
        "type": "string",
        "enum": ["hot", "warm", "cold", "not_interested", "unknown"],
    },
    "property_type": {"type": "string"},
    "configuration": {"type": "string"},
    "budget": {"type": "string"},
    "preferred_location": {"type": "string"},
    "timeline": {"type": "string"},
    "financing": {"type": "string"},
    "site_visit": {"type": "boolean"},
    "site_visit_time": {"type": "string"},
    "callback_requested": {"type": "boolean"},
    "callback_time": {"type": "string"},
    "do_not_call": {"type": "boolean"},
    "objections": {"type": "array", "items": {"type": "string"}},
    "summary": {"type": "string"},
    "confidence": {"type": "number"},
}
_REQUIRED = list(_PROPS)

GROQ_SCHEMA = {
    "type": "object",
    "properties": _PROPS,
    "required": _REQUIRED,
    "additionalProperties": False,
}
GEMINI_SCHEMA = {"type": "object", "properties": _PROPS, "required": _REQUIRED}


class ExtractionError(RuntimeError):
    pass


def _blank_to_none(payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload)
    for k, v in list(out.items()):
        if isinstance(v, str) and not v.strip():
            out[k] = None
    if out.get("summary") is None:
        out["summary"] = ""
    if not isinstance(out.get("objections"), list):
        out["objections"] = []
    else:
        out["objections"] = [str(o).strip() for o in out["objections"] if str(o).strip()]
    return out


def _reconcile(lead: ConfirmedLead) -> ConfirmedLead:
    """Keep `confirmed` and `interest_level` consistent.

    Models sometimes return interest_level="warm" alongside confirmed=false, which
    would silently drop a live lead from the confirmed list. hot/warm/cold all mean
    the prospect is still in play; a do-not-call always wins.
    """
    if lead.do_not_call:
        lead.confirmed = False
        lead.interest_level = "not_interested"
    elif lead.interest_level in {"hot", "warm", "cold"}:
        lead.confirmed = True
    elif lead.interest_level == "not_interested":
        lead.confirmed = False
    elif lead.interest_level == "unknown" and lead.confirmed:
        # Affirmed interest but no level given -- treat as warm rather than losing it.
        lead.interest_level = "warm"
    return lead


def _to_lead(payload: dict[str, Any], engine: str) -> ConfirmedLead:
    data = _blank_to_none(payload)
    data["engine"] = engine
    allowed = set(ConfirmedLead.model_fields)
    lead = ConfirmedLead.model_validate({k: v for k, v in data.items() if k in allowed})
    return _reconcile(lead)


def _user_prompt(transcript: str, lead_context: dict[str, Any] | None) -> str:
    ctx = ""
    if lead_context:
        pairs = [f"{k}: {v}" for k, v in lead_context.items() if v]
        if pairs:
            ctx = "Known enquiry details:\n" + "\n".join(pairs) + "\n\n"
    return f"{ctx}Call transcript:\n{transcript}\n\nExtract the lead as JSON."


# --------------------------------------------------------------------------- #
# Groq
# --------------------------------------------------------------------------- #
async def extract_groq(
    transcript: str,
    settings: Settings,
    lead_context: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> ConfirmedLead:
    if not settings.groq_api_key:
        raise ExtractionError("GROQ_API_KEY is not set")

    body = {
        "model": settings.groq_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(transcript, lead_context)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "confirmed_lead", "strict": True, "schema": GROQ_SCHEMA},
        },
    }
    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    try:
        last = ""
        # Strict mode requires every property; the model very occasionally drops one,
        # and shared-tier 429s happen. Both are worth one cheap retry before we
        # give up this engine and fall through to the next in the chain.
        for attempt in range(2):
            resp = await http.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                json=body,
            )
            if resp.status_code < 400:
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                if not content:
                    raise ExtractionError("Groq returned no content")
                return _to_lead(json.loads(content), f"groq:{settings.groq_model}")

            last = f"Groq {resp.status_code}: {resp.text[:300]}"
            retryable = resp.status_code == 429 or (
                resp.status_code == 400 and "json_validate_failed" in resp.text
            )
            if attempt == 0 and retryable:
                log.info("groq %s -- retrying once", resp.status_code)
                await asyncio.sleep(1.0)
                continue
            raise ExtractionError(last)
        raise ExtractionError(last or "Groq failed")
    except httpx.RequestError as exc:
        raise ExtractionError(f"Groq unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Groq returned invalid JSON: {exc}") from exc
    finally:
        if own:
            await http.aclose()


# --------------------------------------------------------------------------- #
# Google Gemini
# --------------------------------------------------------------------------- #
async def extract_gemini(
    transcript: str,
    settings: Settings,
    lead_context: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> ConfirmedLead:
    if not settings.google_api_key:
        raise ExtractionError("GOOGLE_API_KEY is not set")

    model = settings.gemini_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": _user_prompt(transcript, lead_context)}]}
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
        },
    }
    own = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))
    try:
        resp = await http.post(
            url, params={"key": settings.google_api_key},
            headers={"Content-Type": "application/json"}, json=body,
        )
        if resp.status_code >= 400:
            raise ExtractionError(f"Gemini {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise ExtractionError(f"Gemini returned no candidates: {str(data)[:200]}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = next((p["text"] for p in reversed(parts) if p.get("text")), None)
        if not text:
            raise ExtractionError("Gemini returned no text part")
        return _to_lead(json.loads(text), f"gemini:{model}")
    except httpx.RequestError as exc:
        raise ExtractionError(f"Gemini unreachable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"Gemini returned invalid JSON: {exc}") from exc
    finally:
        if own:
            await http.aclose()


# --------------------------------------------------------------------------- #
# Offline heuristics
# --------------------------------------------------------------------------- #
_NEGATIVE = (
    "not interested", "no longer interested", "already bought", "already purchased",
    "already booked", "not looking", "wrong number", "didn't enquire", "did not enquire",
    "no thanks", "not required", "cancel my", "close my enquiry",
)
_DNC = (
    "do not call", "don't call", "dont call", "stop calling", "remove me",
    "remove my number", "unsubscribe", "never call",
)
_POSITIVE = (
    "interested", "yes", "looking for", "looking to buy", "want to buy", "keen",
    "definitely", "still looking", "go ahead", "tell me more",
)
_BUDGET = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(crore|cr\b|lakhs?|lacs?|lakh|million|k\b)", re.I
)
_CONFIG = re.compile(r"(\d)\s*[- ]?\s*(?:bhk|bedroom)", re.I)
_PROPERTY = {
    "villa": "villa", "plot": "plot", "apartment": "apartment", "flat": "apartment",
    "commercial": "commercial", "office": "commercial", "studio": "studio",
    "penthouse": "penthouse", "row house": "row house", "duplex": "duplex",
    "shop": "commercial", "warehouse": "commercial",
}
_TIMELINE = re.compile(
    r"(immediately|right away|asap|this month|next month|this week|next week|"
    r"(?:in\s+)?(?:a|one|two|three|four|five|six|\d+)\s*(?:month|months|week|weeks|year|years)|"
    r"after diwali|end of the year|next year)", re.I
)
_FINANCE = {
    "home loan": "home loan", "loan": "loan", "emi": "loan", "mortgage": "loan",
    "cash": "cash", "self funded": "cash", "full payment": "cash", "outright": "cash",
}
_VISIT = ("site visit", "visit", "come and see", "see the property", "show me the",
          "walk through", "come over")
_VISIT_TIME = re.compile(
    r"((?:mon|tues|wednes|thurs|fri|satur|sun)day|tomorrow|today|this weekend|weekend)"
    r"(?:[^.,]{0,20}?(\d{1,2}(?::\d{2})?\s*(?:am|pm)))?", re.I
)
_CALLBACK = ("call me back", "call back", "callback", "call me later", "call me after",
             "ring me back", "get back to me")
_OBJECTIONS = {
    "too high": "price too high", "too expensive": "price too high",
    "prices": "price concern", "budget is tight": "budget constrained",
    "too far": "location too far", "far from": "location too far",
    "no parking": "parking concern", "small": "size concern",
    "not ready": "possession timing", "possession": "possession timing",
    "loan": None,  # handled as financing, not an objection
}


def extract_rules(
    transcript: str,
    settings: Settings | None = None,
    lead_context: dict[str, Any] | None = None,
) -> ConfirmedLead:
    """Keyword/heuristic extraction. Deterministic, offline, no key required."""
    user_lines = [
        line.split(":", 1)[1].strip()
        for line in transcript.splitlines()
        if line.lower().startswith("user:") and ":" in line
    ]
    user_text = " ".join(user_lines)
    low = user_text.lower()
    full_low = transcript.lower()

    if not user_text.strip():
        return ConfirmedLead(
            confirmed=False, interest_level="unknown", confidence=0.3, engine="rules",
            summary="No prospect speech in the transcript (unanswered, voicemail or silent call).",
        )

    signals = 0
    negative = any(p in low for p in _NEGATIVE)
    dnc = any(p in low for p in _DNC)
    positive = any(p in low for p in _POSITIVE)
    if negative or dnc or positive:
        signals += 1

    budget = None
    if (m := _BUDGET.search(user_text)):
        unit = m.group(2).lower().rstrip(".")
        unit = {"cr": "crore", "lac": "lakh", "lacs": "lakh", "lakhs": "lakh"}.get(unit, unit)
        budget = f"{m.group(1)} {unit}"
        signals += 1

    configuration = None
    if (m := _CONFIG.search(user_text)):
        configuration = f"{m.group(1)}BHK"
        signals += 1
    elif "studio" in low:
        configuration = "studio"

    property_type = next((v for k, v in _PROPERTY.items() if k in low), None)
    if property_type is None:
        property_type = next((v for k, v in _PROPERTY.items() if k in full_low), None)
    if property_type:
        signals += 1

    timeline = None
    if (m := _TIMELINE.search(user_text)):
        timeline = m.group(1).strip()
        signals += 1

    financing = next((v for k, v in _FINANCE.items() if k in low), None)
    if financing:
        signals += 1

    # The prospect may agree to a visit the agent proposed ("Saturday 11am works"),
    # so accept a visit offer anywhere in the transcript plus a user-side day/time.
    user_visit_words = any(p in low for p in _VISIT)
    visit_offered = any(p in full_low for p in _VISIT)
    user_time = _VISIT_TIME.search(user_text)
    site_visit = (user_visit_words or (visit_offered and user_time is not None)) and not negative
    site_visit_time = None
    if site_visit and user_time:
        site_visit_time = " ".join(p for p in user_time.groups() if p).strip()
        signals += 1

    callback_requested = any(p in low for p in _CALLBACK)
    callback_time = None
    if callback_requested and (m := _TIMELINE.search(user_text)):
        callback_time = m.group(1).strip()

    objections = sorted({v for k, v in _OBJECTIONS.items() if v and k in low})

    location = None
    if (m := re.search(r"\bin\s+([A-Z][a-zA-Z]{3,})", user_text)):
        location = m.group(1)
    elif (m := re.search(r"\bin\s+([A-Z][a-zA-Z]{3,})", transcript)):
        location = m.group(1)

    confirmed = positive and not negative and not dnc

    if not confirmed:
        interest = "not_interested"
    elif site_visit or (timeline and re.search(r"immediat|asap|right away|this month|this week|one month|two month|1 month|2 month|tomorrow", timeline, re.I)):
        interest = "hot"
    elif callback_requested or objections:
        interest = "cold"
    else:
        interest = "warm"

    if confirmed:
        bits = [
            f"Confirmed interest in {configuration or property_type or 'a property'}",
            f"budget {budget}" if budget else "",
            f"timeline {timeline}" if timeline else "",
            f"site visit {site_visit_time}" if site_visit_time else
            ("site visit agreed" if site_visit else ""),
            f"financing via {financing}" if financing else "",
        ]
        summary = ", ".join(b for b in bits if b) + "."
    elif dnc:
        summary = "Asked not to be contacted again -- suppress this number."
    elif negative:
        summary = "No longer a live enquiry (already bought, not looking, or denied enquiring)."
    else:
        summary = "Prospect engaged but gave no clear confirmation of interest."

    confidence = min(0.85, 0.35 + 0.1 * signals)
    return _reconcile(ConfirmedLead(
        confirmed=confirmed, interest_level=interest, property_type=property_type,
        configuration=configuration, budget=budget, preferred_location=location,
        timeline=timeline, financing=financing, site_visit=site_visit,
        site_visit_time=site_visit_time, callback_requested=callback_requested,
        callback_time=callback_time, do_not_call=dnc, objections=objections,
        summary=summary, confidence=round(confidence, 2), engine="rules",
    ))


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
async def extract_lead(
    transcript: str,
    settings: Settings,
    lead_context: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> ConfirmedLead:
    """Run the configured engine chain, returning the first successful extraction."""
    errors: list[str] = []
    for engine in [*settings.engine_chain(), "rules"]:
        try:
            if engine == "groq":
                return await extract_groq(transcript, settings, lead_context, client)
            if engine == "gemini":
                return await extract_gemini(transcript, settings, lead_context, client)
            if engine == "rules":
                return extract_rules(transcript, settings, lead_context)
            errors.append(f"{engine}: unknown engine")
        except ExtractionError as exc:
            log.warning("extractor %s failed: %s", engine, exc)
            errors.append(f"{engine}: {exc}")
        except Exception as exc:  # noqa: BLE001 - never let one engine kill the chain
            log.exception("extractor %s crashed", engine)
            errors.append(f"{engine}: {exc!r}")

    lead = extract_rules(transcript, settings, lead_context)
    lead.engine = "rules(after-failures)"
    lead.summary = (lead.summary + f" [engine fallbacks: {'; '.join(errors)}]")[:800]
    return lead
