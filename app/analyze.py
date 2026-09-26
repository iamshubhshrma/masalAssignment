"""Lead analysis: the AI call that turns a raw enquiry into a prioritised brief.

Produces the six outputs the salesperson needs (summary, intent, requirements,
objections, next action, suggested response) plus a 0-100 priority score with
its reasoning, which is what the lead list ranks on.

The rubric lives in the prompt rather than in Python because the inputs are free
text -- "around 1.4Cr, but flexible if the view is good" is not something a
regex can score. The *consistency* rules that follow (score/temperature
agreement) are enforced in code, because models drift on those.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from .config import Settings
from .llm import call_json, strict_object
from .models import Analysis, LeadIntake

log = logging.getLogger("masal.analyze")

SYSTEM = """You are an experienced real-estate sales manager triaging inbound leads for a busy agent.

You read one lead and return a brief the agent can act on in under ten seconds.

Ground every field in what the lead actually said. Never invent a budget, a
timeline, or a requirement that is not stated or clearly implied. If something
is missing, say so in missing_info rather than guessing.

SCORING RUBRIC -- priority_score, 0 to 100. Start at 50 and adjust:

  Budget         +20 stated and specific   +10 a usable range   -10 absent
                 -15 clearly below what their requirement implies
  Timeline       +20 within 1 month   +12 within 3 months   +4 3-6 months
                 -10 over a year, "just looking", or unstated
  Specificity    +15 names a configuration, area or project   -10 vague
  Intent signals +15 asks to visit, asks for a call, mentions loan pre-approval,
                     mentions selling an existing property to fund this
                 -20 explicitly comparing prices only, or says not serious
  Engagement     +10 wrote a detailed message   -5 one-line generic enquiry
  Red flags      -30 already bought, agent/broker fishing, wrong city, spam

Reserve scores below 15 for genuinely dead leads -- spam, wrong city, already
bought, a broker fishing. A real person with a real enquiry who is simply early
belongs at 20-40, not 0: the agent still needs them ranked against each other.

Then clamp to 0-100 and set:
  temperature  hot >= 75, warm 45-74, cold < 45
  urgency      high if action is needed today, medium within the week, else low

CALIBRATION -- this matters more than any single adjustment above.

A real inbound batch is mostly mediocre. Roughly 15% should land hot, 35% warm
and 50% cold. If you find yourself scoring most leads above 75 you are being
far too generous: re-read the rubric and apply the negatives. "Hot" means the
agent should drop what they are doing and call right now. It is not a reward
for the lead being pleasant or detailed.

Anchor examples -- match new leads against these:

  95  "3BHK Whitefield, 1.4cr, loan pre-approved, can I visit this Saturday?"
      Budget + short timeline + explicit visit request + financing sorted.

  72  "Looking for a 3BHK in Whitefield, budget around 1.4cr, sometime in the
       next 3-4 months." Specific and real, but no urgency and no ask.

  48  "Interested in 2BHK Sarjapur, under 80 lakhs, planning next year."
      Real requirement and a budget, but a year out. Nurture, do not chase.

  30  "What are villa prices in Sarjapur? Just browsing for now."
      A genuine person, no budget, no timeline, no commitment.

  10  "Send me your full inventory and commission structure." Broker fishing.

Note that the 48 example is COLD, not hot, despite naming a budget and a
configuration -- because the timeline is a year away. Timeline and a concrete
ask are what separate warm from hot, not the amount of detail given.

score_reasoning: one sentence naming the two or three factors that moved the
score most. The agent must be able to argue with it.

suggested_response: a ready-to-send message to the CUSTOMER, in their own
language and register, 2-4 sentences. No placeholders like [name] -- use their
actual name. Move the conversation one concrete step forward.

NEVER invent facts in that message. Do not quote prices, price ranges, market
rates, availability, floor plans, project names, offers or discounts that do not
appear in the lead above. You do not know the inventory. If the customer asked
about price, do not answer with a number -- ask the qualifying question that
lets the agent answer, or offer to send exact figures. Inventing a number the
agent then has to walk back is far worse than asking.

next_action: what the AGENT should do, as an imperative ("Call and book a
Saturday site visit at Aurum Heights"). Never "follow up" alone.

Use "" for any string you cannot ground, and [] for empty lists."""

SCHEMA: dict[str, Any] = strict_object({
    "summary": {
        "type": "string",
        "description": "Two sentences max. Who they are and what they want.",
    },
    "intent": {
        "type": "string",
        "enum": ["ready_to_buy", "actively_searching", "exploring", "investment",
                 "rental", "price_shopping", "not_serious", "unclear"],
    },
    "intent_detail": {"type": "string", "description": "One short clause of nuance."},
    "key_requirements": {
        "type": "array", "items": {"type": "string"},
        "description": "Short noun phrases: '3BHK', 'east-facing', 'near ORR'.",
    },
    "objections": {
        "type": "array", "items": {"type": "string"},
        "description": "Concerns or blockers they raised. [] if none.",
    },
    "next_action": {"type": "string"},
    "next_action_timing": {
        "type": "string",
        "description": "'today', 'within 24 hours', 'this week'.",
    },
    "suggested_response": {"type": "string"},
    "priority_score": {"type": "integer", "description": "0-100 per the rubric."},
    "temperature": {"type": "string", "enum": ["hot", "warm", "cold"]},
    "urgency": {"type": "string", "enum": ["high", "medium", "low"]},
    "score_reasoning": {"type": "string"},
    "missing_info": {
        "type": "array", "items": {"type": "string"},
        "description": "What the agent should ask next to qualify properly.",
    },
    # These three are judgement calls the model makes; the ceilings they imply
    # are applied in code so the ranking cannot drift prompt-to-prompt.
    "timeline_bucket": {
        "type": "string",
        "enum": ["immediate", "within_1_month", "1_3_months", "3_6_months",
                 "6_12_months", "over_12_months", "unstated"],
        "description": "Normalise whatever they said about timing into one bucket.",
    },
    "has_budget": {
        "type": "boolean",
        "description": "True only if they stated a budget figure or a usable range.",
    },
    "explicit_ask": {
        "type": "boolean",
        "description": "True if they asked for a visit, a call back, or to be contacted.",
    },
})

# A lead cannot outrank these ceilings no matter how well it reads.
TIMELINE_CAP = {
    "immediate": 100, "within_1_month": 100, "1_3_months": 85,
    "3_6_months": 70, "6_12_months": 55, "over_12_months": 40, "unstated": 60,
}
NO_BUDGET_CAP = 65
NO_ASK_CAP = 80


def _apply_caps(a: Analysis, raw: dict[str, Any]) -> Analysis:
    """Clamp the score to what the hard signals justify.

    The model is good at *judging* ("sometime after Diwali" -> 3_6_months) and
    bad at *pricing* that judgement consistently -- left alone it drifts upward
    and marks almost everything hot, which destroys the ranking. So the model
    supplies the buckets and this applies the policy. Deterministic, tunable in
    one place, and explainable to a customer.
    """
    caps: list[tuple[int, str]] = []
    bucket = raw.get("timeline_bucket") or "unstated"
    caps.append((TIMELINE_CAP.get(bucket, 60), f"timeline {bucket.replace('_', ' ')}"))
    if raw.get("has_budget") is False:
        caps.append((NO_BUDGET_CAP, "no budget stated"))
    if raw.get("explicit_ask") is False:
        caps.append((NO_ASK_CAP, "no explicit ask"))

    cap, reason = min(caps, key=lambda c: c[0])
    if a.priority_score > cap:
        a.score_reasoning = (
            f"{a.score_reasoning} Capped at {cap} ({reason})."
        ).strip()
        a.priority_score = cap
    return a


def _reconcile(a: Analysis) -> Analysis:
    """Keep the score, temperature and urgency mutually consistent."""
    if a.priority_score >= 75:
        a.temperature = "hot"
    elif a.priority_score >= 45:
        a.temperature = "warm"
    else:
        a.temperature = "cold"

    if a.intent in {"not_serious", "price_shopping"} and a.priority_score > 60:
        a.priority_score = 60
        a.temperature = "warm"
    if a.temperature == "hot" and a.urgency == "low":
        a.urgency = "medium"
    if a.temperature == "cold" and a.urgency == "high":
        # "act today" drives the agent's queue -- a cold lead never belongs there.
        a.urgency = "medium"
    return a


async def analyze_lead(
    intake: LeadIntake,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> Analysis:
    """Run the analysis. Raises LLMError if every provider fails."""
    payload, engine = await call_json(
        SYSTEM,
        [{"role": "user", "content": f"Lead:\n{intake.as_context()}\n\nAnalyse this lead."}],
        SCHEMA,
        settings,
        client,
    )
    allowed = set(Analysis.model_fields)
    analysis = Analysis.model_validate({k: v for k, v in payload.items() if k in allowed})
    analysis.engine = engine
    return _reconcile(_apply_caps(analysis, payload))
