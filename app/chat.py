"""Per-lead conversational assistant.

Grounded in one lead: the intake fields, the AI analysis, and any voice-call
transcript. It is deliberately not a general chatbot -- if the answer is not in
this lead's context it says so instead of inventing property details.
"""
from __future__ import annotations

import logging

import httpx

from .config import Settings
from .llm import call_text
from .models import ChatTurn, Lead

log = logging.getLogger("masal.chat")

MAX_HISTORY = 12  # turns replayed to the model

SYSTEM = """You are a sales coach sitting beside a real-estate agent, helping with ONE specific lead.

The lead's details and the AI analysis of them are below. Answer only from that
context plus general real-estate sales craft.

Rules:
- Be concrete and short. The agent is mid-shift, often about to dial. Two or
  three sentences, or a tight bulleted list. No preamble, no recap of the brief.
- Never invent facts about this customer, your inventory, prices, availability
  or the market. If the lead does not say it, you do not know it. Say what is
  missing and how to ask for it.
- If asked to rewrite or retone a message, return the finished message ready to
  send, nothing else around it.
- If the question is unrelated to this lead or to selling to them, say so in one
  line and steer back.
- Talk like a colleague, not a support bot. No "As an AI".

LEAD
{context}

AI ANALYSIS
{analysis}
{voice}"""


def _analysis_block(lead: Lead) -> str:
    a = lead.analysis
    if a is None:
        return "Not analysed yet."
    parts = [
        f"Priority: {a.priority_score}/100 ({a.temperature}, {a.urgency} urgency)",
        f"Intent: {a.intent_label}. {a.intent_detail}".strip(),
        f"Summary: {a.summary}",
        f"Key requirements: {', '.join(a.key_requirements) or 'none captured'}",
        f"Objections: {', '.join(a.objections) or 'none raised'}",
        f"Recommended next action: {a.next_action} ({a.next_action_timing})",
        f"Currently suggested reply: {a.suggested_response}",
    ]
    if a.missing_info:
        parts.append(f"Still unknown: {', '.join(a.missing_info)}")
    return "\n".join(parts)


def _voice_block(lead: Lead) -> str:
    v = lead.voice
    if not v.transcript:
        return ""
    return (
        "\n\nVOICE CALL TRANSCRIPT (an AI agent already phoned this lead)\n"
        + v.transcript_text()[:4000]
    )


def build_system(lead: Lead) -> str:
    return SYSTEM.format(
        context=lead.as_context(),
        analysis=_analysis_block(lead),
        voice=_voice_block(lead),
    )


async def ask(
    lead: Lead,
    question: str,
    settings: Settings,
    client: httpx.AsyncClient | None = None,
) -> ChatTurn:
    """Answer a follow-up question about this lead. Raises LLMError on failure."""
    history = [{"role": t.role, "content": t.content} for t in lead.chat[-MAX_HISTORY:]]
    history.append({"role": "user", "content": question})

    text, engine = await call_text(build_system(lead), history, settings, client, max_tokens=700)
    return ChatTurn(role="assistant", content=text.strip(), engine=engine)


SUGGESTIONS = [
    "What should I emphasise on the call?",
    "Make the suggested reply more assertive",
    "What are the risks with this lead?",
    "Draft a WhatsApp follow-up",
    "What should I ask to qualify them?",
]
