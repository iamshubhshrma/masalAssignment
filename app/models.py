"""Schemas: lead intake, AI analysis, chat, and the optional voice follow-up."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Temperature = Literal["hot", "warm", "cold"]
Urgency = Literal["high", "medium", "low"]
Intent = Literal[
    "ready_to_buy",
    "actively_searching",
    "exploring",
    "investment",
    "rental",
    "price_shopping",
    "not_serious",
    "unclear",
]

INTENT_LABELS: dict[str, str] = {
    "ready_to_buy": "Ready to buy",
    "actively_searching": "Actively searching",
    "exploring": "Exploring options",
    "investment": "Investment buyer",
    "rental": "Looking to rent",
    "price_shopping": "Price shopping",
    "not_serious": "Not serious",
    "unclear": "Unclear",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Intake
# --------------------------------------------------------------------------- #
class LeadIntake(BaseModel):
    """What the salesperson types in (or what bulk triage extracts)."""

    name: str = Field(min_length=1, max_length=120)
    location: str = Field(default="", max_length=160)
    property_requirement: str = Field(default="", max_length=300)
    budget: str = Field(default="", max_length=120)
    timeline: str = Field(default="", max_length=120)
    message: str = Field(default="", max_length=6000)
    phone: str = Field(default="", max_length=20)
    source: str = Field(default="", max_length=80)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("name cannot be blank")
        return v

    @field_validator("phone")
    @classmethod
    def _clean_phone(cls, v: str) -> str:
        v = re.sub(r"[\s\-()]", "", (v or "").strip())
        if v and not re.match(r"^\+[1-9]\d{7,14}$", v):
            raise ValueError(f"'{v}' is not valid E.164 (e.g. +919876543210)")
        return v

    def as_context(self) -> str:
        """Render the lead as the block every prompt sees. One definition, reused."""
        rows = [
            ("Name", self.name),
            ("Location", self.location),
            ("Property requirement", self.property_requirement),
            ("Budget", self.budget),
            ("Buying timeline", self.timeline),
            ("Source", self.source),
        ]
        lines = [f"{k}: {v}" for k, v in rows if v]
        if self.message:
            lines.append(f"Customer's own message:\n\"\"\"\n{self.message.strip()}\n\"\"\"")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# AI analysis
# --------------------------------------------------------------------------- #
class Analysis(BaseModel):
    """The six required AI outputs, plus the scoring that drives prioritisation."""

    summary: str = ""
    intent: Intent = "unclear"
    intent_detail: str = ""
    key_requirements: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    next_action: str = ""
    next_action_timing: str = ""
    suggested_response: str = ""

    priority_score: int = 0
    temperature: Temperature = "cold"
    urgency: Urgency = "low"
    score_reasoning: str = ""
    missing_info: list[str] = Field(default_factory=list)

    engine: str = ""
    analyzed_at: str = Field(default_factory=utcnow)

    @field_validator("priority_score")
    @classmethod
    def _clamp(cls, v: int) -> int:
        return max(0, min(100, int(v)))

    @field_validator("key_requirements", "objections", "missing_info", mode="before")
    @classmethod
    def _clean_list(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        return [str(i).strip() for i in v if str(i).strip()][:12]

    @property
    def intent_label(self) -> str:
        return INTENT_LABELS.get(self.intent, self.intent)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    at: str = Field(default_factory=utcnow)
    engine: str = ""


# --------------------------------------------------------------------------- #
# Voice follow-up (optional feature)
# --------------------------------------------------------------------------- #
TERMINAL_STATUSES = {"completed", "failed", "error", "cancelled", "forwarded"}
PENDING_STATUSES = {"registered", "ongoing", "retry"}

NO_CONVERSATION_SUB_STATUSES = {
    "VOICEMAIL_DETECTED", "VOICEMAIL_DETECTED_VIA_LLM", "USER_DID_NOT_JOIN",
    "NOT_ABLE_TO_CALL", "DND_SKIPPED", "RATE_LIMITED", "RETRY_LIMIT_REACHED",
    "PRE_CALL_API_FAILED", "CAMPAIGN_TIME_EXCEEDED", "FAILED",
}


class Turn(BaseModel):
    speaker: Literal["bot", "user"]
    text: str


class ConfirmedLead(BaseModel):
    """What the transcript extractor reads out of a voice call.

    Kept separate from Analysis: this is grounded only in what was *said on the
    phone*, before it is merged into the lead and re-analysed.
    """

    confirmed: bool = False
    interest_level: Literal["hot", "warm", "cold", "not_interested", "unknown"] = "unknown"
    property_type: str | None = None
    configuration: str | None = None
    budget: str | None = None
    preferred_location: str | None = None
    timeline: str | None = None
    financing: str | None = None
    site_visit: bool = False
    site_visit_time: str | None = None
    callback_requested: bool = False
    callback_time: str | None = None
    do_not_call: bool = False
    objections: list[str] = Field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    engine: str = "none"

    @field_validator("confidence")
    @classmethod
    def _clamp_conf(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class CallOutcome(BaseModel):
    """What the voice agent learned, merged back into the lead."""

    reached: bool = False
    confirmed: bool = False
    do_not_call: bool = False
    budget: str | None = None
    timeline: str | None = None
    site_visit: bool = False
    site_visit_time: str | None = None
    notes: str = ""
    engine: str = ""


class VoiceCall(BaseModel):
    call_id: str | None = None
    status: str = "not_called"
    sub_status: str | None = None
    duration: float | None = None
    recording_url: str | None = None
    transcript: list[Turn] = Field(default_factory=list)
    initiated_at: str | None = None
    polls: int = 0
    error: str | None = None
    outcome: CallOutcome | None = None

    @property
    def is_pending(self) -> bool:
        return self.call_id is not None and self.status in PENDING_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def expects_transcript(self) -> bool:
        if self.status != "completed":
            return False
        return (self.sub_status or "").upper() not in NO_CONVERSATION_SUB_STATUSES

    @property
    def awaiting_transcript(self) -> bool:
        return self.call_id is not None and self.is_terminal and self.outcome is None

    def transcript_text(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.transcript)


# --------------------------------------------------------------------------- #
# The stored record
# --------------------------------------------------------------------------- #
class Lead(LeadIntake):
    id: str
    analysis: Analysis | None = None
    analysis_error: str | None = None
    chat: list[ChatTurn] = Field(default_factory=list)
    voice: VoiceCall = Field(default_factory=VoiceCall)
    created_at: str = Field(default_factory=utcnow)
    updated_at: str = Field(default_factory=utcnow)

    @property
    def score(self) -> int:
        return self.analysis.priority_score if self.analysis else -1


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class TriageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class ReanalyzeRequest(BaseModel):
    lead_ids: list[str] | None = None


# --------------------------------------------------------------------------- #
# Transcript normalisation (voice)
# --------------------------------------------------------------------------- #
_BOT_WORDS = {"bot", "assistant", "agent", "ai", "system", "caller", "outbound"}
_USER_WORDS = {"user", "human", "customer", "callee", "client", "prospect", "lead"}


def _normalise_speaker(role: str) -> str:
    r = role.strip().lower()
    if r in _USER_WORDS:
        return "user"
    return "bot"  # unknown roles never count as prospect speech


def _turn_from_line(line: str) -> Turn | None:
    line = line.strip()
    if not line or ":" not in line:
        return None
    speaker, _, text = line.partition(":")
    speaker, text = speaker.strip().lower(), text.strip()
    if not text or len(speaker) > 20 or speaker not in _BOT_WORDS | _USER_WORDS:
        return None
    return Turn(speaker=_normalise_speaker(speaker), text=text)


def parse_turns(raw: Any) -> list[Turn]:
    """Normalise any transcript shape Ringg returns into Turn objects."""
    turns: list[Turn] = []

    if isinstance(raw, str):
        return [t for line in raw.splitlines() if (t := _turn_from_line(line))]
    if not isinstance(raw, list):
        return turns

    for item in raw:
        if isinstance(item, str):
            if (t := _turn_from_line(item)):
                turns.append(t)
            continue
        if not isinstance(item, dict):
            continue

        matched = False
        for speaker in ("bot", "user"):
            text = item.get(speaker)
            if isinstance(text, str) and text.strip():
                turns.append(Turn(speaker=speaker, text=text.strip()))
                matched = True
        if matched:
            continue

        role = item.get("role") or item.get("speaker") or item.get("from") or item.get("source")
        text = (item.get("content") or item.get("text")
                or item.get("message") or item.get("transcript"))
        if isinstance(text, str) and text.strip() and isinstance(role, str):
            turns.append(Turn(speaker=_normalise_speaker(role), text=text.strip()))
    return turns
