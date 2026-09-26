"""Runtime configuration, loaded from the environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


@dataclass
class Settings:
    # Ringg
    ringg_api_key: str = field(default_factory=lambda: os.getenv("RINGG_API_KEY", "").strip())
    ringg_base_url: str = field(
        default_factory=lambda: (
            os.getenv("RINGG_BASE_URL", "").strip() or "https://prod-api.ringg.ai/ca/api/v0"
        ).rstrip("/")
    )
    ringg_agent_id: str = field(default_factory=lambda: os.getenv("RINGG_AGENT_ID", "").strip())
    ringg_from_number_id: str = field(
        default_factory=lambda: os.getenv("RINGG_FROM_NUMBER_ID", "").strip()
    )
    ringg_from_number: str = field(
        default_factory=lambda: os.getenv("RINGG_FROM_NUMBER", "").strip()
    )

    # App
    mock_mode: bool = field(default_factory=lambda: _bool("MOCK_MODE", True))
    max_call_length: int = field(default_factory=lambda: _int("MAX_CALL_LENGTH", 240))
    # Ringg flips call_status to `completed` before post-call processing finishes,
    # so the transcript can arrive seconds later. Keep re-polling a finished call
    # this many times before giving up and recording "no conversation".
    transcript_grace_polls: int = field(
        default_factory=lambda: _int("TRANSCRIPT_GRACE_POLLS", 12)
    )

    # LLM
    # LLM_ENGINE is the current name; EXTRACTOR_ENGINE is kept for older .env files.
    llm_engine: str = field(
        default_factory=lambda: (
            os.getenv("LLM_ENGINE", "").strip().lower()
            or os.getenv("EXTRACTOR_ENGINE", "").strip().lower()
            or "auto"
        )
    )
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", "").strip())
    groq_model: str = field(
        default_factory=lambda: os.getenv("GROQ_MODEL", "").strip() or "openai/gpt-oss-120b"
    )
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", "").strip())
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "").strip() or "gemini-2.5-flash"
    )

    data_file: Path = field(default_factory=lambda: BASE_DIR / "data" / "leads.json")

    # ---- derived -------------------------------------------------------------
    @property
    def caller_field(self) -> tuple[str, str] | None:
        """Ringg needs exactly one of from_number_id / from_number."""
        if self.ringg_from_number_id:
            return ("from_number_id", self.ringg_from_number_id)
        if self.ringg_from_number:
            return ("from_number", self.ringg_from_number)
        return None

    def ringg_ready(self) -> tuple[bool, list[str]]:
        """Can we place a real call? Returns (ok, missing-field names)."""
        missing = []
        if not self.ringg_api_key:
            missing.append("RINGG_API_KEY")
        if not self.ringg_agent_id:
            missing.append("RINGG_AGENT_ID")
        if self.caller_field is None:
            missing.append("RINGG_FROM_NUMBER_ID or RINGG_FROM_NUMBER")
        return (not missing, missing)

    def engine_chain(self) -> list[str]:
        """LLM providers to try, in order.

        A named preference is honoured but never exclusive: free tiers rate-limit
        hard (Groq is 8k tokens/min), so any other configured provider stays in
        the chain as a fallback rather than letting one 429 fail the request.
        """
        available: list[str] = []
        if self.groq_api_key:
            available.append("groq")
        if self.google_api_key:
            available.append("gemini")

        pref = self.llm_engine
        if pref in available:
            return [pref] + [e for e in available if e != pref]
        return available

    def providers_configured(self) -> bool:
        return bool(self.engine_chain())

    @property
    def voice_enabled(self) -> bool:
        """Voice calling is a bonus feature -- off unless fully configured."""
        if self.mock_mode:
            return True
        return self.ringg_ready()[0]


settings = Settings()


def reload_settings() -> Settings:
    """Re-read the environment (used by tests)."""
    global settings
    settings = Settings()
    return settings
