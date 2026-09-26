"""Fixtures. Tests never call a real AI provider -- `fake_llm` intercepts them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def settings_obj(monkeypatch, tmp_path: Path):
    from app.config import Settings

    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("GOOGLE_API_KEY", "goog_test")
    monkeypatch.setenv("LLM_ENGINE", "auto")
    monkeypatch.setenv("MOCK_MODE", "true")
    s = Settings()
    s.data_file = tmp_path / "leads.json"
    return s


ANALYSIS_PAYLOAD = {
    "summary": "Rahul wants a 3BHK in Whitefield and can visit Saturday.",
    "intent": "ready_to_buy",
    "intent_detail": "loan pre-approved",
    "key_requirements": ["3BHK", "east-facing"],
    "objections": [],
    "next_action": "Call and confirm the Saturday viewing",
    "next_action_timing": "today",
    "suggested_response": "Hi Rahul, I can arrange a Saturday viewing. What time suits?",
    "priority_score": 92,
    "temperature": "hot",
    "urgency": "high",
    "score_reasoning": "Specific budget, immediate visit request.",
    "missing_info": ["preferred floor"],
    "timeline_bucket": "within_1_month",
    "has_budget": True,
    "explicit_ask": True,
}

TRIAGE_PAYLOAD = {
    "leads": [
        {"name": "Rahul", "location": "Whitefield", "property_requirement": "3BHK",
         "budget": "1.4 cr", "timeline": "Saturday", "message": "Looking for a 3BHK.",
         "phone": "+919876543210", "source": "WhatsApp"},
        {"name": "Meera Nair", "location": "Sarjapur", "property_requirement": "2BHK",
         "budget": "80 lakhs", "timeline": "next year", "message": "not urgent",
         "phone": "", "source": "WhatsApp"},
    ]
}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace both providers with deterministic stubs.

    Returns a dict the test can mutate: `payload` for JSON calls, `text` for
    chat, `fail` to make a named provider raise, and `calls` recording usage.
    """
    import app.llm as llm

    ctl = {"payload": dict(ANALYSIS_PAYLOAD), "text": "Emphasise the Saturday slot.",
           "fail": set(), "calls": []}

    def make(name):
        async def provider(system, messages, settings, http, schema, max_tokens):
            ctl["calls"].append({"provider": name, "system": system,
                                 "messages": messages, "schema": schema})
            if name in ctl["fail"]:
                raise llm.LLMError(f"{name} forced failure")
            return json.dumps(ctl["payload"]) if schema else ctl["text"]
        return provider

    monkeypatch.setitem(llm._PROVIDERS, "groq", make("groq"))
    monkeypatch.setitem(llm._PROVIDERS, "gemini", make("gemini"))
    return ctl


@pytest.fixture
def client(monkeypatch, tmp_path, fake_llm):
    from fastapi.testclient import TestClient

    from app import main
    from app.config import settings
    from app.ringg import MockRinggClient
    from app.store import LeadStore

    monkeypatch.setattr(settings, "groq_api_key", "gsk_test")
    monkeypatch.setattr(settings, "google_api_key", "goog_test")
    monkeypatch.setattr(settings, "llm_engine", "auto")
    monkeypatch.setattr(settings, "mock_mode", True)
    monkeypatch.setattr(main, "store", LeadStore(tmp_path / "leads.json"))
    monkeypatch.setattr(main, "ringg", MockRinggClient(settings, seed=1))

    with TestClient(main.app) as c:
        c.llm = fake_llm
        yield c


LEAD_BODY = {
    "name": "Rahul Mehta", "location": "Whitefield", "property_requirement": "3BHK",
    "budget": "1.4 crore", "timeline": "2 months", "phone": "+919876543210",
    "source": "Website", "message": "Pre-approved, want to visit Saturday.",
}
