"""Voice follow-up (own feature) plus store and transcript-shape handling."""
from __future__ import annotations

import httpx
import pytest

from app.models import Lead, LeadIntake, parse_turns
from app.ringg import MockRinggClient, RinggClient, RinggError
from app.store import LeadStore
from app.transcript import resolve_turns
from tests.conftest import ANALYSIS_PAYLOAD, LEAD_BODY


# ---- store ------------------------------------------------------------------ #
@pytest.mark.asyncio
async def test_store_persists_and_reloads(tmp_path):
    path = tmp_path / "leads.json"
    s1 = LeadStore(path)
    await s1.add(LeadIntake(name="Rahul", budget="1.4cr"))
    s2 = LeadStore(path)
    s2.load()
    assert [l.name for l in s2.all()] == ["Rahul"]


@pytest.mark.asyncio
async def test_store_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "leads.json"
    path.write_text("{ not json")
    s = LeadStore(path)
    s.load()
    assert s.all() == []


# ---- Ringg contract (kept from the voice build) ----------------------------- #
def test_call_payload_sends_exactly_one_caller_option(settings_obj, monkeypatch):
    monkeypatch.setattr(settings_obj, "ringg_api_key", "k")
    monkeypatch.setattr(settings_obj, "ringg_agent_id", "a")
    monkeypatch.setattr(settings_obj, "ringg_from_number_id", "n1")
    monkeypatch.setattr(settings_obj, "ringg_from_number", "")
    p = RinggClient(settings_obj).build_call_payload("Rahul", "+919876543210")
    assert ("from_number_id" in p) ^ ("from_number" in p)
    for f in ("name", "mobile_number", "agent_id"):
        assert f in p


@pytest.mark.asyncio
async def test_api_key_header_and_401(settings_obj, monkeypatch):
    monkeypatch.setattr(settings_obj, "ringg_api_key", "k")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("X-API-KEY")
        return httpx.Response(401, json={"detail": "nope"})

    async with httpx.AsyncClient(base_url="https://x.test",
                                 transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(RinggError) as ei:
            await RinggClient(settings_obj, http).workspace()
    assert seen["key"] == "k" and ei.value.status == 401


# ---- transcript shapes ------------------------------------------------------ #
def test_parse_turns_handles_every_shape():
    assert [t.speaker for t in parse_turns([{"bot": "hi"}, {"user": "yes"}])] == ["bot", "user"]
    assert [t.speaker for t in parse_turns([{"role": "assistant", "content": "hi"},
                                            {"role": "user", "content": "yes"}])] == ["bot", "user"]
    assert [t.speaker for t in parse_turns("bot: hi\nuser: yes")] == ["bot", "user"]
    # an unrecognised role must never be counted as prospect speech
    assert parse_turns([{"role": "narrator", "content": "x"}])[0].speaker == "bot"


@pytest.mark.asyncio
async def test_transcription_url_can_be_an_actual_url():
    """Regression: the field name is literal -- it is sometimes a URL to fetch."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=[{"bot": "hi"}, {"user": "interested"}])
    )) as http:
        turns, note = await resolve_turns(
            {"transcription_url": "https://cdn.test/t.json"}, http)
    assert note is None and [t.speaker for t in turns] == ["bot", "user"]


@pytest.mark.asyncio
async def test_unrecognised_payload_reports_its_keys():
    turns, note = await resolve_turns({"call_status": "completed", "odd_field": "x"})
    assert turns == [] and "odd_field" in note


# ---- voice flow through the API --------------------------------------------- #
def test_call_requires_a_phone_number(client):
    lead_id = client.post("/api/leads",
                          json={**LEAD_BODY, "phone": ""}).json()["lead"]["id"]
    r = client.post(f"/api/leads/{lead_id}/call")
    assert r.status_code == 400 and "phone" in r.json()["detail"]


def test_call_is_not_placed_twice(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    assert client.post(f"/api/leads/{lead_id}/call").status_code == 200
    assert client.post(f"/api/leads/{lead_id}/call").status_code == 400


def test_call_outcome_is_merged_back_and_lead_rescored(client):
    """The transcript is new evidence, so the lead is analysed again with it."""
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.post(f"/api/leads/{lead_id}/call")

    for _ in range(6):
        out = client.post("/api/calls/refresh").json()
        if out["still_pending"] == 0:
            break

    lead = client.get(f"/api/leads/{lead_id}").json()
    v = lead["voice"]
    assert v["status"] in {"completed", "failed"}
    assert v["outcome"] is not None
    if v["transcript"]:
        assert v["outcome"]["reached"] is True
        assert "[Voice call transcript]" in lead["message"], "transcript folded into the lead"


def test_refresh_is_safe_with_no_calls(client):
    assert client.post("/api/calls/refresh").json()["refreshed"] == 0


def test_force_sync_recovers_a_call_finalised_without_a_transcript(client, monkeypatch):
    """Ringg reports `completed` before attaching the transcript."""
    from app import main
    from app.config import settings

    monkeypatch.setattr(settings, "transcript_grace_polls", 0)
    monkeypatch.setattr(main, "ringg", MockRinggClient(settings, seed=1, transcript_lag=1))

    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.post(f"/api/leads/{lead_id}/call")
    for _ in range(3):
        client.post("/api/calls/refresh")

    out = client.post(f"/api/leads/{lead_id}/call/sync").json()
    assert out["voice"]["transcript"], "force sync must re-fetch from Ringg"


def test_voice_disabled_when_unconfigured(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "mock_mode", False)
    monkeypatch.setattr(settings, "ringg_api_key", "")
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    assert client.post(f"/api/leads/{lead_id}/call").status_code == 400
    assert client.get("/api/health").json()["voice_enabled"] is False
