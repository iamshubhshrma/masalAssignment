"""End-to-end API: intake, ranking, chat grounding, triage, export."""
from __future__ import annotations

import csv
import io

from tests.conftest import ANALYSIS_PAYLOAD, LEAD_BODY, TRIAGE_PAYLOAD


# ---- meta ------------------------------------------------------------------- #
def test_health_reports_ai_config(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["ai_configured"] is True
    assert h["ai_chain"] == ["groq", "gemini"]
    assert h["chat_suggestions"]


def test_ui_is_served_and_uncached(client):
    r = client.get("/")
    assert r.status_code == 200 and "Masal Leads" in r.text
    assert "no-store" in r.headers["cache-control"]
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


# ---- intake ----------------------------------------------------------------- #
def test_create_lead_analyses_immediately(client):
    r = client.post("/api/leads", json=LEAD_BODY)
    assert r.status_code == 201
    lead = r.json()["lead"]
    a = lead["analysis"]
    assert a["priority_score"] == 92 and a["temperature"] == "hot"
    assert a["summary"] and a["next_action"] and a["suggested_response"]
    assert lead["analysis_error"] is None


def test_name_is_required(client):
    assert client.post("/api/leads", json={**LEAD_BODY, "name": "  "}).status_code == 422


def test_only_name_is_required(client):
    """Every other intake field is optional -- real enquiries are incomplete."""
    r = client.post("/api/leads", json={"name": "Walk-in"})
    assert r.status_code == 201


def test_bad_phone_is_rejected(client):
    r = client.post("/api/leads", json={**LEAD_BODY, "phone": "98765"})
    assert r.status_code == 422
    assert "E.164" in str(r.json())


def test_phone_is_normalised(client):
    r = client.post("/api/leads", json={**LEAD_BODY, "phone": "+91 98765-43210"})
    assert r.json()["lead"]["phone"] == "+919876543210"


def test_analysis_failure_is_recorded_not_fatal(client):
    client.llm["fail"] = {"groq", "gemini"}
    r = client.post("/api/leads", json=LEAD_BODY)
    assert r.status_code == 201, "the lead must still be saved"
    lead = r.json()["lead"]
    assert lead["analysis"] is None and lead["analysis_error"]


def test_failed_analysis_can_be_retried(client):
    client.llm["fail"] = {"groq", "gemini"}
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.llm["fail"] = set()
    out = client.post(f"/api/leads/{lead_id}/reanalyze").json()
    assert out["lead"]["analysis"]["priority_score"] == 92


# ---- ranking ---------------------------------------------------------------- #
def test_list_is_ranked_by_priority(client):
    for score, name in [(30, "Cold"), (92, "Hot"), (60, "Warm")]:
        client.llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": score,
                                 "timeline_bucket": "immediate"}
        client.post("/api/leads", json={**LEAD_BODY, "name": name, "phone": ""})

    names = [l["name"] for l in client.get("/api/leads").json()["leads"]]
    assert names == ["Hot", "Warm", "Cold"]


def test_unanalysed_leads_sort_first_so_they_are_not_buried(client):
    client.post("/api/leads", json={**LEAD_BODY, "name": "Scored", "phone": ""})
    client.llm["fail"] = {"groq", "gemini"}
    client.post("/api/leads", json={**LEAD_BODY, "name": "Unscored", "phone": ""})
    assert client.get("/api/leads").json()["leads"][0]["name"] == "Unscored"


def test_stats_summarise_the_pipeline(client):
    for score in (92, 60, 20):
        client.llm["payload"] = {**ANALYSIS_PAYLOAD, "priority_score": score,
                                 "timeline_bucket": "immediate"}
        client.post("/api/leads", json={**LEAD_BODY, "phone": ""})
    s = client.get("/api/leads").json()["stats"]
    assert s == {**s, "total": 3, "analyzed": 3, "hot": 1, "warm": 1, "cold": 1}
    assert s["avg_score"] == 57.3


# ---- chat ------------------------------------------------------------------- #
def test_chat_is_grounded_in_this_lead(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    r = client.post(f"/api/leads/{lead_id}/chat",
                    json={"question": "What should I emphasise?"})
    assert r.status_code == 200

    system = client.llm["calls"][-1]["system"]
    for token in ("Rahul Mehta", "Whitefield", "1.4 crore"):
        assert token in system, "the lead's own details must be in the prompt"
    assert "92/100" in system, "the analysis must be in the prompt"


def test_chat_history_is_persisted_and_replayed(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.post(f"/api/leads/{lead_id}/chat", json={"question": "First question"})
    client.post(f"/api/leads/{lead_id}/chat", json={"question": "Second question"})

    chat = client.get(f"/api/leads/{lead_id}").json()["chat"]
    assert [t["role"] for t in chat] == ["user", "assistant", "user", "assistant"]

    replayed = [m["content"] for m in client.llm["calls"][-1]["messages"]]
    assert "First question" in replayed


def test_chat_on_unknown_lead_is_404(client):
    assert client.post("/api/leads/nope/chat", json={"question": "hi"}).status_code == 404


def test_empty_question_rejected(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    assert client.post(f"/api/leads/{lead_id}/chat", json={"question": ""}).status_code == 422


def test_chat_can_be_cleared(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.post(f"/api/leads/{lead_id}/chat", json={"question": "hi"})
    client.request("DELETE", f"/api/leads/{lead_id}/chat")
    assert client.get(f"/api/leads/{lead_id}").json()["chat"] == []


def test_ai_outage_surfaces_as_503(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    client.llm["fail"] = {"groq", "gemini"}
    r = client.post(f"/api/leads/{lead_id}/chat", json={"question": "hi"})
    assert r.status_code == 503 and "attempts" in r.json()


# ---- bulk triage (own feature) ---------------------------------------------- #
def test_triage_splits_and_scores_each_lead(client):
    client.llm["payload"] = TRIAGE_PAYLOAD
    r = client.post("/api/triage", json={"text": "raw whatsapp dump ..."})
    assert r.status_code == 201

    body = r.json()
    assert body["added"] == 2
    assert {l["name"] for l in body["leads"]} == {"Rahul", "Meera Nair"}
    # first call splits, the rest analyse
    assert len(client.llm["calls"]) == 3


def test_triage_handles_no_enquiries_found(client):
    client.llm["payload"] = {"leads": []}
    body = client.post("/api/triage", json={"text": "OTP 1234"}).json()
    assert body["added"] == 0 and "No enquiries" in body["message"]


def test_triage_keeps_a_lead_whose_phone_is_malformed(client):
    client.llm["payload"] = {"leads": [
        {**TRIAGE_PAYLOAD["leads"][0], "phone": "98765 (mobile)"}]}
    body = client.post("/api/triage", json={"text": "..."}).json()
    assert body["added"] == 1, "a bad phone must not discard the whole lead"
    assert body["leads"][0]["phone"] == ""


def test_triage_falls_back_to_phone_for_a_missing_name(client):
    client.llm["payload"] = {"leads": [
        {**TRIAGE_PAYLOAD["leads"][0], "name": "", "phone": "+919876543210"}]}
    assert client.post("/api/triage", json={"text": "..."}).json()["leads"][0]["name"] \
        == "+919876543210"


def test_triage_rejects_empty_input(client):
    assert client.post("/api/triage", json={"text": ""}).status_code == 422


# ---- lifecycle -------------------------------------------------------------- #
def test_delete_and_clear(client):
    lead_id = client.post("/api/leads", json=LEAD_BODY).json()["lead"]["id"]
    assert client.delete(f"/api/leads/{lead_id}").status_code == 200
    assert client.delete(f"/api/leads/{lead_id}").status_code == 404

    client.post("/api/leads", json={**LEAD_BODY, "phone": ""})
    assert client.request("DELETE", "/api/leads").json()["cleared"] == 1


def test_reanalyze_all_fills_only_the_gaps(client):
    client.llm["fail"] = {"groq", "gemini"}
    client.post("/api/leads", json={**LEAD_BODY, "phone": ""})
    client.llm["fail"] = set()
    client.post("/api/leads", json={**LEAD_BODY, "name": "Fine", "phone": ""})

    out = client.post("/api/reanalyze", json={}).json()
    assert out["reanalyzed"] == 1
    assert all(l["analysis"] for l in out["leads"])


# ---- export ----------------------------------------------------------------- #
def test_csv_export_carries_the_analysis(client):
    client.post("/api/leads", json=LEAD_BODY)
    r = client.get("/api/leads.csv")
    assert "text/csv" in r.headers["content-type"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert rows[0]["name"] == "Rahul Mehta"
    assert rows[0]["priority_score"] == "92"
    assert rows[0]["key_requirements"] == "3BHK; east-facing"


def test_csv_is_valid_when_empty(client):
    assert list(csv.DictReader(io.StringIO(client.get("/api/leads.csv").text))) == []
