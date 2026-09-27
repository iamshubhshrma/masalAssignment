"""Masal Leads -- AI lead triage for real-estate salespeople.

Flow:
  POST /api/leads           intake form  -> analysed immediately
  POST /api/triage          raw blob     -> split into leads, each analysed
  GET  /api/leads           ranked list, highest priority first
  POST /api/leads/{id}/chat grounded follow-up questions about that lead
  POST /api/leads/{id}/call optional: AI voice confirmation call (Ringg)
"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .analyze import analyze_lead
from .chat import SUGGESTIONS, ask
from .config import BASE_DIR, settings
from .extract import extract_lead
from .llm import LLMError
from .models import (
    CallOutcome,
    ChatRequest,
    ChatTurn,
    Lead,
    LeadIntake,
    ReanalyzeRequest,
    TriageRequest,
)
from .ringg import MockRinggClient, RinggClient, RinggError, make_client
from .store import LeadStore
from .transcript import resolve_turns

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("masal")

UI_BUILD = "2026-09-27.1"

store = LeadStore(settings.data_file)
ringg: RinggClient | MockRinggClient = make_client(settings)
_http: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    store.load()
    _http = httpx.AsyncClient(timeout=httpx.Timeout(90.0))
    log.info("masal up | AI chain=%s | voice=%s",
             settings.engine_chain() or "NONE CONFIGURED",
             "mock" if settings.mock_mode else "live")
    yield
    if _http is not None:
        await _http.aclose()
    await ringg.aclose()


app = FastAPI(title="Masal Leads", version=UI_BUILD, lifespan=lifespan)


@app.middleware("http")
async def _no_cache_ui(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith(("/static", "/api")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


@app.exception_handler(LLMError)
async def _llm_error(request: Request, exc: LLMError) -> JSONResponse:
    return JSONResponse(status_code=503,
                        content={"error": str(exc), "attempts": exc.attempts})


@app.exception_handler(RinggError)
async def _ringg_error(request: Request, exc: RinggError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
async def _analyse_into(lead: Lead) -> Lead:
    """Analyse one lead, recording the failure on the lead rather than raising."""
    try:
        lead.analysis = await analyze_lead(lead, settings, _http)
        lead.analysis_error = None
    except LLMError as exc:
        lead.analysis_error = str(exc)
        log.warning("analysis failed for %s: %s", lead.id, exc)
    return await store.update(lead)


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/api/health")
async def health() -> dict[str, Any]:
    chain = settings.engine_chain()
    return {
        "ok": True,
        "ui_build": UI_BUILD,
        "ai_configured": bool(chain),
        "ai_chain": chain,
        "primary_model": (settings.groq_model if chain and chain[0] == "groq"
                          else settings.gemini_model if chain else None),
        "voice_enabled": settings.voice_enabled,
        "voice_mock": settings.mock_mode,
        "chat_suggestions": SUGGESTIONS,
        "stats": store.stats(),
    }


# --------------------------------------------------------------------------- #
# Leads
# --------------------------------------------------------------------------- #
@app.get("/api/leads")
async def list_leads() -> dict[str, Any]:
    return {"leads": [l.model_dump(mode="json") for l in store.all()],
            "stats": store.stats()}


@app.post("/api/leads", status_code=201)
async def create_lead(intake: LeadIntake = Body(...)) -> dict[str, Any]:
    """Save a lead from the intake form and analyse it straight away."""
    lead = await store.add(intake)
    lead = await _analyse_into(lead)
    return {"lead": lead.model_dump(mode="json"), "stats": store.stats()}


@app.get("/api/leads/{lead_id}")
async def get_lead(lead_id: str) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    return lead.model_dump(mode="json")


@app.delete("/api/leads/{lead_id}")
async def delete_lead(lead_id: str) -> dict[str, Any]:
    if not await store.delete(lead_id):
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    return {"deleted": lead_id, "stats": store.stats()}


@app.post("/api/leads/{lead_id}/reanalyze")
async def reanalyze_one(lead_id: str) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    lead = await _analyse_into(lead)
    return {"lead": lead.model_dump(mode="json"), "stats": store.stats()}


@app.post("/api/reanalyze")
async def reanalyze_many(req: ReanalyzeRequest = Body(default=ReanalyzeRequest())) -> dict[str, Any]:
    """Re-score leads -- used after a prompt change, or to fill in failures."""
    targets = ([l for lid in req.lead_ids if (l := store.get(lid))] if req.lead_ids
               else [l for l in store.all() if l.analysis is None])
    if targets:
        await asyncio.gather(*(_analyse_into(l) for l in targets), return_exceptions=True)
    return {"reanalyzed": len(targets),
            "leads": [l.model_dump(mode="json") for l in store.all()],
            "stats": store.stats()}


@app.delete("/api/leads")
async def clear_leads() -> dict[str, Any]:
    return {"cleared": await store.clear(), "stats": store.stats()}


# --------------------------------------------------------------------------- #
# Bulk triage (own feature)
# --------------------------------------------------------------------------- #
@app.post("/api/triage", status_code=201)
async def triage(req: TriageRequest = Body(...)) -> dict[str, Any]:
    """Split a raw paste into leads, then analyse each one concurrently."""
    from .triage import triage_blob

    intakes = await triage_blob(req.text, settings, _http)
    if not intakes:
        return {"added": 0, "leads": [l.model_dump(mode="json") for l in store.all()],
                "stats": store.stats(),
                "message": "No enquiries found in that text."}

    created = await store.add_many(intakes)
    await asyncio.gather(*(_analyse_into(l) for l in created), return_exceptions=True)
    return {"added": len(created),
            "leads": [l.model_dump(mode="json") for l in store.all()],
            "stats": store.stats()}


# --------------------------------------------------------------------------- #
# Conversational interface
# --------------------------------------------------------------------------- #
@app.post("/api/leads/{lead_id}/chat")
async def chat(lead_id: str, req: ChatRequest = Body(...)) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")

    answer = await ask(lead, req.question, settings, _http)
    lead = await store.append_chat(
        lead, ChatTurn(role="user", content=req.question), answer
    )
    return {"answer": answer.model_dump(mode="json"),
            "chat": [t.model_dump(mode="json") for t in lead.chat]}


@app.delete("/api/leads/{lead_id}/chat")
async def clear_chat(lead_id: str) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    lead.chat = []
    await store.update(lead)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Voice confirmation call (own feature, optional)
# --------------------------------------------------------------------------- #
@app.post("/api/leads/{lead_id}/call")
async def start_call(lead_id: str) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    if not settings.voice_enabled:
        raise HTTPException(400, "Voice calling is not configured on this deployment.")
    if not lead.phone:
        raise HTTPException(400, "This lead has no phone number.")
    if lead.voice.call_id:
        raise HTTPException(400, "A call has already been placed for this lead.")

    data = await ringg.initiate_call(lead.name, lead.phone, {
        "callee_name": lead.name,
        "project_name": lead.property_requirement or "",
        "lead_source": lead.source or "",
        "enquiry_notes": (lead.message or "")[:400],
    })
    lead.voice.call_id = data["call_id"]
    lead.voice.status = data.get("call_status") or "registered"
    lead.voice.initiated_at = data.get("initiated_at")
    lead.voice.error = None
    lead = await store.update(lead)
    return lead.model_dump(mode="json")


async def _sync_call(lead: Lead, force: bool = False) -> Lead:
    v = lead.voice
    if not v.call_id:
        return lead
    try:
        details = await ringg.call_details(v.call_id, send_analysis=True)
    except RinggError as exc:
        v.error = str(exc)
        return await store.update(lead)

    v.status = details.get("call_status") or v.status
    v.sub_status = details.get("call_sub_status") or v.sub_status
    v.recording_url = details.get("recording_url") or v.recording_url
    if (d := details.get("call_duration")) is not None:
        v.duration = d

    turns, note = await resolve_turns(details, _http)
    if turns:
        v.transcript = turns
        v.error = None
    elif v.is_terminal and v.expects_transcript and note:
        v.error = note
    lead = await store.update(lead)

    if v.is_terminal and v.outcome is None:
        exhausted = v.polls >= settings.transcript_grace_polls
        if v.transcript or force or exhausted or not v.expects_transcript:
            lead = await _merge_call_outcome(lead)
        else:
            v.polls += 1
            lead = await store.update(lead)
    return lead


async def _merge_call_outcome(lead: Lead) -> Lead:
    """Fold what the call learned back into the lead, then re-score it."""
    v = lead.voice
    if not v.transcript:
        v.outcome = CallOutcome(
            reached=False,
            notes=f"No conversation captured ({v.status}"
                  f"{', ' + v.sub_status if v.sub_status else ''}).",
        )
        return await store.update(lead)

    confirmed = await extract_lead(v.transcript_text(), settings,
                                   {"prospect name": lead.name}, _http)
    v.outcome = CallOutcome(
        reached=True, confirmed=confirmed.confirmed, do_not_call=confirmed.do_not_call,
        budget=confirmed.budget, timeline=confirmed.timeline,
        site_visit=confirmed.site_visit, site_visit_time=confirmed.site_visit_time,
        notes=confirmed.summary, engine=confirmed.engine,
    )
    # The call is new evidence -- append it and re-run the analysis on the richer lead.
    lead.message = (
        f"{lead.message}\n\n[Voice call transcript]\n{v.transcript_text()}"
    ).strip()
    if confirmed.budget and not lead.budget:
        lead.budget = confirmed.budget
    if confirmed.timeline and not lead.timeline:
        lead.timeline = confirmed.timeline
    lead = await store.update(lead)
    return await _analyse_into(lead)


@app.post("/api/calls/refresh")
async def refresh_calls() -> dict[str, Any]:
    targets = [l for l in store.all()
               if l.voice.call_id and (l.voice.is_pending or l.voice.awaiting_transcript)]
    if targets:
        await asyncio.gather(*(_sync_call(l) for l in targets), return_exceptions=True)
    return {"refreshed": len(targets),
            "still_pending": len([l for l in store.all()
                                  if l.voice.is_pending or l.voice.awaiting_transcript]),
            "leads": [l.model_dump(mode="json") for l in store.all()],
            "stats": store.stats()}


@app.post("/api/leads/{lead_id}/call/sync")
async def force_sync(lead_id: str) -> dict[str, Any]:
    lead = store.get(lead_id)
    if lead is None:
        raise HTTPException(404, "This lead no longer exists (the server may have restarted).")
    lead.voice.outcome = None
    lead.voice.polls = 0
    await store.update(lead)
    lead = await _sync_call(lead, force=True)
    return lead.model_dump(mode="json")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
@app.get("/api/leads.csv")
async def leads_csv() -> Response:
    cols = ["name", "phone", "location", "property_requirement", "budget", "timeline",
            "source", "priority_score", "temperature", "urgency", "intent",
            "summary", "key_requirements", "objections", "next_action",
            "next_action_timing", "suggested_response", "score_reasoning"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for l in store.all():
        row = l.model_dump(mode="json")
        a = l.analysis
        if a:
            row.update(a.model_dump(mode="json"))
            row["key_requirements"] = "; ".join(a.key_requirements)
            row["objections"] = "; ".join(a.objections)
        w.writerow(row)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="leads.csv"'})


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
STATIC_DIR = BASE_DIR / "static"


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
