"""Ringg AI REST client.

Endpoints used (base https://prod-api.ringg.ai/ca/api/v0, header X-API-KEY):
  GET  /workspace                     validate key, read credits
  GET  /agent/all                     list assistants
  GET  /workspace/numbers             list caller numbers
  POST /calling/outbound/individual   place one call
  GET  /calling/call-details          status + transcript (+ analysis)
"""
from __future__ import annotations

import asyncio
import random
import uuid
from typing import Any

import httpx

from .config import Settings


class RinggError(RuntimeError):
    """A Ringg API call failed. `status` is the HTTP code where there was one."""

    def __init__(self, message: str, status: int | None = None, payload: Any = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


class RinggClient:
    """Thin async wrapper over the Ringg REST API."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self._client = client
        self._owned = client is None

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.settings.ringg_base_url, timeout=httpx.Timeout(30.0)
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owned:
            await self._client.aclose()
            self._client = None

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self.settings.ringg_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def _request(self, method: str, path: str, **kw) -> dict[str, Any]:
        if not self.settings.ringg_api_key:
            raise RinggError("RINGG_API_KEY is not set")
        http = await self._http()
        try:
            resp = await http.request(method, path, headers=self._headers, **kw)
        except httpx.RequestError as exc:
            raise RinggError(f"could not reach Ringg: {exc}") from exc

        if resp.status_code == 401:
            raise RinggError("Ringg rejected the API key (401)", 401)
        if resp.status_code >= 400:
            detail: Any
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text[:400]
            raise RinggError(
                f"Ringg {method} {path} failed ({resp.status_code}): {detail}",
                resp.status_code,
                detail,
            )
        try:
            body = resp.json()
        except ValueError as exc:
            raise RinggError(f"Ringg returned non-JSON for {path}") from exc
        return body if isinstance(body, dict) else {"data": body}

    # ---- reads ---------------------------------------------------------------
    async def workspace(self) -> dict[str, Any]:
        body = await self._request("GET", "/workspace")
        return body.get("workspace_info", body)

    async def agents(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/agent/all")
        data = body.get("data") or {}
        return data.get("agents", []) if isinstance(data, dict) else []

    async def numbers(self) -> list[dict[str, Any]]:
        body = await self._request("GET", "/workspace/numbers")
        return body.get("workspace_numbers", [])

    async def call_details(self, call_id: str, send_analysis: bool = True) -> dict[str, Any]:
        body = await self._request(
            "GET",
            "/calling/call-details",
            params={"id": call_id, "send_analysis": str(send_analysis).lower()},
        )
        return body.get("data", body)

    # ---- writes --------------------------------------------------------------
    def build_call_payload(
        self, name: str, phone: str, custom_args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Assemble the POST /calling/outbound/individual body.

        Ringg requires name, mobile_number, agent_id and exactly one of
        from_number_id / from_number.
        """
        caller = self.settings.caller_field
        if caller is None:
            raise RinggError("set RINGG_FROM_NUMBER_ID or RINGG_FROM_NUMBER (exactly one)")
        if not self.settings.ringg_agent_id:
            raise RinggError("RINGG_AGENT_ID is not set")

        payload: dict[str, Any] = {
            "name": name,
            "mobile_number": phone,
            "agent_id": self.settings.ringg_agent_id,
            caller[0]: caller[1],
            "call_category": "real_estate_lead_confirmation",
            "call_config": {"max_call_length": self.settings.max_call_length},
        }
        if custom_args:
            payload["custom_args_values"] = {
                k: ("" if v is None else str(v)) for k, v in custom_args.items()
            }
        return payload

    async def initiate_call(
        self, name: str, phone: str, custom_args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        payload = self.build_call_payload(name, phone, custom_args)
        body = await self._request("POST", "/calling/outbound/individual", json=payload)
        data = body.get("data") or {}
        if not data.get("call_id"):
            raise RinggError(f"Ringg did not return a call_id: {body}", payload=body)
        return data


# --------------------------------------------------------------------------- #
# Mock client -- same interface, canned data shaped like the documented API.
# Lets the full pipeline (and the UI) be exercised with no key and no real calls.
# --------------------------------------------------------------------------- #

_MOCK_CONVERSATIONS: list[tuple[str, str, list[dict[str, str]]]] = [
    (
        "completed",
        "ACCEPTED",
        [
            {"bot": "Hi, this is Anaya from Skyline Realty about your 3BHK enquiry in Whitefield. Is now a good time?"},
            {"user": "Yes, go ahead."},
            {"bot": "Great. Are you still looking to buy in Whitefield?"},
            {"user": "Yes, definitely interested. I'm looking at a 3BHK, budget is around 1.4 crore."},
            {"bot": "Understood. What's your timeline?"},
            {"user": "In the next two months. I'll need a home loan for most of it."},
            {"bot": "We can help with that. Could you visit the site this Saturday?"},
            {"user": "Saturday 11 am works for me. Please send the location."},
            {"bot": "Booked for Saturday 11 am. Thank you!"},
        ],
    ),
    (
        "completed",
        "ACCEPTED",
        [
            {"bot": "Hello, calling from Skyline Realty about the villa plots in Sarjapur."},
            {"user": "I filled that form a while ago but I've already bought somewhere else."},
            {"bot": "Understood. Shall I close your enquiry?"},
            {"user": "Yes, please remove me from your list. Don't call again."},
        ],
    ),
    (
        "completed",
        "ACCEPTED",
        [
            {"bot": "Hi, this is Anaya from Skyline Realty about your apartment enquiry."},
            {"user": "Yeah I remember. I'm still looking but not urgently."},
            {"bot": "What configuration are you considering?"},
            {"user": "Maybe a 2BHK. Budget is tight, around 80 lakhs. Honestly the prices in that area feel too high."},
            {"bot": "Would you like to visit the site?"},
            {"user": "Not right now. Call me back after Diwali, maybe in three or four months."},
        ],
    ),
    (
        "completed",
        "ACCEPTED",
        [
            {"bot": "Hi, Anaya from Skyline Realty regarding the commercial space in Indiranagar."},
            {"user": "Yes! I'm very interested. I need about 2000 square feet of office space."},
            {"bot": "What's your budget range?"},
            {"user": "Up to 2.5 crore, and I can pay cash. I want to close this month."},
            {"bot": "Can we schedule a visit tomorrow?"},
            {"user": "Tomorrow evening at 6 is perfect."},
        ],
    ),
    ("failed", "no answer", []),
    ("completed", "VOICEMAIL_DETECTED", []),
]


class MockRinggClient:
    """Deterministic-ish stand-in for RinggClient.

    Calls start as `registered` and become terminal after `poll_threshold`
    detail reads, so the UI shows a real pending -> completed transition.

    `transcript_lag` reproduces the production race: Ringg reports a call as
    `completed` before post-call processing attaches the transcript, so for this
    many further reads the call comes back terminal with an empty transcript.
    """

    poll_threshold = 1

    def __init__(self, settings: Settings, seed: int | None = None, transcript_lag: int = 0):
        self.settings = settings
        self._calls: dict[str, dict[str, Any]] = {}
        self._reads: dict[str, int] = {}
        self._rand = random.Random(seed)
        self._next = 0
        self.transcript_lag = transcript_lag

    async def aclose(self) -> None:  # parity with RinggClient
        return None

    async def workspace(self) -> dict[str, Any]:
        return {
            "id": "mock-workspace-0001",
            "name": "Mock Workspace (no real calls)",
            "credits": 3025,
            "locked_credits": 0,
        }

    async def agents(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "830f767a-397e-4b39-82ff-235cd344e2f9",
                "agent_display_name": "Real Estate Lead Confirmation (mock)",
                "agent_type": "outbound",
                "call_count": 47,
            }
        ]

    async def numbers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "5d7f9a2b-1c3e-4f6a-8b9c-0d1e2f3a4b5c",
                "number": "+918035736726",
                "display_name": "Mock Sales Line",
                "is_spam": False,
            }
        ]

    def build_call_payload(
        self, name: str, phone: str, custom_args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "name": name,
            "mobile_number": phone,
            "agent_id": self.settings.ringg_agent_id or "mock-agent",
            "from_number_id": self.settings.ringg_from_number_id or "mock-number-id",
            "call_category": "real_estate_lead_confirmation",
            "call_config": {"max_call_length": self.settings.max_call_length},
            "custom_args_values": {k: str(v) for k, v in (custom_args or {}).items()},
        }

    async def initiate_call(
        self, name: str, phone: str, custom_args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        call_id = str(uuid.uuid4())
        status, sub_status, turns = _MOCK_CONVERSATIONS[self._next % len(_MOCK_CONVERSATIONS)]
        self._next += 1
        self._calls[call_id] = {
            "id": call_id,
            "callee_name": name,
            "to_number": phone,
            "final_status": status,
            "final_sub_status": sub_status,
            "turns": turns,
        }
        self._reads[call_id] = 0
        return {
            "call_id": call_id,
            "call_direction": "outbound",
            "call_status": "registered",
            "initiated_at": "2026-09-25T07:33:49.629642",
            "agent_id": self.settings.ringg_agent_id or "mock-agent",
            "custom_args_values": custom_args or {},
        }

    async def call_details(self, call_id: str, send_analysis: bool = True) -> dict[str, Any]:
        await asyncio.sleep(0.01)
        call = self._calls.get(call_id)
        if call is None:
            raise RinggError(f"unknown call id {call_id}", 404)
        self._reads[call_id] += 1
        settled = self._reads[call_id] > self.poll_threshold

        base: dict[str, Any] = {
            "id": call_id,
            "call_direction": "outbound",
            "from_number": "+918035736726",
            "to_number": call["to_number"],
            "callee_name": call["callee_name"],
            "agent_id": self.settings.ringg_agent_id or "mock-agent",
            "initiation_time": "2026-09-25T07:33:49.629642Z",
        }
        if not settled:
            base.update({"call_status": "ongoing", "call_sub_status": "ACCEPTED",
                         "transcription_url": []})
            return base

        # Terminal, but post-call processing may not have attached the transcript yet.
        transcript_ready = self._reads[call_id] > self.poll_threshold + self.transcript_lag
        base.update(
            {
                "call_status": call["final_status"],
                "call_sub_status": call["final_sub_status"],
                "transcription_url": call["turns"] if transcript_ready else [],
            }
        )
        if not transcript_ready:
            return base
        if call["turns"]:
            base["recording_url"] = f"https://example.invalid/recordings/{call_id}.mp3"
            base["call_duration"] = round(12 + self._rand.random() * 90, 2)
        return base


def make_client(settings: Settings) -> RinggClient | MockRinggClient:
    return MockRinggClient(settings) if settings.mock_mode else RinggClient(settings)
