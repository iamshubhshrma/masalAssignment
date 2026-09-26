"""Lead store: in-memory dict with JSON-file persistence.

Not a database on purpose -- this is a single-operator tool and the assignment
allows in-memory. All mutations go through one asyncio lock so concurrent
analysis tasks cannot interleave writes. On ephemeral hosting the JSON file is
simply lost on restart, which is acceptable for a demo; swapping in Postgres
means replacing this one class.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path

from .models import ChatTurn, Lead, LeadIntake, utcnow

log = logging.getLogger("masal.store")

TEMPERATURE_ORDER = {"hot": 0, "warm": 1, "cold": 2}


class LeadStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._leads: dict[str, Lead] = {}
        self._lock = asyncio.Lock()

    # ---- persistence ---------------------------------------------------------
    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            for item in json.loads(self.path.read_text() or "[]"):
                lead = Lead.model_validate(item)
                self._leads[lead.id] = lead
            log.info("loaded %d lead(s) from %s", len(self._leads), self.path)
        except (ValueError, OSError) as exc:
            log.warning("could not load %s: %s -- starting empty", self.path, exc)

    def _flush(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(
                [l.model_dump(mode="json") for l in self._leads.values()], indent=2))
            tmp.replace(self.path)
        except OSError as exc:
            log.warning("could not persist leads: %s", exc)

    # ---- reads ---------------------------------------------------------------
    def get(self, lead_id: str) -> Lead | None:
        return self._leads.get(lead_id)

    def all(self) -> list[Lead]:
        """Every lead, highest priority first -- the order the agent works in.

        Unanalysed leads sort to the top so they are never silently buried.
        """
        def key(l: Lead):
            if l.analysis is None:
                return (0, 0, l.created_at)
            return (1, -l.analysis.priority_score, l.created_at)
        return sorted(self._leads.values(), key=key)

    def stats(self) -> dict[str, float | int]:
        leads = list(self._leads.values())
        done = [l for l in leads if l.analysis]
        temps = {"hot": 0, "warm": 0, "cold": 0}
        for l in done:
            temps[l.analysis.temperature] = temps.get(l.analysis.temperature, 0) + 1
        scores = [l.analysis.priority_score for l in done]
        return {
            "total": len(leads),
            "analyzed": len(done),
            "pending_analysis": len(leads) - len(done),
            "hot": temps["hot"],
            "warm": temps["warm"],
            "cold": temps["cold"],
            "act_today": len([l for l in done if l.analysis.urgency == "high"]),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
            "with_objections": len([l for l in done if l.analysis.objections]),
        }

    # ---- writes --------------------------------------------------------------
    async def add(self, intake: LeadIntake) -> Lead:
        async with self._lock:
            lead = Lead(id=uuid.uuid4().hex[:12], **intake.model_dump())
            self._leads[lead.id] = lead
            self._flush()
        return lead

    async def add_many(self, intakes: list[LeadIntake]) -> list[Lead]:
        created: list[Lead] = []
        async with self._lock:
            for intake in intakes:
                lead = Lead(id=uuid.uuid4().hex[:12], **intake.model_dump())
                self._leads[lead.id] = lead
                created.append(lead)
            self._flush()
        return created

    async def update(self, lead: Lead) -> Lead:
        async with self._lock:
            lead.updated_at = utcnow()
            self._leads[lead.id] = lead
            self._flush()
        return lead

    async def append_chat(self, lead: Lead, *turns: ChatTurn) -> Lead:
        async with self._lock:
            stored = self._leads.get(lead.id, lead)
            stored.chat.extend(turns)
            stored.updated_at = utcnow()
            self._leads[stored.id] = stored
            self._flush()
        return stored

    async def delete(self, lead_id: str) -> bool:
        async with self._lock:
            existed = self._leads.pop(lead_id, None) is not None
            if existed:
                self._flush()
        return existed

    async def clear(self) -> int:
        async with self._lock:
            n = len(self._leads)
            self._leads.clear()
            self._flush()
        return n
