"""Diagnostic: show exactly what Ringg returns for a call, and what we parse.

    ./.venv/bin/python check_call.py                 list recent calls
    ./.venv/bin/python check_call.py <call_id>       inspect one call
    ./.venv/bin/python check_call.py <call_id> --raw dump the full payload
"""
import asyncio
import json
import sys

import httpx

from app.config import settings
from app.ringg import RinggClient
from app.transcript import resolve_turns


def _preview(v, n=110):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    s = s.replace("\n", " ")
    return s[:n] + ("…" if len(s) > n else "")


async def main():
    if not settings.ringg_api_key:
        print("RINGG_API_KEY is not set"); return

    client = RinggClient(settings)
    http = httpx.AsyncClient(timeout=30.0)
    try:
        if len(sys.argv) < 2:
            body = await client._request("GET", "/calling/history", params={"limit": 10})
            print(json.dumps(body, indent=2, default=str)[:4000])
            return

        call_id = sys.argv[1]
        d = await client.call_details(call_id, send_analysis=True)

        if "--raw" in sys.argv:
            print(json.dumps(d, indent=2, default=str)); return

        print("=" * 72)
        print(f"call        {d.get('id', call_id)}")
        print(f"status      {d.get('call_status')}  /  {d.get('call_sub_status')}")
        print(f"to          {d.get('to_number')}   from {d.get('from_number')}")
        print(f"duration    {d.get('call_duration')}")
        print(f"recording   {'yes' if d.get('recording_url') else 'no'}")

        print("\n--- every field Ringg returned ---")
        for k, v in sorted(d.items()):
            kind = type(v).__name__
            size = f"[{len(v)}]" if isinstance(v, (list, str, dict)) else ""
            flag = ""
            if isinstance(v, str) and v.strip().lower().startswith(("http://", "https://")):
                flag = "  <-- URL"
            print(f"  {k:24s} {kind:6s}{size:6s} {_preview(v)}{flag}")

        turns, note = await resolve_turns(d, http)
        print("\n--- what the app parses ---")
        if turns:
            print(f"{len(turns)} turns "
                  f"({sum(t.speaker == 'user' for t in turns)} from the prospect)\n")
            for t in turns[:24]:
                print(f"  {t.speaker:5s} | {t.text[:90]}")
        else:
            print("NO TURNS PARSED")
            print(f"  {note}")
            print("\n  Paste the field list above and I can add support for this shape.")
    finally:
        await http.aclose()
        await client.aclose()


asyncio.run(main())
