from __future__ import annotations

import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .memory import MemoryStore
from .services import Services
from .router import patient_routing_stream

router = APIRouter()

memory = MemoryStore(ttl_seconds=3600, pending_ttl_seconds=900)
services = Services()


@router.post("/api/messenger-generate")
async def messenger_generate(payload: dict):
    session_id = payload.get("session_id") or "anon"
    text = (payload.get("text") or "").strip()

    state = await memory.aget(session_id)
    try:
        memory.append_turn(state, "user", text)

        async def event_stream():
            async for env in patient_routing_stream(text, state, services, memory):
                obj = {
                    "text": env.text,
                    "attachments": env.attachments,
                    "handoff": env.handoff,
                }
                yield (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")
    finally:
        await memory.aset(state)