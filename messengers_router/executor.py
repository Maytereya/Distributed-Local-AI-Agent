"""Исполнитель `Plan` для patient-router.

Изолирует вызовы сервисного слоя и единообразную обработку ошибок/деградаций.
"""

from __future__ import annotations

from typing import Any

from . import evidence_keys as ek
from .mess_types import Evidence, Plan, SessionState
from .policies import handoff_message, require_auth_for_test_result
from .services import Services


async def execute_plan(plan: Plan, state: SessionState, services: Services) -> Evidence:
    ev = Evidence()

    def _put_optional_step_fallback(tool_name: str, query: str, payload: dict[str, Any]) -> None:
        ev.put(
            f"{tool_name}_optional_suppressed",
            {
                "tool": tool_name,
                "query": query,
                "note": str(payload.get("note") or ""),
                "handoff_reason": str(payload.get("handoff_reason") or ""),
            },
        )

    for step in plan.steps:
        if step.auth == "patient_token":
            need_auth, msg = require_auth_for_test_result(state.is_authenticated)
            if need_auth:
                ev.put(ek.AUTH_REQUIRED, True)
                ev.put(ek.AUTH_MESSAGE, msg)
                return ev

        tool = step.tool
        inp = step.input
        q = inp.get("query", "")
        ent = inp.get("entities") or {}

        try:
            if tool == "doctors_info":
                payload = await services.doctors_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.DOCTORS_INFO, payload)
            elif tool == "doctors_schedule_week":
                payload = await services.doctors_schedule_week(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.DOCTOR_SCHEDULE, payload)
            elif tool == "test_assist":
                payload = await services.test_assist(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.TEST_ASSIST, payload)
            elif tool == "test_prepare":
                payload = await services.test_prepare(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.PREPARE, payload)
            elif tool == "test_result_status":
                payload = await services.test_result_status(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.TEST_RESULT_STATUS, payload)
            elif tool == "price_info":
                payload = await services.price_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.PRICE, payload)
            elif tool == "main_index_info":
                payload = await services.main_index_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.MAIN_INDEX_INFO, payload)
            elif tool == "service_bundle_info":
                payload = await services.service_bundle_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.SERVICE_BUNDLE, payload)
            elif tool == "address_info":
                payload = await services.address_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.ADDRESS, payload)
            elif tool == "news_info":
                payload = await services.news_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put(ek.NEWS, payload)
            else:
                ev.put(ek.UNKNOWN_TOOL, tool)
        except Exception as e:
            if not step.required:
                ev.put(
                    f"{tool}_optional_error",
                    {
                        "tool": tool,
                        "message": str(e),
                        "query": q,
                    },
                )
                continue
            ev.put(
                f"{tool}_error",
                {
                    "tool": tool,
                    "message": str(e),
                    "query": q,
                },
            )
            ev.put(ek.HANDOFF_REQUIRED, True)
            ev.put(ek.HANDOFF_REASON, "service_error")
            ev.put(
                ek.HANDOFF_MESSAGE,
                handoff_message("service_error"),
            )
            return ev

    return ev
