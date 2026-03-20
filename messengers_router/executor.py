"""Исполнитель `Plan` для patient-router.

Изолирует вызовы сервисного слоя и единообразную обработку ошибок/деградаций.
"""

from __future__ import annotations

from typing import Any

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
                ev.put("auth_required", True)
                ev.put("auth_message", msg)
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
                ev.put("doctors_info", payload)
            elif tool == "doctors_schedule_week":
                payload = await services.doctors_schedule_week(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("doctor_schedule", payload)
            elif tool == "appointment_help":
                payload = await services.appointment_help(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("appointment", payload)
            elif tool == "test_assist":
                payload = await services.test_assist(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("test_assist", payload)
            elif tool == "test_prepare":
                payload = await services.test_prepare(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("prepare", payload)
            elif tool == "test_result_status":
                payload = await services.test_result_status(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("test_result_status", payload)
            elif tool == "price_info":
                payload = await services.price_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("price", payload)
            elif tool == "main_index_info":
                payload = await services.main_index_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("main_index_info", payload)
            elif tool == "service_bundle_info":
                payload = await services.service_bundle_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("service_bundle", payload)
            elif tool == "address_info":
                payload = await services.address_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("address", payload)
            elif tool == "news_info":
                payload = await services.news_info(q, ent)
                if not step.required and isinstance(payload, dict) and payload.get("handoff_required"):
                    _put_optional_step_fallback(tool, q, payload)
                    continue
                ev.put("news", payload)
            else:
                ev.put("unknown_tool", tool)
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
            ev.put("handoff_required", True)
            ev.put("handoff_reason", "service_error")
            ev.put(
                "handoff_message",
                handoff_message("service_error"),
            )
            return ev

    return ev
