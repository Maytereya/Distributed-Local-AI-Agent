"""Adapter layer between FreeTalk entities and legacy backend payloads."""

from __future__ import annotations

import re
from typing import Any

from .adapter_contracts import AdapterToolRequest, AdapterToolResult, PreparedToolCall


_RESULT_FIELD_LABELS: dict[str, str] = {
    "result_surname": "фамилия",
    "result_year_of_birth": "год рождения",
    "result_analysis_code": "код анализа",
    "result_analysis_number": "номер анализа",
}

_RESULT_SLOT_ALIASES: dict[str, tuple[str, ...]] = {
    "result_surname": ("result_surname",),
    "result_year_of_birth": ("result_year_of_birth",),
    "result_analysis_code": ("result_analysis_code",),
    "result_analysis_number": ("result_analysis_number",),
}

_DOCTOR_ENTITY_KEYS: tuple[str, ...] = (
    "doctor_name",
    "specialty",
    "service_name",
    "branch_name",
    "city",
    "date",
    "date_from",
    "date_to",
    "time",
    "time_from",
    "time_to",
)

_SERVICE_ENTITY_KEYS: tuple[str, ...] = (
    "service_name",
    "test_name",
    "service_variant",
    "doctor_name",
    "doctor_id",
    "specialty",
    "branch_name",
    "city",
)

_ADDRESS_ENTITY_KEYS: tuple[str, ...] = (
    "appointment_action",
    "branch_name",
    "city",
    "service_name",
    "test_name",
    "service_variant",
    "doctor_name",
    "doctor_id",
    "specialty",
)

_PRICE_QUERY_STOPWORDS = {
    "стоимость",
    "сколько",
    "стоит",
    "цена",
    "цену",
    "прайс",
    "узнать",
    "подскажите",
    "подскажи",
    "пожалуйста",
    "клиника",
    "клинике",
}


def _extract_appointment_schedule_context(payload: dict[str, Any]) -> tuple[list[dict[str, str]], list[str]]:
    windows: list[dict[str, str]] = []
    branch_options: list[str] = []
    schedule = payload.get("schedule") if isinstance(payload, dict) else []
    if not isinstance(schedule, list):
        return windows, branch_options

    seen_branches: set[str] = set()
    for row in schedule:
        if not isinstance(row, dict):
            continue
        doctor_name = str(row.get("fio") or "").strip()
        row_schedule = row.get("schedule")
        if not isinstance(row_schedule, dict):
            continue
        for branch_name, days in row_schedule.items():
            branch = str(branch_name or "").strip()
            if branch and branch not in seen_branches:
                seen_branches.add(branch)
                branch_options.append(branch)
            if not isinstance(days, list):
                continue
            for day in days:
                if not isinstance(day, dict):
                    continue
                date_value = str(day.get("date") or "").strip()
                day_slots = day.get("slots")
                if isinstance(day_slots, list):
                    for slot in day_slots:
                        slot_value = str(slot or "").strip()
                        if not (date_value and slot_value):
                            continue
                        windows.append(
                            {
                                "doctor_name": doctor_name,
                                "branch_name": branch,
                                "date": date_value,
                                "time": slot_value[:5] if len(slot_value) >= 5 else slot_value,
                            }
                        )
                start = str(day.get("start") or "").strip()
                end = str(day.get("end") or "").strip()
                if not day_slots and date_value and (start or end):
                    windows.append(
                        {
                            "doctor_name": doctor_name,
                            "branch_name": branch,
                            "date": date_value,
                            "time": start[:5] if start else "",
                            "time_to": end[:5] if end else "",
                        }
                    )
    return windows[:40], branch_options


def _first_present(entities: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = str(entities.get(key) or "").strip()
        if value:
            return value
    return ""


def _normalize_result_ft_entities(entities: dict[str, Any]) -> dict[str, Any]:
    ft_entities: dict[str, Any] = {}
    for field_name, aliases in _RESULT_SLOT_ALIASES.items():
        value = _first_present(entities, aliases)
        if value:
            ft_entities[field_name] = value
    return ft_entities


def _normalize_doctor_ft_entities(entities: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in _DOCTOR_ENTITY_KEYS:
        value = str((entities or {}).get(key) or "").strip()
        if value:
            out[key] = value
    return out


def _doctor_last_name(value: str) -> str:
    parts = [part for part in str(value or "").strip().split() if part]
    return parts[0] if parts else ""


def _compose_service_name(service_name: str, service_variant: str) -> str:
    base = str(service_name or "").strip()
    variant = str(service_variant or "").strip().lower()
    if not base or not variant:
        return base
    if variant in base.lower():
        return base
    return f"{base} {variant}".strip()


def _price_query_text(prepared_call: PreparedToolCall) -> str:
    entities = dict(prepared_call.ft_entities or {})
    for key in ("service_name", "test_name"):
        value = str(entities.get(key) or "").strip()
        if value:
            return value
    return str(prepared_call.backend_query or "").strip()


def _normalize_price_text(value: str) -> str:
    text = str(value or "").strip().lower().replace("ё", "е")
    text = re.sub(r"\bоак\b", " общий анализ крови ", text)
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _price_token_key(token: str) -> str:
    raw = str(token or "").strip().lower()
    if raw.startswith("общ"):
        return "общ"
    if raw.startswith("кров"):
        return "кров"
    if raw.startswith("моч"):
        return "моч"
    if raw.startswith("анализ"):
        return "анализ"
    if raw.startswith("биохим"):
        return "биохим"
    if raw.startswith("глюкоз"):
        return "глюкоз"
    if len(raw) > 6:
        return raw[:6]
    return raw


def _price_tokens(value: str) -> list[str]:
    normalized = _normalize_price_text(value)
    tokens: list[str] = []
    seen: set[str] = set()
    for token in normalized.split():
        if len(token) < 2 or token in _PRICE_QUERY_STOPWORDS:
            continue
        key = _price_token_key(token)
        if not key or key in seen:
            continue
        seen.add(key)
        tokens.append(key)
    return tokens


def _price_relevance_score(row: dict[str, Any], *, query_text: str, query_tokens: list[str]) -> float:
    name = str(row.get("serviceName") or row.get("name") or "").strip()
    if not name or not query_tokens:
        return 0.0
    row_tokens = set(_price_tokens(name))
    if not row_tokens:
        return 0.0
    query_set = set(query_tokens)
    overlap = query_set & row_tokens
    coverage = len(overlap) / max(1, len(query_set))
    score = coverage * 100.0
    normalized_query = _normalize_price_text(query_text)
    normalized_name = _normalize_price_text(name)
    if normalized_query and normalized_query == normalized_name:
        score += 120.0
    elif normalized_query and normalized_query in normalized_name:
        score += 80.0
    if query_set <= row_tokens:
        score += 60.0
    elif coverage < 0.67:
        score -= 30.0
    return score


def _rank_price_rows(rows: Any, *, prepared_call: PreparedToolCall) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(rows, list):
        return [], False
    price_rows = [dict(row) for row in rows if isinstance(row, dict)]
    if len(price_rows) < 2:
        return price_rows, False
    query_text = _price_query_text(prepared_call)
    query_tokens = _price_tokens(query_text)
    if not query_tokens:
        return price_rows, False

    scored = [
        (
            _price_relevance_score(row, query_text=query_text, query_tokens=query_tokens),
            index,
            row,
        )
        for index, row in enumerate(price_rows)
    ]
    ranked = [row for _, _, row in sorted(scored, key=lambda item: (-item[0], item[1]))]
    changed = [row.get("serviceName") or row.get("name") for row in ranked] != [
        row.get("serviceName") or row.get("name") for row in price_rows
    ]
    return ranked, changed


def _result_backend_entities(ft_entities: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    backend_entities = {
        key: value
        for key, value in (original or {}).items()
        if str(key or "").strip() and not str(key).startswith("result_")
    }
    mappings = {
        "result_surname": "surname",
        "result_year_of_birth": "year",
        "result_analysis_code": "filial",
        "result_analysis_number": "number",
    }
    for ft_key, backend_key in mappings.items():
        value = str(ft_entities.get(ft_key) or "").strip()
        if value:
            backend_entities[backend_key] = value
    return backend_entities


def _normalize_result_missing_fields(raw_fields: Any) -> tuple[list[str], list[str]]:
    if not isinstance(raw_fields, list):
        return [], []
    slots: list[str] = []
    labels: list[str] = []
    seen_slots: set[str] = set()
    for item in raw_fields:
        text = str(item or "").strip().lower()
        if not text:
            continue
        slot = ""
        if "фам" in text or text == "surname":
            slot = "result_surname"
        elif "год" in text or text == "year":
            slot = "result_year_of_birth"
        elif "код" in text or "фили" in text or text == "filial":
            slot = "result_analysis_code"
        elif "номер" in text or "заказ" in text or text == "number":
            slot = "result_analysis_number"
        if not slot or slot in seen_slots:
            continue
        seen_slots.add(slot)
        slots.append(slot)
        labels.append(_RESULT_FIELD_LABELS[slot])
    return slots, labels


def _normalize_service_ft_entities(entities: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    source = dict(entities or {})
    service_name = _compose_service_name(
        str(source.get("service_name") or ""),
        str(source.get("service_variant") or ""),
    )
    test_name = str(source.get("test_name") or "").strip()
    if service_name:
        out["service_name"] = service_name
    if test_name:
        out["test_name"] = test_name
    service_variant = str(source.get("service_variant") or "").strip()
    if service_variant:
        out["service_variant"] = service_variant
    for key in _SERVICE_ENTITY_KEYS:
        if key in {"service_name", "test_name", "service_variant"}:
            continue
        value = str(source.get(key) or "").strip()
        if value:
            out[key] = value
    return out


def _service_backend_entities(ft_entities: dict[str, Any]) -> dict[str, Any]:
    backend: dict[str, Any] = {}
    for key in ("service_name", "test_name", "doctor_name", "doctor_id", "specialty", "branch_name", "city"):
        value = str(ft_entities.get(key) or "").strip()
        if value:
            backend[key] = value
    return backend


def _normalize_address_ft_entities(entities: dict[str, Any]) -> dict[str, Any]:
    service_entities = _normalize_service_ft_entities(entities)
    out: dict[str, Any] = {}
    for key in _ADDRESS_ENTITY_KEYS:
        value = str(service_entities.get(key) or (entities or {}).get(key) or "").strip()
        if value:
            out[key] = value
    return out


def _address_backend_entities(ft_entities: dict[str, Any]) -> dict[str, Any]:
    backend: dict[str, Any] = {}
    branch_name = str(ft_entities.get("branch_name") or "").strip()
    city = str(ft_entities.get("city") or "").strip()
    if branch_name:
        backend["branch"] = branch_name
    if city:
        backend["city"] = city
    for key in ("service_name", "test_name", "doctor_name", "doctor_id", "specialty"):
        value = str(ft_entities.get(key) or "").strip()
        if value:
            backend[key] = value
    if str(ft_entities.get("appointment_action") or "").strip().lower() in {"book", "reschedule", "cancel"}:
        backend["__appointment_mode"] = True
    return backend


def _extract_payload_doctor_name(tool_name: str, payload: dict[str, Any]) -> str:
    bucket = payload.get("doctors") if str(tool_name or "").strip() == "doctors_info" else payload.get("schedule")
    if isinstance(bucket, list):
        names: list[str] = []
        for row in bucket:
            if not isinstance(row, dict):
                continue
            fio = str(row.get("fio") or "").strip()
            if fio:
                names.append(fio)
        unique_names = list(dict.fromkeys(names))
        if len(unique_names) == 1:
            return unique_names[0]

    entities_used = payload.get("entities_used") if isinstance(payload, dict) else {}
    if isinstance(entities_used, dict):
        for key in ("doctor_name_resolved", "raw_name", "doctor_name", "doctor_query", "doctor_resolved", "last_name"):
            value = str(entities_used.get(key) or "").strip()
            if value:
                return value
    return ""


def _normalize_doctor_entities_used(
    tool_name: str,
    payload: dict[str, Any],
    prepared_call: PreparedToolCall,
) -> dict[str, Any]:
    out = dict(prepared_call.ft_entities or {})
    raw = payload.get("entities_used") if isinstance(payload.get("entities_used"), dict) else {}
    doctor_name = _extract_payload_doctor_name(tool_name, payload)
    if doctor_name:
        out["doctor_name"] = doctor_name
    specialty = str(raw.get("specialty_query") or raw.get("specialty") or "").strip()
    if specialty and not str(out.get("specialty") or "").strip():
        out["specialty"] = specialty
    service_name = str(raw.get("service_query") or raw.get("service_name") or "").strip()
    if service_name and not str(out.get("service_name") or "").strip():
        out["service_name"] = service_name
    region_query = str(raw.get("region_query") or raw.get("region_name") or "").strip()
    if region_query and not str(out.get("branch_name") or out.get("city") or "").strip():
        out["branch_name"] = region_query
    return out


def _normalize_service_entities_used(
    tool_name: str,
    payload: dict[str, Any],
    prepared_call: PreparedToolCall,
) -> dict[str, Any]:
    out = dict(prepared_call.ft_entities or {})
    raw = payload.get("entities_used") if isinstance(payload.get("entities_used"), dict) else {}
    effective_service = str(
        raw.get("service_name_effective")
        or raw.get("service_name")
        or raw.get("test_name")
        or ""
    ).strip()
    if effective_service:
        if tool_name == "test_assist" or (
            str(prepared_call.ft_entities.get("test_name") or "").strip()
            and not str(prepared_call.ft_entities.get("service_name") or "").strip()
        ):
            out["test_name"] = effective_service
        else:
            out["service_name"] = effective_service

    doctor_name = str(raw.get("doctor_name_resolved") or raw.get("doctor_name") or "").strip()
    if doctor_name:
        out["doctor_name"] = doctor_name
    doctor_id = str(raw.get("doctor_id_resolved") or raw.get("doctor_id") or "").strip()
    if doctor_id:
        out["doctor_id"] = doctor_id
    specialty = str(raw.get("specialty_query") or raw.get("specialty") or "").strip()
    if specialty and not str(out.get("specialty") or "").strip():
        out["specialty"] = specialty
    return out


def _normalize_address_entities_used(
    payload: dict[str, Any],
    prepared_call: PreparedToolCall,
) -> dict[str, Any]:
    out = dict(prepared_call.ft_entities or {})
    raw = payload.get("entities_used") if isinstance(payload.get("entities_used"), dict) else {}

    branch_name = str(
        raw.get("branch_name")
        or raw.get("branch")
        or raw.get("region")
        or raw.get("region_name")
        or ""
    ).strip()
    if branch_name:
        out["branch_name"] = branch_name

    city = str(raw.get("city") or "").strip()
    if city:
        out["city"] = city

    effective_service = str(raw.get("service_name") or raw.get("test_name") or "").strip()
    if effective_service:
        if str(prepared_call.ft_entities.get("test_name") or "").strip() and not str(prepared_call.ft_entities.get("service_name") or "").strip():
            out["test_name"] = effective_service
        elif str(prepared_call.ft_entities.get("service_name") or "").strip():
            out["service_name"] = effective_service

    doctor_name = str(raw.get("doctor_name_resolved") or raw.get("doctor_name") or "").strip()
    if doctor_name:
        out["doctor_name"] = doctor_name
    specialty = str(raw.get("specialty_query") or raw.get("specialty") or "").strip()
    if specialty and not str(out.get("specialty") or "").strip():
        out["specialty"] = specialty
    return out


def _extract_address_branch_options(payload: dict[str, Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in (payload.get("addresses") or []):
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


class FreeTalkAdapter:
    """Translate FT-facing entities to legacy backend payloads and back."""

    def prepare_tool_call(
        self,
        *,
        tool_name: str,
        user_message: str,
        entities: dict[str, Any],
    ) -> PreparedToolCall:
        request = AdapterToolRequest(
            tool_name=str(tool_name or "").strip(),
            user_message=str(user_message or ""),
            ft_entities=dict(entities or {}),
        )
        if request.tool_name == "test_result_status":
            return self._prepare_test_result_call(request)
        if request.tool_name == "doctors_info":
            return self._prepare_doctors_info_call(request)
        if request.tool_name == "doctors_schedule_week":
            return self._prepare_doctors_schedule_call(request)
        if request.tool_name in {"service_bundle_info", "price_info", "test_prepare", "test_assist"}:
            return self._prepare_service_call(request)
        if request.tool_name == "address_info":
            return self._prepare_address_call(request)
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=dict(request.ft_entities),
            ft_entities=dict(request.ft_entities),
            metadata={"mode": "passthrough"},
        )

    def normalize_tool_payload(
        self,
        *,
        tool_name: str,
        payload: dict[str, Any],
        prepared_call: PreparedToolCall,
    ) -> AdapterToolResult:
        if str(tool_name or "").strip() == "test_result_status":
            return self._normalize_test_result_payload(payload, prepared_call)
        if str(tool_name or "").strip() in {"doctors_info", "doctors_schedule_week"}:
            return self._normalize_doctor_payload(tool_name, payload, prepared_call)
        if str(tool_name or "").strip() in {"service_bundle_info", "price_info", "test_prepare", "test_assist"}:
            return self._normalize_service_payload(tool_name, payload, prepared_call)
        if str(tool_name or "").strip() == "address_info":
            return self._normalize_address_payload(payload, prepared_call)
        return AdapterToolResult(
            tool_name=str(tool_name or "").strip(),
            ft_payload=dict(payload or {}),
            metadata={"mode": "passthrough"},
        )

    def _prepare_test_result_call(self, request: AdapterToolRequest) -> PreparedToolCall:
        ft_entities = _normalize_result_ft_entities(request.ft_entities)
        backend_entities = _result_backend_entities(ft_entities, request.ft_entities)
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=backend_entities,
            ft_entities=ft_entities,
            metadata={
                "domain": "test_result",
                "contract": "ft_result_v1",
            },
        )

    def _prepare_doctors_info_call(self, request: AdapterToolRequest) -> PreparedToolCall:
        ft_entities = _normalize_doctor_ft_entities(request.ft_entities)
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=dict(ft_entities),
            ft_entities=ft_entities,
            metadata={
                "domain": "doctor_info",
                "contract": "ft_doctor_v1",
            },
        )

    def _prepare_doctors_schedule_call(self, request: AdapterToolRequest) -> PreparedToolCall:
        ft_entities = _normalize_doctor_ft_entities(request.ft_entities)
        backend_entities = dict(ft_entities)
        last_name = _doctor_last_name(str(ft_entities.get("doctor_name") or ""))
        if last_name:
            backend_entities["last_name"] = last_name
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=backend_entities,
            ft_entities=ft_entities,
            metadata={
                "domain": "doctor_schedule",
                "contract": "ft_doctor_v1",
            },
        )

    def _prepare_service_call(self, request: AdapterToolRequest) -> PreparedToolCall:
        ft_entities = _normalize_service_ft_entities(request.ft_entities)
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=_service_backend_entities(ft_entities),
            ft_entities=ft_entities,
            metadata={
                "domain": "service_query",
                "contract": "ft_service_v1",
            },
        )

    def _prepare_address_call(self, request: AdapterToolRequest) -> PreparedToolCall:
        ft_entities = _normalize_address_ft_entities(request.ft_entities)
        return PreparedToolCall(
            tool_name=request.tool_name,
            backend_query=request.user_message,
            backend_entities=_address_backend_entities(ft_entities),
            ft_entities=ft_entities,
            metadata={
                "domain": "address_query",
                "contract": "ft_address_v1",
            },
        )

    def _normalize_test_result_payload(
        self,
        payload: dict[str, Any],
        prepared_call: PreparedToolCall,
    ) -> AdapterToolResult:
        out = dict(payload or {})
        raw_entities_used = out.get("entities_used") if isinstance(out.get("entities_used"), dict) else {}
        raw_missing_fields = out.get("missing_fields")

        normalized_entities_used = _normalize_result_ft_entities(raw_entities_used)
        if not normalized_entities_used and prepared_call.ft_entities:
            normalized_entities_used = dict(prepared_call.ft_entities)

        missing_slots_ft, missing_fields_user = _normalize_result_missing_fields(raw_missing_fields)
        if missing_slots_ft:
            out["missing_slots_ft"] = missing_slots_ft
            out["missing_fields_backend"] = list(raw_missing_fields) if isinstance(raw_missing_fields, list) else []
            out["missing_fields"] = missing_fields_user
        if normalized_entities_used:
            out["entities_used_ft"] = normalized_entities_used

        adapter_meta = dict(out.get("adapter_meta") or {})
        adapter_meta.update(
            {
                "domain": "test_result",
                "contract": "ft_result_v1",
            }
        )
        out["adapter_meta"] = adapter_meta

        return AdapterToolResult(
            tool_name=prepared_call.tool_name,
            ft_payload=out,
            metadata=adapter_meta,
        )

    def _normalize_doctor_payload(
        self,
        tool_name: str,
        payload: dict[str, Any],
        prepared_call: PreparedToolCall,
    ) -> AdapterToolResult:
        out = dict(payload or {})
        normalized_entities_used = _normalize_doctor_entities_used(tool_name, out, prepared_call)
        if normalized_entities_used:
            out["entities_used_ft"] = normalized_entities_used
        if tool_name == "doctors_schedule_week":
            appointment_windows, branch_options = _extract_appointment_schedule_context(out)
            if appointment_windows:
                out["appointment_windows"] = appointment_windows
            if branch_options:
                out["appointment_branch_options"] = branch_options
        adapter_meta = dict(out.get("adapter_meta") or {})
        adapter_meta.update(
            {
                "domain": "doctor_schedule" if tool_name == "doctors_schedule_week" else "doctor_info",
                "contract": "ft_doctor_v1",
            }
        )
        out["adapter_meta"] = adapter_meta
        return AdapterToolResult(
            tool_name=str(tool_name or "").strip(),
            ft_payload=out,
            metadata=adapter_meta,
        )

    def _normalize_service_payload(
        self,
        tool_name: str,
        payload: dict[str, Any],
        prepared_call: PreparedToolCall,
    ) -> AdapterToolResult:
        out = dict(payload or {})
        if str(tool_name or "").strip() == "price_info":
            family_variants = out.get("family_variants")
            prices = out.get("prices")
            ranking_changed = False
            if isinstance(family_variants, list) and family_variants and not isinstance(prices, list):
                ranked, ranking_changed = _rank_price_rows(family_variants, prepared_call=prepared_call)
                visible_limit = max(1, int(out.get("visible_limit") or 5))
                showing_all = bool(out.get("showing_all"))
                out["prices"] = list(ranked if showing_all else ranked[:visible_limit])
            elif isinstance(prices, list) and prices:
                ranked, ranking_changed = _rank_price_rows(prices, prepared_call=prepared_call)
                out["prices"] = ranked
            if ranking_changed:
                out["price_ranking_applied"] = True
        normalized_entities_used = _normalize_service_entities_used(tool_name, out, prepared_call)
        if normalized_entities_used:
            out["entities_used_ft"] = normalized_entities_used
        adapter_meta = dict(out.get("adapter_meta") or {})
        adapter_meta.update(
            {
                "domain": "service_query",
                "contract": "ft_service_v1",
            }
        )
        out["adapter_meta"] = adapter_meta
        return AdapterToolResult(
            tool_name=str(tool_name or "").strip(),
            ft_payload=out,
            metadata=adapter_meta,
        )

    def _normalize_address_payload(
        self,
        payload: dict[str, Any],
        prepared_call: PreparedToolCall,
    ) -> AdapterToolResult:
        out = dict(payload or {})
        normalized_entities_used = _normalize_address_entities_used(out, prepared_call)
        if normalized_entities_used:
            out["entities_used_ft"] = normalized_entities_used
        branch_options = _extract_address_branch_options(out)
        if branch_options:
            out["appointment_branch_options"] = branch_options
        adapter_meta = dict(out.get("adapter_meta") or {})
        adapter_meta.update(
            {
                "domain": "address_query",
                "contract": "ft_address_v1",
            }
        )
        out["adapter_meta"] = adapter_meta
        return AdapterToolResult(
            tool_name="address_info",
            ft_payload=out,
            metadata=adapter_meta,
        )
