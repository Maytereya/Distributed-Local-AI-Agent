"""Adapter layer between FreeTalk entities and legacy backend payloads."""

from __future__ import annotations

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
    "branch_name",
    "city",
    "service_name",
    "test_name",
    "service_variant",
    "doctor_name",
    "doctor_id",
    "specialty",
)


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
        value = str(service_entities.get(key) or "").strip()
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
    return backend


def _extract_payload_doctor_name(tool_name: str, payload: dict[str, Any]) -> str:
    bucket = payload.get("doctors") if str(tool_name or "").strip() == "doctors_info" else payload.get("schedule")
    if isinstance(bucket, list):
        for row in bucket:
            if not isinstance(row, dict):
                continue
            fio = str(row.get("fio") or "").strip()
            if fio:
                return fio

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
