import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.adapter import FreeTalkAdapter
from localragagent.freetalk.agent import FreeTalkAgent
from localragagent.freetalk.routing_contract import ClinicalDecision
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.contracts import DialogState, SessionContext


def _cfg() -> FreeTalkConfig:
    return FreeTalkConfig(
        mode_key="Free-talk-Ai",
        redis_url="redis://redis:6379/0",
        redis_prefix="ft",
        session_ttl_sec=86400,
        history_tail_turns=14,
        compaction_trigger_turns=9999,
        summary_keep_turns=8,
        max_tool_steps=3,
        llm_timeout_s=5,
        llm_queue_timeout_ms=2000,
        include_meili_tools=False,
        enable_web_search_tool=False,
        web_search_url="http://searxng:8080",
        web_search_timeout_s=4,
        web_search_max_results=5,
        web_search_language="ru-RU",
        web_search_healthcheck_timeout_s=2,
        web_search_healthcheck_ttl_s=5,
        context_window_tokens=24576,
        context_warn_ratio=0.82,
        context_estimate_chars_per_token=4,
        context_response_reserve_tokens=2048,
        persistent_memory_path=PROJECT_ROOT / "app_data/free_talk_memory/dialog_summaries.jsonl",
        system_prompt_path=PROJECT_ROOT / "src/localragagent/freetalk/system_prompt.txt",
    )


class InMemoryMemory:
    def __init__(self) -> None:
        self.turns: dict[str, list[dict[str, object]]] = {}
        self.summaries: dict[str, str] = {}
        self.meta: dict[str, dict[str, str]] = {}

    async def load_context(self, session_id: str, *, history_tail_turns: int) -> SessionContext:
        sid = str(session_id or "").strip()
        all_turns = list(self.turns.get(sid, []))
        tail = all_turns[-history_tail_turns:] if history_tail_turns > 0 else all_turns
        return SessionContext(session_id=sid, summary=self.summaries.get(sid, ""), turns=tail)

    async def append_exchange(
        self,
        session_id: str,
        *,
        user_text: str,
        assistant_text: str,
        source: str,
        source_fragments: list[dict[str, str]] | None = None,
    ) -> None:
        sid = str(session_id or "").strip()
        bucket = self.turns.setdefault(sid, [])
        bucket.append({"role": "user", "content": str(user_text or "")})
        bucket.append({"role": "assistant", "content": str(assistant_text or ""), "source": str(source or "")})
        _ = source_fragments

    async def get_turn_count(self, session_id: str) -> int:
        return len(self.turns.get(str(session_id or "").strip(), []))

    async def save_summary(self, session_id: str, summary: str) -> None:
        self.summaries[str(session_id or "").strip()] = str(summary or "")

    async def get_meta_int(self, session_id: str, key: str, default: int = 0) -> int:
        try:
            return int(self.meta.get(str(session_id or "").strip(), {}).get(str(key or "").strip(), default))
        except Exception:
            return int(default)

    async def set_meta_int(self, session_id: str, key: str, value: int) -> None:
        self.meta.setdefault(str(session_id or "").strip(), {})[str(key or "").strip()] = str(int(value))

    async def get_meta_str(self, session_id: str, key: str, default: str = "") -> str:
        return str(self.meta.get(str(session_id or "").strip(), {}).get(str(key or "").strip(), default) or "")

    async def set_meta_str(self, session_id: str, key: str, value: str) -> None:
        self.meta.setdefault(str(session_id or "").strip(), {})[str(key or "").strip()] = str(value or "")

    async def clear_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip()
        self.turns.pop(sid, None)
        self.summaries.pop(sid, None)
        self.meta.pop(sid, None)


class InMemoryPersist:
    async def append_snapshot(
        self,
        *,
        session_id: str,
        summary: str,
        key_facts: list[str],
        open_loops: list[str],
        extra: dict[str, object] | None = None,
    ) -> None:
        _ = session_id, summary, key_facts, open_loops, extra


class AdapterServices:
    def __init__(self) -> None:
        self.last_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def test_result_status(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        return {
            "ready": False,
            "missing_fields": ["surname", "year", "filial", "number"],
            "entities_used": dict(entities),
            "note": "missing_result_fields",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {"test_result_status": self.test_result_status}


class ResultAdapterAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = user_message, context, dialog_state, remembered_doctor
        return ClinicalDecision(
            intent="test_result",
            confidence=0.95,
            entities={
                "result_surname": "Иванов",
                "result_year_of_birth": "1989",
                "result_analysis_code": "Бг",
                "result_analysis_number": "1234",
            },
            missing_slots=[],
            clarify_question="",
            tool_plan=["test_result_status"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


def test_adapter_prepares_test_result_call_with_legacy_backend_fields():
    adapter = FreeTalkAdapter()

    prepared = adapter.prepare_tool_call(
        tool_name="test_result_status",
        user_message="Иванов, 1989, Бг, 1234",
        entities={
            "result_surname": "Иванов",
            "result_year_of_birth": "1989",
            "result_analysis_code": "Бг",
            "result_analysis_number": "1234",
        },
    )

    assert prepared.ft_entities["result_surname"] == "Иванов"
    assert prepared.ft_entities["result_analysis_code"] == "Бг"
    assert prepared.backend_entities["surname"] == "Иванов"
    assert prepared.backend_entities["year"] == "1989"
    assert prepared.backend_entities["filial"] == "Бг"
    assert prepared.backend_entities["number"] == "1234"
    assert "result_analysis_code" not in prepared.backend_entities


def test_adapter_normalizes_test_result_missing_fields_to_user_facing_shape():
    adapter = FreeTalkAdapter()
    prepared = adapter.prepare_tool_call(
        tool_name="test_result_status",
        user_message="Иванов, 1989, Бг, 1234",
        entities={
            "result_surname": "Иванов",
            "result_year_of_birth": "1989",
            "result_analysis_code": "Бг",
            "result_analysis_number": "1234",
        },
    )

    result = adapter.normalize_tool_payload(
        tool_name="test_result_status",
        payload={
            "ready": False,
            "missing_fields": ["surname", "year", "filial", "number"],
            "entities_used": {
                "surname": "Иванов",
                "year": "1989",
                "filial": "Бг",
                "number": "1234",
            },
        },
        prepared_call=prepared,
    )

    assert result.ft_payload["missing_slots_ft"] == [
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    ]
    assert result.ft_payload["missing_fields"] == [
        "фамилия",
        "год рождения",
        "код анализа",
        "номер анализа",
    ]
    assert result.ft_payload["entities_used_ft"]["result_analysis_code"] == "Бг"
    assert result.ft_payload["adapter_meta"]["domain"] == "test_result"


def test_agent_uses_adapter_to_send_legacy_result_fields_to_backend():
    services = AdapterServices()
    agent = ResultAdapterAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=InMemoryMemory(),  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )

    reply = asyncio.run(agent.chat("Проверь результат анализа", "adapter_result"))

    assert reply.tool_name == "test_result_status"


def test_adapter_normalizes_schedule_payload_to_appointment_context():
    adapter = FreeTalkAdapter()
    prepared = adapter.prepare_tool_call(
        tool_name="doctors_schedule_week",
        user_message="расписание Трубина",
        entities={"doctor_name": "Трубин Алексей Юрьевич"},
    )

    result = adapter.normalize_tool_payload(
        tool_name="doctors_schedule_week",
        payload={
            "schedule": [
                {
                    "fio": "Трубин Алексей Юрьевич",
                    "schedule": {
                        "г. Самара, пр. Ленина, 5": [
                            {
                                "date": "2026-04-16",
                                "slots": ["09:00", "09:30"],
                                "start": "09:00",
                                "end": "12:00",
                            }
                        ]
                    },
                }
            ],
            "entities_used": {"doctor_name_resolved": "Трубин Алексей Юрьевич"},
        },
        prepared_call=prepared,
    )

    assert result.ft_payload["entities_used_ft"]["doctor_name"] == "Трубин Алексей Юрьевич"
    assert result.ft_payload["appointment_branch_options"] == ["г. Самара, пр. Ленина, 5"]
    assert result.ft_payload["appointment_windows"][0]["date"] == "2026-04-16"
    assert result.ft_payload["appointment_windows"][0]["time"] == "09:00"


class DoctorAdapterServices:
    def __init__(self) -> None:
        self.last_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def doctors_schedule_week(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        return {
            "schedule": [
                {
                    "fio": "Дразнин Антон Владимирович",
                    "schedule": {
                        "г. Самара, пр. Ленина, 5": [
                            {
                                "date": "2026-04-14",
                                "slots": ["09:00", "09:30"],
                                "start": "09:00",
                                "end": "18:00",
                            }
                        ]
                    },
                }
            ],
            "entities_used": {
                "last_name": "Дразнин",
                "raw_name": "Дразнин Антон Владимирович",
                "region_name": "Ленина",
            },
            "note": "doctors_schedule_week",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {"doctors_schedule_week": self.doctors_schedule_week}


class DoctorScheduleAdapterAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = user_message, context, dialog_state, remembered_doctor
        return ClinicalDecision(
            intent="doctor_schedule",
            confidence=0.95,
            entities={
                "doctor_name": "Дразнин Антон Владимирович",
                "branch_name": "Ленина",
                "time": "утром",
                "time_from": "08:00",
            },
            missing_slots=[],
            clarify_question="",
            tool_plan=["doctors_schedule_week"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


def test_adapter_prepares_doctor_schedule_call_with_derived_last_name():
    adapter = FreeTalkAdapter()

    prepared = adapter.prepare_tool_call(
        tool_name="doctors_schedule_week",
        user_message="Покажи расписание Дразнина",
        entities={
            "doctor_name": "Дразнин Антон Владимирович",
            "branch_name": "Ленина",
            "time": "утром",
        },
    )

    assert prepared.ft_entities["doctor_name"] == "Дразнин Антон Владимирович"
    assert prepared.backend_entities["doctor_name"] == "Дразнин Антон Владимирович"
    assert prepared.backend_entities["last_name"] == "Дразнин"
    assert prepared.backend_entities["branch_name"] == "Ленина"


def test_adapter_normalizes_doctor_schedule_entities_used_to_ft_shape():
    adapter = FreeTalkAdapter()
    prepared = adapter.prepare_tool_call(
        tool_name="doctors_schedule_week",
        user_message="Покажи расписание Дразнина",
        entities={
            "doctor_name": "Дразнин Антон Владимирович",
            "branch_name": "Ленина",
            "time": "утром",
        },
    )

    result = adapter.normalize_tool_payload(
        tool_name="doctors_schedule_week",
        payload={
            "schedule": [{"fio": "Дразнин Антон Владимирович"}],
            "entities_used": {
                "last_name": "Дразнин",
                "raw_name": "Дразнин Антон Владимирович",
                "region_name": "Ленина",
            },
            "note": "doctors_schedule_week",
        },
        prepared_call=prepared,
    )

    assert result.ft_payload["entities_used_ft"]["doctor_name"] == "Дразнин Антон Владимирович"
    assert result.ft_payload["entities_used_ft"]["branch_name"] == "Ленина"
    assert result.ft_payload["adapter_meta"]["domain"] == "doctor_schedule"


def test_agent_uses_adapter_to_send_doctor_schedule_entities_to_backend():
    services = DoctorAdapterServices()
    agent = DoctorScheduleAdapterAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=InMemoryMemory(),  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )

    reply = asyncio.run(agent.chat("Покажи расписание Дразнина", "adapter_doctor"))

    assert reply.tool_name == "doctors_schedule_week"
    assert services.last_entities["doctor_name"] == "Дразнин Антон Владимирович"
    assert services.last_entities["last_name"] == "Дразнин"
    assert services.last_entities["branch_name"] == "Ленина"
    assert reply.tool_payload["entities_used_ft"]["doctor_name"] == "Дразнин Антон Владимирович"


class ServiceAdapterServices:
    def __init__(self) -> None:
        self.last_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        return {
            "prices": [{"serviceName": "ФГДС с седацией", "cost": 5200}],
            "entities_used": {
                "service_name_effective": "ФГДС с седацией",
                "doctor_name_resolved": "Дразнин Антон Владимирович",
            },
            "note": "price_info",
        }

    async def test_assist(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        return {
            "tests": [{"serviceName": "Общий анализ крови", "cost": 650}],
            "entities_used": dict(entities),
            "note": "test_assist",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "price_info": self.price_info,
            "test_assist": self.test_assist,
        }


class ServiceAdapterAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = user_message, context, dialog_state, remembered_doctor
        return ClinicalDecision(
            intent="price",
            confidence=0.95,
            entities={
                "service_name": "ФГДС",
                "service_variant": "с седацией",
                "doctor_name": "Дразнин Антон Владимирович",
            },
            missing_slots=[],
            clarify_question="",
            tool_plan=["price_info"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


def test_adapter_prepares_price_call_with_composed_service_name():
    adapter = FreeTalkAdapter()

    prepared = adapter.prepare_tool_call(
        tool_name="price_info",
        user_message="Сколько стоит ФГДС с седацией?",
        entities={
            "service_name": "ФГДС",
            "service_variant": "с седацией",
            "doctor_name": "Дразнин Антон Владимирович",
        },
    )

    assert prepared.ft_entities["service_name"] == "ФГДС с седацией"
    assert prepared.ft_entities["service_variant"] == "с седацией"
    assert prepared.backend_entities["service_name"] == "ФГДС с седацией"
    assert "service_variant" not in prepared.backend_entities


def test_adapter_normalizes_price_entities_used_to_ft_shape():
    adapter = FreeTalkAdapter()
    prepared = adapter.prepare_tool_call(
        tool_name="price_info",
        user_message="Сколько стоит ФГДС с седацией?",
        entities={
            "service_name": "ФГДС",
            "service_variant": "с седацией",
            "doctor_name": "Дразнин Антон Владимирович",
        },
    )

    result = adapter.normalize_tool_payload(
        tool_name="price_info",
        payload={
            "prices": [{"serviceName": "ФГДС с седацией", "cost": 5200}],
            "entities_used": {
                "service_name_effective": "ФГДС с седацией",
                "doctor_name_resolved": "Дразнин Антон Владимирович",
            },
            "note": "price_info",
        },
        prepared_call=prepared,
    )

    assert result.ft_payload["entities_used_ft"]["service_name"] == "ФГДС с седацией"
    assert result.ft_payload["entities_used_ft"]["doctor_name"] == "Дразнин Антон Владимирович"
    assert result.ft_payload["adapter_meta"]["domain"] == "service_query"


def test_adapter_keeps_test_name_for_test_assist_calls():
    adapter = FreeTalkAdapter()

    prepared = adapter.prepare_tool_call(
        tool_name="test_assist",
        user_message="Какие анализы на холестерин есть?",
        entities={"test_name": "холестерин"},
    )

    assert prepared.ft_entities["test_name"] == "холестерин"
    assert prepared.backend_entities["test_name"] == "холестерин"
    assert "service_name" not in prepared.backend_entities


def test_agent_uses_adapter_to_send_composed_service_name_to_backend():
    services = ServiceAdapterServices()
    agent = ServiceAdapterAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=InMemoryMemory(),  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )

    reply = asyncio.run(agent.chat("Сколько стоит ФГДС с седацией?", "adapter_service"))

    assert reply.tool_name == "price_info"
    assert services.last_entities["service_name"] == "ФГДС с седацией"
    assert services.last_entities["doctor_name"] == "Дразнин Антон Владимирович"
    assert "service_variant" not in services.last_entities
    assert reply.tool_payload["entities_used_ft"]["service_name"] == "ФГДС с седацией"


class AddressAdapterServices:
    def __init__(self) -> None:
        self.last_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": ""}

    async def address_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        return {
            "addresses": ["г. Самара, пр. Ленина, 5"],
            "branches": [{"address": "г. Самара, пр. Ленина, 5", "city": "Самара"}],
            "entities_used": {
                "branch": "Ленина",
                "city": "Самара",
                "service_name": "ФГДС с седацией",
            },
            "note": "address_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {"address_info": self.address_info}


class AddressAdapterAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = user_message, context, dialog_state, remembered_doctor
        return ClinicalDecision(
            intent="address",
            confidence=0.95,
            entities={
                "branch_name": "Ленина",
                "city": "Самара",
                "service_name": "ФГДС",
                "service_variant": "с седацией",
            },
            missing_slots=[],
            clarify_question="",
            tool_plan=["address_info"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


def test_adapter_prepares_address_call_with_branch_mapping():
    adapter = FreeTalkAdapter()

    prepared = adapter.prepare_tool_call(
        tool_name="address_info",
        user_message="Где на Ленина делают ФГДС с седацией?",
        entities={
            "branch_name": "Ленина",
            "city": "Самара",
            "service_name": "ФГДС",
            "service_variant": "с седацией",
        },
    )

    assert prepared.ft_entities["branch_name"] == "Ленина"
    assert prepared.ft_entities["service_name"] == "ФГДС с седацией"
    assert prepared.backend_entities["branch"] == "Ленина"
    assert prepared.backend_entities["city"] == "Самара"
    assert prepared.backend_entities["service_name"] == "ФГДС с седацией"
    assert "branch_name" not in prepared.backend_entities


def test_adapter_normalizes_address_entities_used_to_ft_shape():
    adapter = FreeTalkAdapter()
    prepared = adapter.prepare_tool_call(
        tool_name="address_info",
        user_message="Где на Ленина делают ФГДС с седацией?",
        entities={
            "branch_name": "Ленина",
            "city": "Самара",
            "service_name": "ФГДС",
            "service_variant": "с седацией",
        },
    )

    result = adapter.normalize_tool_payload(
        tool_name="address_info",
        payload={
            "addresses": ["г. Самара, пр. Ленина, 5"],
            "entities_used": {
                "branch": "Ленина",
                "city": "Самара",
                "service_name": "ФГДС с седацией",
            },
            "note": "address_info",
        },
        prepared_call=prepared,
    )

    assert result.ft_payload["entities_used_ft"]["branch_name"] == "Ленина"
    assert result.ft_payload["entities_used_ft"]["city"] == "Самара"
    assert result.ft_payload["entities_used_ft"]["service_name"] == "ФГДС с седацией"
    assert result.ft_payload["adapter_meta"]["domain"] == "address_query"


def test_agent_uses_adapter_to_send_address_entities_to_backend():
    services = AddressAdapterServices()
    agent = AddressAdapterAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=InMemoryMemory(),  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )

    reply = asyncio.run(agent.chat("Где на Ленина делают ФГДС с седацией?", "adapter_address"))

    assert reply.tool_name == "address_info"
    assert services.last_entities["branch"] == "Ленина"
    assert services.last_entities["city"] == "Самара"
    assert services.last_entities["service_name"] == "ФГДС с седацией"
    assert reply.tool_payload["entities_used_ft"]["branch_name"] == "Ленина"
