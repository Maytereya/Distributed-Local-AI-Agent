import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.agent import FreeTalkAgent
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.contracts import DialogState, SessionContext
from localragagent.freetalk.routing_contract import ClinicalDecision


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
        turns = list(self.turns.get(sid, []))
        tail = turns[-history_tail_turns:] if history_tail_turns > 0 else turns
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
        raw = self.meta.get(str(session_id or "").strip(), {}).get(str(key or "").strip(), default)
        try:
            return int(raw)
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


class AppointmentServices:
    def __init__(self) -> None:
        self.schedule_calls: list[dict[str, object]] = []

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        probe = str(raw_text_or_name or "").lower()
        if "трубин" in probe:
            return {"status": "exact", "canonical": "Трубин Алексей Юрьевич", "query": raw_text_or_name}
        if "дразнин" in probe:
            return {"status": "exact", "canonical": "Дразнин Антон Владимирович", "query": raw_text_or_name}
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def doctors_schedule_week(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.schedule_calls.append(dict(entities))
        doctor_name = str(entities.get("doctor_name") or "").strip()
        if "Дразнин" in doctor_name:
            return {
                "schedule": [
                    {
                        "fio": "Дразнин Антон Владимирович",
                        "schedule": {
                            "г. Самара, пр. Ленина, 5": [
                                {
                                    "date": "2026-04-20",
                                    "slots": ["15:30", "16:00", "17:00"],
                                    "start": "15:30",
                                    "end": "18:00",
                                }
                            ]
                        },
                    }
                ],
                "entities_used": {"doctor_name_resolved": "Дразнин Антон Владимирович"},
                "schedule_unavailable_reason": "",
                "note": "doctors_schedule_week",
            }
        return {
            "schedule": [
                {
                    "fio": "Трубин Алексей Юрьевич",
                    "schedule": {
                        "г. Самара, пр. Ленина, 5": [
                            {
                                "date": "2026-04-15",
                                "slots": ["08:30", "11:00", "12:00"],
                                "start": "08:30",
                                "end": "13:30",
                            },
                            {
                                "date": "2026-04-16",
                                "slots": ["09:00", "09:30", "10:00"],
                                "start": "09:00",
                                "end": "11:30",
                            },
                        ]
                    },
                }
            ],
            "entities_used": {"doctor_name_resolved": "Трубин Алексей Юрьевич"},
            "schedule_unavailable_reason": "",
            "note": "doctors_schedule_week",
        }

    async def doctors_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query, entities
        doctor_name = str(entities.get("doctor_name") or "").strip()
        if "Дразнин" in doctor_name:
            return {
                "doctors": [{"fio": "Дразнин Антон Владимирович", "specialization": "Уролог"}],
                "entities_used": {"doctor_name_resolved": "Дразнин Антон Владимирович"},
                "note": "doctors_info",
            }
        return {
            "doctors": [{"fio": "Трубин Алексей Юрьевич", "specialization": "Уролог"}],
            "entities_used": {"doctor_name_resolved": "Трубин Алексей Юрьевич"},
            "note": "doctors_info",
        }

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query, entities
        return {
            "prices": [{"serviceName": "Общий анализ крови", "cost": 650}],
            "entities_used": {"service_name_effective": "Общий анализ крови"},
            "note": "price_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "doctors_schedule_week": self.doctors_schedule_week,
            "doctors_info": self.doctors_info,
            "price_info": self.price_info,
        }


class AppointmentAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = context, dialog_state, remembered_doctor
        text = str(user_message or "").strip().lower()
        if "стоим" in text or "анализ" in text:
            return ClinicalDecision(
                intent="price",
                confidence=0.95,
                entities={"service_name": "Общий анализ крови"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["price_info"],
                source="test",
            )
        if "распис" in text or "трубин" in text:
            return ClinicalDecision(
                intent="doctor_schedule",
                confidence=0.95,
                entities={},
                missing_slots=[],
                clarify_question="",
                tool_plan=["doctors_schedule_week", "doctors_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.1,
            entities={},
            missing_slots=[],
            clarify_question="",
            tool_plan=[],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


class ToolHandoffServices:
    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def service_bundle_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query, entities
        return {
            "handoff_required": True,
            "handoff_message": "В моей базе данных информации недостаточно. Передаю диалог оператору.",
            "note": "handoff_required",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "service_bundle_info": self.service_bundle_info,
        }


class ToolHandoffAgent(FreeTalkAgent):
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
            intent="service_info",
            confidence=0.95,
            entities={"service_name": "Неизвестная услуга"},
            missing_slots=[],
            clarify_question="",
            tool_plan=["service_bundle_info"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


def test_schedule_to_appointment_flow_uses_slot_and_finishes_with_handoff():
    memory = InMemoryMemory()
    services = AppointmentServices()
    agent = AppointmentAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "appointment_flow"

    reply1 = asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    assert reply1.tool_name == "doctors_schedule_week"
    assert "15 апреля" in reply1.text
    assert "выберите дату и время" in reply1.text.lower()
    assert len(services.schedule_calls) == 1

    reply2 = asyncio.run(agent.chat("Нет, мне надо записаться к нему. На 16.04, 09:00", session_id))
    assert "фио" in reply2.text.lower()
    assert len(services.schedule_calls) == 1

    state2 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state2["intent"] == "appointment"
    assert state2["missing_slots"] == ["patient_name"]
    assert state2["entities"]["doctor_name"] == "Трубин Алексей Юрьевич"
    assert state2["entities"]["date"] == "2026-04-16"
    assert state2["entities"]["time"] == "09:00"

    reply3 = asyncio.run(agent.chat("Иванов Иван Иванович", session_id))
    assert "16 апреля" in reply3.text
    assert "09:00" in reply3.text
    assert "Трубин Алексей Юрьевич" in reply3.text

    state3 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state3["phase"] == "appointment_confirm"

    reply4 = asyncio.run(agent.chat("да", session_id))
    assert "зафиксирован" in reply4.text.lower()
    assert reply4.next_session_id
    assert reply4.next_session_id != session_id
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
    assert asyncio.run(memory.get_turn_count(session_id)) == 0


def test_active_appointment_topic_switch_requests_cancel_confirmation_and_resumes():
    memory = InMemoryMemory()
    services = AppointmentServices()
    agent = AppointmentAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "appointment_guardrail"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    reply = asyncio.run(agent.chat("Скажите стоимость общего анализа крови", session_id))
    assert "прекратить запись" in reply.text.lower()

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "appointment_cancel_confirm"

    resume = asyncio.run(agent.chat("нет", session_id))
    assert "фио" in resume.text.lower()

    resumed_state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert resumed_state["phase"] == "appointment_collecting"
    assert resumed_state["missing_slots"] == ["patient_name"]


def test_active_appointment_schedule_request_returns_cached_schedule_preview():
    memory = InMemoryMemory()
    services = AppointmentServices()
    agent = AppointmentAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "appointment_schedule_preview"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    reply = asyncio.run(agent.chat("Дай пожалуйста расписание врача", session_id))
    low = reply.text.lower()
    assert "нашел расписание" in low
    assert "16 апреля" in low
    assert "09:00" in low
    assert "выберите дату и время" in low
    assert "сообщите, пожалуйста, ваше фио" not in low

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["intent"] == "appointment"
    assert state["phase"] == "appointment_collecting"
    assert state["missing_slots"] == ["patient_name"]


def test_new_appointment_after_handoff_does_not_reuse_old_slot_context():
    memory = InMemoryMemory()
    services = AppointmentServices()
    agent = AppointmentAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "appointment_new_doctor_after_handoff"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))
    asyncio.run(agent.chat("Рахманов Вахоб Дмитриевич", session_id))
    finish = asyncio.run(agent.chat("Да", session_id))
    assert "зафиксирован" in finish.text.lower()

    price_reply = asyncio.run(agent.chat("Спасибо. Скажи пожалуйста, сколько стоит общий анализ мочи в клинике?", session_id))
    assert price_reply.tool_name == "price_info"
    assert "нашел цены" in price_reply.text.lower()

    new_appointment = asyncio.run(agent.chat("Можно еще записаться к врачу Дразнину?", session_id))
    low = new_appointment.text.lower()
    assert "нашел расписание" in low
    assert "дразнин" in low
    assert "20 апреля" in low
    assert "выберите, пожалуйста, дату и время для записи" not in low


def test_tool_handoff_uses_global_session_reset():
    memory = InMemoryMemory()
    agent = ToolHandoffAgent(
        config=_cfg(),
        services=ToolHandoffServices(),  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "tool_handoff_reset"

    reply = asyncio.run(agent.chat("Расскажи подробнее про неизвестную услугу", session_id))

    assert "передаю диалог оператору" in reply.text.lower()
    assert reply.next_session_id
    assert reply.next_session_id != session_id
    assert asyncio.run(memory.get_turn_count(session_id)) == 0
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
