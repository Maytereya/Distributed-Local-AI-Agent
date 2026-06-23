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
from localragagent.freetalk.appointment_policy import apply_appointment_precheck
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.contracts import DialogState, SessionContext
from localragagent.freetalk.routing_contract import ClinicalDecision
from localragagent.freetalk.signal_parsers import extract_contextual_entities


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


class ArbiterAppointmentAgent(AppointmentAgent):
    async def _llm_json(self, prompt: str) -> dict[str, object]:
        text = str(prompt or "").lower()
        if "interrupt/topic-switch arbiter" in text and "ладно, другой вопрос" in text:
            return {"decision": "switch", "reason": "ambiguous topic change"}
        return await super()._llm_json(prompt)


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


def _appointment_dialog_state(
    *,
    phase: str = "appointment_confirm",
    entities: dict[str, object] | None = None,
    missing_slots: list[str] | None = None,
    open_question: str = "",
) -> DialogState:
    slots = list(missing_slots or [])
    flow_stage = "confirm" if "confirm" in phase else "collecting"
    return DialogState(
        route="clinical",
        intent="appointment",
        entities=dict(entities or {}),
        missing_slots=slots,
        phase=phase,
        open_question=open_question,
        flow_active=True,
        flow_kind="appointment",
        flow_stage=flow_stage,
        flow_interruptible=True,
        flow_resume_question=open_question,
        expected_slots=slots,
    )


def test_appointment_confirmation_time_repair_updates_only_time():
    text = "Нет, не 09:00, а 11:00"
    result = apply_appointment_precheck(
        user_message=text,
        dialog_state=_appointment_dialog_state(
            entities={
                "appointment_action": "book",
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "date_from": "2026-04-16",
                "date_to": "2026-04-16",
                "time": "09:00",
                "time_from": "09:00",
                "branch_name": "г. Самара, пр. Ленина, 5",
                "patient_name": "Иванов Иван Иванович",
                "appointment_windows": [
                    {"date": "2026-04-16", "time": "09:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                    {"date": "2026-04-16", "time": "11:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                ],
            },
            open_question="Продолжить запись?",
        ),
        memory_entities={},
        contextual_entities=extract_contextual_entities(text),
    )

    assert result.handled is True
    assert result.next_state is not None
    assert result.next_state.phase == "appointment_confirm"
    assert result.next_state.entities["time"] == "11:00"
    assert result.next_state.entities["doctor_name"] == "Трубин Алексей Юрьевич"
    assert result.next_state.entities["patient_name"] == "Иванов Иван Иванович"
    assert "11:00" in result.reply_text


def test_appointment_unavailable_time_keeps_flow_and_offers_cached_alternatives():
    text = "Нет, в 19:00"
    result = apply_appointment_precheck(
        user_message=text,
        dialog_state=_appointment_dialog_state(
            entities={
                "appointment_action": "book",
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "date_from": "2026-04-16",
                "date_to": "2026-04-16",
                "time": "09:00",
                "time_from": "09:00",
                "branch_name": "г. Самара, пр. Ленина, 5",
                "patient_name": "Иванов Иван Иванович",
                "appointment_windows": [
                    {"date": "2026-04-16", "time": "09:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                    {"date": "2026-04-16", "time": "11:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                ],
            },
            open_question="Продолжить запись?",
        ),
        memory_entities={},
        contextual_entities=extract_contextual_entities(text),
    )

    assert result.handled is True
    assert result.next_state is not None
    low = result.reply_text.lower()
    assert "нет записи" in low
    assert "09:00" in result.reply_text
    assert "11:00" in result.reply_text
    assert result.next_state.phase == "appointment_collecting"
    assert "time" in result.next_state.missing_slots
    assert "time" not in result.next_state.entities
    assert result.next_state.entities["patient_name"] == "Иванов Иван Иванович"


def test_appointment_time_range_uses_cached_available_window():
    text = "вечером"
    result = apply_appointment_precheck(
        user_message=text,
        dialog_state=_appointment_dialog_state(
            phase="appointment_collecting",
            missing_slots=["time"],
            entities={
                "appointment_action": "book",
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "date_from": "2026-04-16",
                "date_to": "2026-04-16",
                "branch_name": "г. Самара, пр. Ленина, 5",
                "patient_name": "Иванов Иван Иванович",
                "appointment_windows": [
                    {"date": "2026-04-16", "time": "11:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                    {"date": "2026-04-16", "time": "17:30", "branch_name": "г. Самара, пр. Ленина, 5"},
                ],
            },
            open_question="Уточните время.",
        ),
        memory_entities={},
        contextual_entities=extract_contextual_entities(text),
    )

    assert result.handled is True
    assert result.next_state is not None
    assert result.next_state.phase == "appointment_confirm"
    assert result.next_state.entities["time"] == "17:30"
    assert "17:30" in result.reply_text


def test_appointment_doctor_repair_keeps_flow_without_old_window_context():
    text = "Нет, не Трубин, а Дразнин"
    result = apply_appointment_precheck(
        user_message=text,
        dialog_state=_appointment_dialog_state(
            entities={
                "appointment_action": "book",
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "date_from": "2026-04-16",
                "date_to": "2026-04-16",
                "time": "09:00",
                "time_from": "09:00",
                "branch_name": "г. Самара, пр. Ленина, 5",
                "patient_name": "Иванов Иван Иванович",
                "appointment_windows": [
                    {"date": "2026-04-16", "time": "09:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                ],
            },
            open_question="Продолжить запись?",
        ),
        memory_entities={},
        contextual_entities=extract_contextual_entities(text),
    )

    assert result.handled is True
    assert result.next_state is not None
    assert result.next_state.phase == "appointment_collecting"
    assert result.next_state.entities["doctor_name"] == "Дразнин"
    assert result.next_state.entities["patient_name"] == "Иванов Иван Иванович"
    assert "appointment_windows" not in result.next_state.entities
    assert "date" in result.next_state.missing_slots
    assert "time" in result.next_state.missing_slots


def test_appointment_patient_name_repair_keeps_selected_window():
    text = "Нет, ФИО Петров Петр Петрович"
    result = apply_appointment_precheck(
        user_message=text,
        dialog_state=_appointment_dialog_state(
            entities={
                "appointment_action": "book",
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "date_from": "2026-04-16",
                "date_to": "2026-04-16",
                "time": "09:00",
                "time_from": "09:00",
                "branch_name": "г. Самара, пр. Ленина, 5",
                "patient_name": "Иванов Иван Иванович",
                "appointment_windows": [
                    {"date": "2026-04-16", "time": "09:00", "branch_name": "г. Самара, пр. Ленина, 5"},
                ],
            },
            open_question="Продолжить запись?",
        ),
        memory_entities={},
        contextual_entities=extract_contextual_entities(text),
    )

    assert result.handled is True
    assert result.next_state is not None
    assert result.next_state.phase == "appointment_confirm"
    assert result.next_state.entities["patient_name"] == "Петров Петр Петрович"
    assert result.next_state.entities["time"] == "09:00"
    assert "Петров Петр Петрович" in result.reply_text


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
    assert str(state2["entities"]["date"]).endswith("-04-16")
    assert state2["entities"]["time"] == "09:00"
    assert state2["flow_active"] is True
    assert state2["flow_kind"] == "appointment"
    assert state2["flow_stage"] == "collecting"
    assert state2["expected_slots"] == ["patient_name"]

    reply3 = asyncio.run(agent.chat("Иванов Иван Иванович", session_id))
    assert "16 апреля" in reply3.text
    assert "09:00" in reply3.text
    assert "Трубин Алексей Юрьевич" in reply3.text

    state3 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state3["phase"] == "appointment_confirm"
    assert state3["flow_active"] is True
    assert state3["flow_kind"] == "appointment"
    assert state3["flow_stage"] == "confirm"
    assert state3["expected_slots"] == []

    reply4 = asyncio.run(agent.chat("да", session_id))
    assert "зафиксирован" in reply4.text.lower()
    assert reply4.next_session_id
    assert reply4.next_session_id != session_id
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
    assert asyncio.run(memory.get_turn_count(session_id)) == 0


def test_active_appointment_topic_switch_uses_global_confirm_and_resumes():
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
    assert "прервать текущий сценарий" in reply.text.lower()

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "interrupt_confirm_topic_switch"

    resume = asyncio.run(agent.chat("нет", session_id))
    assert "фио" in resume.text.lower()

    resumed_state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert resumed_state["phase"] == "appointment_collecting"
    assert resumed_state["missing_slots"] == ["patient_name"]


def test_active_appointment_topic_switch_yes_reenters_new_question_immediately():
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
    session_id = "appointment_topic_switch_yes"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    confirm = asyncio.run(agent.chat("Скажите стоимость общего анализа крови", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    reply = asyncio.run(agent.chat("да", session_id))
    assert reply.tool_name == "price_info"
    assert "нашел цены" in reply.text.lower()
    state = asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", ""))
    assert state == ""


def test_active_appointment_no_preference_picks_earliest_slot_and_moves_to_patient_name():
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
    session_id = "appointment_no_preference"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    start = asyncio.run(agent.chat("Мне надо записаться к нему", session_id))
    assert "дату и время" in start.text.lower()

    reply = asyncio.run(agent.chat("без разницы", session_id))
    assert "фио" in reply.text.lower()

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "appointment_collecting"
    assert state["entities"]["doctor_name"] == "Трубин Алексей Юрьевич"
    assert state["entities"]["date"] == "2026-04-15"
    assert state["entities"]["time"] == "08:30"
    assert state["missing_slots"] == ["patient_name"]


def test_active_appointment_uncertainty_keeps_patient_name_step():
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
    session_id = "appointment_uncertainty"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    reply = asyncio.run(agent.chat("не знаю", session_id))
    low = reply.text.lower()
    assert "фио" in low
    assert "для записи нужно" in low

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "appointment_collecting"
    assert state["missing_slots"] == ["patient_name"]


def test_active_appointment_repeated_uncertainty_escalates_to_handoff():
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
    session_id = "appointment_repeated_uncertainty"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    first = asyncio.run(agent.chat("не знаю", session_id))
    assert "фио" in first.text.lower()

    second = asyncio.run(agent.chat("не помню", session_id))
    low_second = second.text.lower()
    assert "оператор" in low_second or "передам диалог оператору" in low_second

    third = asyncio.run(agent.chat("все равно не помню", session_id))
    assert "оператор" in third.text.lower()
    assert third.next_session_id
    assert third.next_session_id != session_id
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""


def test_mixed_appointment_date_and_topic_switch_preserves_selected_date_until_confirm():
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
    session_id = "appointment_mixed_topic_switch"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему", session_id))

    confirm = asyncio.run(agent.chat("На 17 мая, а сколько стоит общий анализ крови?", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    resume = asyncio.run(agent.chat("нет", session_id))
    assert "время" in resume.text.lower()

    resumed_state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert resumed_state["intent"] == "appointment"
    assert resumed_state["entities"]["date"].endswith("-05-17")
    assert "time" in resumed_state["missing_slots"]


def test_mixed_appointment_date_and_topic_switch_yes_reenters_price_question():
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
    session_id = "appointment_mixed_topic_switch_yes"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему", session_id))
    asyncio.run(agent.chat("На 17 мая, а сколько стоит общий анализ крови?", session_id))

    reply = asyncio.run(agent.chat("да", session_id))
    assert reply.tool_name == "price_info"
    assert "нашел цены" in reply.text.lower()


def test_appointment_policy_does_not_own_general_topic_switch_anymore():
    result = apply_appointment_precheck(
        user_message="Сколько стоит общий анализ крови?",
        dialog_state=DialogState(
            route="clinical",
            intent="appointment",
            entities={
                "doctor_name": "Трубин Алексей Юрьевич",
                "date": "2026-04-16",
                "time": "09:00",
            },
            missing_slots=["patient_name"],
            phase="appointment_collecting",
            open_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            flow_active=True,
            flow_kind="appointment",
            flow_stage="collecting",
            flow_interruptible=True,
            flow_resume_question="Сообщите, пожалуйста, ваше ФИО для записи.",
            expected_slots=["patient_name"],
        ),
        memory_entities={
            "doctor_name": "Трубин Алексей Юрьевич",
            "appointment_windows": [{"date": "2026-04-16", "time": "09:00", "branch_name": "г. Самара, пр. Ленина, 5"}],
        },
        contextual_entities={},
    )
    assert result.handled is False


def test_ambiguous_active_appointment_uses_llm_arbiter_for_topic_switch():
    memory = InMemoryMemory()
    services = AppointmentServices()
    agent = ArbiterAppointmentAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "appointment_arbiter_switch"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    reply = asyncio.run(agent.chat("Ладно, другой вопрос", session_id))
    assert "прервать текущий сценарий" in reply.text.lower()
    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "interrupt_confirm_topic_switch"


def test_active_appointment_stop_uses_global_interrupt_and_clears_flow():
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
    session_id = "appointment_interrupt_yes"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    stopped = asyncio.run(agent.chat("неправильно, стоп", session_id))
    assert "диалог очищен" in stopped.text.lower()
    assert stopped.next_session_id
    assert stopped.next_session_id != session_id
    assert asyncio.run(memory.get_turn_count(session_id)) == 0
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_entity_memory", "")) == ""
    assert asyncio.run(memory.get_meta_str(session_id, "last_doctor_name", "")) == ""


def test_active_appointment_feedback_resets_without_resume_confirmation():
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
    session_id = "appointment_interrupt_no"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))
    asyncio.run(agent.chat("Мне надо записаться к нему. На 16.04, 09:00", session_id))

    reset = asyncio.run(agent.chat("это бред", session_id))
    assert "сбрасываю" in reset.text.lower()
    assert reset.next_session_id
    assert reset.next_session_id != session_id
    assert asyncio.run(memory.get_turn_count(session_id)) == 0
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_entity_memory", "")) == ""


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


def test_hard_reset_session_clears_session_and_rotates_id():
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
    session_id = "interrupt_hard_reset"

    asyncio.run(agent.chat("А есть расписание работы Трубина?", session_id))

    cleared = asyncio.run(agent.chat("очисти диалог", session_id))
    assert "диалог очищен" in cleared.text.lower()
    assert cleared.next_session_id
    assert cleared.next_session_id != session_id
    assert asyncio.run(memory.get_turn_count(session_id)) == 0
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""
