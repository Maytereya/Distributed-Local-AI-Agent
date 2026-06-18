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
from localragagent.freetalk.routing_contract import ClinicalDecision
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.contracts import DialogState, SessionContext
from localragagent.freetalk.dialog_state import dialog_state_from_payload, dialog_state_payload


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
        assistant_turn: dict[str, object] = {
            "role": "assistant",
            "content": str(assistant_text or ""),
            "source": str(source or ""),
        }
        if source_fragments:
            assistant_turn["source_fragments"] = list(source_fragments)
        bucket.append(assistant_turn)

    async def get_turn_count(self, session_id: str) -> int:
        sid = str(session_id or "").strip()
        return len(self.turns.get(sid, []))

    async def save_summary(self, session_id: str, summary: str) -> None:
        self.summaries[str(session_id or "").strip()] = str(summary or "")

    async def get_meta_int(self, session_id: str, key: str, default: int = 0) -> int:
        sid = str(session_id or "").strip()
        raw = self.meta.get(sid, {}).get(str(key or "").strip(), "")
        try:
            return int(raw)
        except Exception:
            return int(default)

    async def set_meta_int(self, session_id: str, key: str, value: int) -> None:
        sid = str(session_id or "").strip()
        self.meta.setdefault(sid, {})[str(key or "").strip()] = str(int(value))

    async def get_meta_str(self, session_id: str, key: str, default: str = "") -> str:
        sid = str(session_id or "").strip()
        return str(self.meta.get(sid, {}).get(str(key or "").strip(), default) or "")

    async def set_meta_str(self, session_id: str, key: str, value: str) -> None:
        sid = str(session_id or "").strip()
        self.meta.setdefault(sid, {})[str(key or "").strip()] = str(value or "")

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


class ResultServices:
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
        required = ("surname", "year", "filial", "number")
        missing = [key for key in required if not str(entities.get(key) or "").strip()]
        if missing:
            labels = {
                "surname": "фамилия",
                "year": "год рождения",
                "filial": "филиал",
                "number": "номер заказа",
            }
            return {
                "ready": False,
                "missing_fields": [labels[key] for key in missing],
                "note": "test_result_status",
            }
        return {
            "ready": True,
            "result_links": ["https://example.org/result/12345"],
            "note": "test_result_status",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {"test_result_status": self.test_result_status}


class ResultDialogAgent(FreeTalkAgent):
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
        if "результат" in text:
            return ClinicalDecision(
                intent="test_result",
                confidence=0.93,
                entities={},
                missing_slots=[
                    "result_surname",
                    "result_year_of_birth",
                    "result_analysis_code",
                    "result_analysis_number",
                ],
                clarify_question="Для проверки результата уточните: фамилию пациента, год рождения, код анализа, номер анализа.",
                tool_plan=["test_result_status"],
                source="test",
            )
        if text == "иванов":
            return ClinicalDecision(
                intent="test_result",
                confidence=0.93,
                entities={"result_surname": "Иванов"},
                missing_slots=[
                    "result_year_of_birth",
                    "result_analysis_code",
                    "result_analysis_number",
                ],
                clarify_question="Для проверки результата уточните: год рождения, код анализа, номер анализа.",
                tool_plan=["test_result_status"],
                source="test",
            )
        return ClinicalDecision(
            intent="test_result",
            confidence=0.93,
            entities={
                "result_year_of_birth": "1990",
                "result_analysis_code": "Бг",
                "result_analysis_number": "12345",
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


class TerminalResultAgent(FreeTalkAgent):
    async def _route_clinical_decision(
        self,
        *,
        user_message: str,
        context: SessionContext,
        dialog_state: DialogState,
        remembered_doctor: str,
    ) -> ClinicalDecision:
        _ = context, dialog_state, remembered_doctor
        calls = int(getattr(self, "route_calls", 0)) + 1
        setattr(self, "route_calls", calls)
        if calls > 1:
            raise AssertionError("complete result lookup must not re-enter clinical router")
        return ClinicalDecision(
            intent="test_result",
            confidence=0.93,
            entities={},
            missing_slots=[
                "result_surname",
                "result_year_of_birth",
                "result_analysis_code",
                "result_analysis_number",
            ],
            clarify_question="Для проверки результата уточните: фамилию пациента, год рождения, код анализа, номер анализа.",
            tool_plan=["test_result_status"],
            source="test",
        )

    async def _llm_json(self, prompt: str) -> dict[str, object]:
        _ = prompt
        return {}

    async def _llm_text(self, prompt: str) -> str:
        _ = prompt
        return ""


class PartialTupleResultAgent(FreeTalkAgent):
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
        if "результат" in text:
            return ClinicalDecision(
                intent="test_result",
                confidence=0.93,
                entities={},
                missing_slots=[
                    "result_surname",
                    "result_year_of_birth",
                    "result_analysis_code",
                    "result_analysis_number",
                ],
                clarify_question="Для проверки результата уточните: фамилию пациента, год рождения, код анализа, номер анализа.",
                tool_plan=["test_result_status"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.15,
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


class MixedResultServices(ResultServices):
    async def doctors_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query, entities
        return {
            "doctors": [
                {"fio": "Трубин Алексей Юрьевич", "specialization": "Уролог"},
                {"fio": "Вахобов Абдуджалол Нозимович", "specialization": "Уролог"},
            ],
            "note": "doctors_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        handlers = super().tool_handlers(include_meili_tools=include_meili_tools)
        handlers["doctors_info"] = self.doctors_info
        return handlers


class MixedResultAgent(FreeTalkAgent):
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
        if "результат" in text:
            return ClinicalDecision(
                intent="test_result",
                confidence=0.93,
                entities={},
                missing_slots=[
                    "result_surname",
                    "result_year_of_birth",
                    "result_analysis_code",
                    "result_analysis_number",
                ],
                clarify_question="Для проверки результата уточните: фамилию пациента, год рождения, код анализа, номер анализа.",
                tool_plan=["test_result_status"],
                source="test",
            )
        if "уролог" in text:
            return ClinicalDecision(
                intent="doctor_info",
                confidence=0.95,
                entities={"specialty": "уролог"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["doctors_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.15,
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


def test_dialog_state_accumulates_result_slots_across_turns():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = ResultDialogAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_dialog"

    reply1 = asyncio.run(agent.chat("Проверь результат анализа", session_id))
    assert "фамилию пациента" in reply1.text.lower()

    state1 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state1["intent"] == "test_result"
    assert set(state1["missing_slots"]) == {
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    }
    assert state1["flow_active"] is True
    assert state1["flow_kind"] == "result_lookup"
    assert state1["flow_stage"] == "collecting"
    assert set(state1["expected_slots"]) == {
        "result_surname",
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    }

    reply2 = asyncio.run(agent.chat("Иванов", session_id))
    assert "год рождения" in reply2.text.lower()
    state2 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state2["flow_active"] is True
    assert state2["flow_kind"] == "result_lookup"
    assert state2["flow_stage"] == "collecting"
    assert set(state2["expected_slots"]) == {
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    }

    state2 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state2["entities"]["result_surname"] == "Иванов"
    assert set(state2["missing_slots"]) == {
        "result_year_of_birth",
        "result_analysis_code",
        "result_analysis_number",
    }

    reply3 = asyncio.run(agent.chat("1990, Бг, 12345", session_id))
    assert reply3.tool_name == "test_result_status"
    assert "результат готов" in reply3.text.lower()
    assert services.last_entities["surname"] == "Иванов"
    assert services.last_entities["year"] == "1990"
    assert services.last_entities["filial"] == "Бг"
    assert services.last_entities["number"] == "12345"
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""


def test_result_lookup_complete_tuple_executes_tool_without_router_reentry():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = TerminalResultAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_terminal_without_router"

    reply1 = asyncio.run(agent.chat("Проверь результат анализа", session_id))
    assert "фамилию пациента" in reply1.text.lower()

    reply2 = asyncio.run(agent.chat("Иванов, 1990, Бг, 12345", session_id))
    assert reply2.tool_name == "test_result_status"
    assert "результат готов" in reply2.text.lower()
    assert getattr(agent, "route_calls", 0) == 1
    assert services.last_entities["surname"] == "Иванов"
    assert services.last_entities["year"] == "1990"
    assert services.last_entities["filial"] == "Бг"
    assert services.last_entities["number"] == "12345"


def test_operator_request_bypasses_guard_and_active_slot_flow():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = ResultDialogAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "operator_global_control"
    state = DialogState(
        route="clinical",
        intent="price",
        missing_slots=["service_or_analysis_name"],
        phase="collecting",
        open_question="Уточните услугу.",
        flow_active=True,
        flow_kind="clarify",
        flow_stage="collecting",
        flow_interruptible=True,
        expected_slots=["service_or_analysis_name"],
    )
    asyncio.run(
        memory.set_meta_str(
            session_id,
            "clinical_dialog_state",
            json.dumps(dialog_state_payload(state), ensure_ascii=False),
        )
    )
    asyncio.run(memory.set_meta_str(session_id, "ctx_guard_state", "awaiting_immediate"))

    reply = asyncio.run(agent.chat("дай оператора", session_id))

    assert reply.handoff is True
    assert reply.next_session_id
    assert "оператор" in reply.text.lower()
    assert session_id not in memory.meta


def test_negative_feedback_resets_even_without_active_flow():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = ResultDialogAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "negative_feedback_global_control"
    asyncio.run(memory.set_meta_str(session_id, "clinical_entity_memory", "{\"doctor_name\":\"Дразнин\"}"))

    reply = asyncio.run(agent.chat("ты несешь бред", session_id))

    assert reply.next_session_id
    assert "сбрасываю" in reply.text.lower()
    assert session_id not in memory.meta


def test_flow_local_partial_result_tuple_accumulates_without_llm_second_turn():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = PartialTupleResultAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_partial_tuple_dialog"

    reply1 = asyncio.run(agent.chat("Проверь результат анализа", session_id))
    assert "фамилию пациента" in reply1.text.lower()

    reply2 = asyncio.run(agent.chat("Иванов, 1990", session_id))
    assert "код анализа" in reply2.text.lower()
    assert "номер анализа" in reply2.text.lower()

    state2 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state2["entities"]["result_surname"] == "Иванов"
    assert state2["entities"]["result_year_of_birth"] == "1990"
    assert set(state2["missing_slots"]) == {
        "result_analysis_code",
        "result_analysis_number",
    }

    reply3 = asyncio.run(agent.chat("Бг, 12345", session_id))
    assert reply3.tool_name == "test_result_status"
    assert "результат готов" in reply3.text.lower()
    assert services.last_entities["surname"] == "Иванов"
    assert services.last_entities["year"] == "1990"
    assert services.last_entities["filial"] == "Бг"
    assert services.last_entities["number"] == "12345"


def test_result_lookup_repeated_uncertainty_stops_without_handoff():
    memory = InMemoryMemory()
    services = ResultServices()
    agent = PartialTupleResultAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_repeated_uncertainty"

    asyncio.run(agent.chat("Проверь результат анализа", session_id))

    first = asyncio.run(agent.chat("не знаю", session_id))
    assert "точные данные" in first.text.lower()

    second = asyncio.run(agent.chat("не помню", session_id))
    assert "не смогу проверить результат автоматически" in second.text.lower()

    third = asyncio.run(agent.chat("все равно не помню", session_id))
    low = third.text.lower()
    assert "сценарий остановлен" in low
    assert third.next_session_id == ""
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""


def test_mixed_result_tuple_and_topic_switch_preserves_partial_state_until_confirm():
    memory = InMemoryMemory()
    services = MixedResultServices()
    agent = MixedResultAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_mixed_topic_switch"

    asyncio.run(agent.chat("Проверь результат анализа", session_id))

    confirm = asyncio.run(agent.chat("Иванов, 1990, а лучше покажи урологов", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state["phase"] == "interrupt_confirm_topic_switch"

    resume = asyncio.run(agent.chat("нет", session_id))
    low = resume.text.lower()
    assert "код анализа" in low
    assert "номер анализа" in low

    resumed_state = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert resumed_state["entities"]["result_surname"] == "Иванов"
    assert resumed_state["entities"]["result_year_of_birth"] == "1990"
    assert set(resumed_state["missing_slots"]) == {
        "result_analysis_code",
        "result_analysis_number",
    }


def test_mixed_result_tuple_and_topic_switch_yes_reenters_new_question():
    memory = InMemoryMemory()
    services = MixedResultServices()
    agent = MixedResultAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "result_mixed_topic_switch_yes"

    asyncio.run(agent.chat("Проверь результат анализа", session_id))
    asyncio.run(agent.chat("Иванов, 1990, а лучше покажи урологов", session_id))

    reply = asyncio.run(agent.chat("да", session_id))
    assert reply.tool_name == "doctors_info"
    assert "уролог" in reply.text.lower()


def test_dialog_state_payload_roundtrip_preserves_flow_descriptor():
    state = DialogState(
        route="clinical",
        intent="appointment",
        entities={"doctor_name": "Трубин Алексей Юрьевич"},
        missing_slots=["patient_name"],
        phase="appointment_collecting",
        open_question="Сообщите, пожалуйста, ваше ФИО для записи.",
        flow_active=True,
        flow_kind="appointment",
        flow_stage="collecting",
        flow_interruptible=True,
        flow_resume_question="Сообщите, пожалуйста, ваше ФИО для записи.",
        expected_slots=["patient_name"],
        flow_non_answer_count=2,
        flow_non_answer_kind="uncertainty",
    )

    restored = dialog_state_from_payload(dialog_state_payload(state))

    assert restored.flow_active is True
    assert restored.flow_kind == "appointment"
    assert restored.flow_stage == "collecting"
    assert restored.flow_interruptible is True
    assert restored.flow_resume_question == "Сообщите, пожалуйста, ваше ФИО для записи."
    assert restored.expected_slots == ["patient_name"]
    assert restored.flow_non_answer_count == 2
    assert restored.flow_non_answer_kind == "uncertainty"


def test_dialog_state_from_payload_normalizes_missing_slots_from_redis():
    restored = dialog_state_from_payload(
        {
            "route": "clinical",
            "intent": "test_result",
            "missing_slots": ["surname", "birth_year", "made_up_slot"],
            "expected_slots": ["result_filial", "result_number", "unknown_slot"],
            "tool_plan": ["test_result_status"],
        }
    )

    assert restored.missing_slots == ["result_surname", "result_year_of_birth"]
    assert restored.expected_slots == ["result_analysis_code", "result_analysis_number"]


class FuzzyPriceServices:
    def __init__(self) -> None:
        self.last_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        probe = str(raw_text_or_name or current_service_name or "").lower()
        if "общий анлиз крови" in probe:
            return {
                "status": "fuzzy",
                "canonical": "Общий анализ крови",
                "query": raw_text_or_name,
            }
        if "общий анализ крови" in probe:
            return {
                "status": "exact",
                "canonical": "Общий анализ крови",
                "query": raw_text_or_name,
            }
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_entities = dict(entities)
        if str(entities.get("service_name") or "").strip() != "Общий анализ крови":
            return {"prices": [], "note": "price_info"}
        return {
            "prices": [{"serviceName": "Общий анализ крови", "cost": 650}],
            "entities_used": {"service_name_effective": "Общий анализ крови"},
            "note": "price_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {"price_info": self.price_info}


class FuzzyPriceAgent(FreeTalkAgent):
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
        if "стоит" in text:
            return ClinicalDecision(
                intent="price",
                confidence=0.94,
                entities={},
                missing_slots=[],
                clarify_question="",
                tool_plan=["price_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.2,
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


class FuzzyPreparePriceServices:
    def __init__(self) -> None:
        self.last_prepare_entities: dict[str, object] = {}
        self.last_price_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        probe = str(raw_text_or_name or current_service_name or "").lower()
        if "анлиз" in probe:
            return {
                "status": "fuzzy",
                "canonical": "Общий анализ крови",
                "query": raw_text_or_name,
            }
        if "анализ крови" in probe:
            return {
                "status": "exact",
                "canonical": "Общий анализ крови",
                "query": raw_text_or_name,
            }
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        _ = raw_text_or_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def test_prepare(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_prepare_entities = dict(entities)
        return {
            "prepare": "Натощак 8 часов, воду пить можно.",
            "entities_used": {"service_name_effective": "Общий анализ крови"},
            "note": "test_prepare",
        }

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_price_entities = dict(entities)
        return {
            "prices": [{"serviceName": "Общий анализ крови", "cost": 650}],
            "entities_used": {"service_name_effective": "Общий анализ крови"},
            "note": "price_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "test_prepare": self.test_prepare,
            "price_info": self.price_info,
        }


class FuzzyPreparePriceAgent(FreeTalkAgent):
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
        if "подготов" in text:
            return ClinicalDecision(
                intent="prepare",
                confidence=0.94,
                entities={},
                missing_slots=[],
                clarify_question="",
                tool_plan=["test_prepare"],
                source="test",
            )
        if "стоит" in text:
            return ClinicalDecision(
                intent="price",
                confidence=0.94,
                entities={"service_name": "Общий анализ крови"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["price_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.2,
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


class DoctorClarifyMixedServices:
    def __init__(self) -> None:
        self.last_doctor_entities: dict[str, object] = {}
        self.last_price_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        probe = str(raw_text_or_name or "").lower()
        if "суворов" in probe:
            return {
                "status": "exact",
                "canonical": "Суворов Алексей Петрович",
                "query": raw_text_or_name,
            }
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def doctors_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_doctor_entities = dict(entities)
        return {
            "doctors": [{"fio": "Суворов Алексей Петрович", "specialization": "Кардиолог"}],
            "entities_used": {"doctor_name_resolved": "Суворов Алексей Петрович"},
            "note": "doctors_info",
        }

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_price_entities = dict(entities)
        return {
            "prices": [{"serviceName": "Прием врача", "cost": 1200}],
            "entities_used": {"service_name_effective": "Прием врача"},
            "note": "price_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "doctors_info": self.doctors_info,
            "price_info": self.price_info,
        }


class DoctorClarifyMixedAgent(FreeTalkAgent):
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
        if "стоит" in text:
            return ClinicalDecision(
                intent="price",
                confidence=0.95,
                entities={"service_name": "Прием врача"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["price_info"],
                source="test",
            )
        if "суворов" in text:
            return ClinicalDecision(
                intent="doctor_info",
                confidence=0.95,
                entities={"doctor_name": "Суворов Алексей Петрович"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["doctors_info"],
                source="test",
            )
        if "врач" in text or "доктор" in text or "расскажи" in text:
            return ClinicalDecision(
                intent="doctor_info",
                confidence=0.93,
                entities={},
                missing_slots=["doctor_name"],
                clarify_question="Уточните, пожалуйста, какого врача вы имеете в виду.",
                tool_plan=["doctors_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.2,
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


class FuzzyDoctorMixedServices:
    def __init__(self) -> None:
        self.last_doctor_entities: dict[str, object] = {}
        self.last_price_entities: dict[str, object] = {}

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        probe = str(raw_text_or_name or "").lower()
        if "дразнн" in probe:
            return {
                "status": "fuzzy",
                "canonical": "Дразнин Антон Владимирович",
                "query": raw_text_or_name,
            }
        if "суворов" in probe:
            return {
                "status": "exact",
                "canonical": "Суворов Алексей Петрович",
                "query": raw_text_or_name,
            }
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def doctors_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_doctor_entities = dict(entities)
        doctor_name = str(entities.get("doctor_name") or "").strip()
        if "Суворов" in doctor_name:
            fio = "Суворов Алексей Петрович"
        else:
            fio = "Дразнин Антон Владимирович"
        return {
            "doctors": [{"fio": fio, "specialization": "Кардиолог"}],
            "entities_used": {"doctor_name_resolved": fio},
            "note": "doctors_info",
        }

    async def price_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.last_price_entities = dict(entities)
        return {
            "prices": [{"serviceName": "Прием врача", "cost": 1200}],
            "entities_used": {"service_name_effective": "Прием врача"},
            "note": "price_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "doctors_info": self.doctors_info,
            "price_info": self.price_info,
        }


class FuzzyDoctorMixedAgent(FreeTalkAgent):
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
        if "стоит" in text:
            return ClinicalDecision(
                intent="price",
                confidence=0.95,
                entities={"service_name": "Прием врача"},
                missing_slots=[],
                clarify_question="",
                tool_plan=["price_info"],
                source="test",
            )
        if "драз" in text or "суворов" in text or "расскажи" in text or "врач" in text:
            return ClinicalDecision(
                intent="doctor_info",
                confidence=0.95,
                entities={},
                missing_slots=[],
                clarify_question="",
                tool_plan=["doctors_info"],
                source="test",
            )
        return ClinicalDecision(
            intent="unknown",
            confidence=0.2,
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


def test_dialog_state_confirms_fuzzy_service_candidate_before_tool_call():
    memory = InMemoryMemory()
    services = FuzzyPriceServices()
    agent = FuzzyPriceAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "fuzzy_price_dialog"

    reply1 = asyncio.run(agent.chat("Сколько стоит общий анлиз крови?", session_id))
    assert "правильно понял" in reply1.text.lower()
    assert "общий анализ крови" in reply1.text.lower()

    state1 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state1["phase"] == "confirm_candidate"
    assert state1["confirmation_target"] == "service_name"
    assert state1["candidate_entities"]["service_name"] == "Общий анализ крови"

    reply2 = asyncio.run(agent.chat("Да", session_id))
    assert reply2.tool_name == "price_info"
    assert "650" in reply2.text
    assert services.last_entities["service_name"] == "Общий анализ крови"
    assert asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")) == ""


def test_confirmation_no_preference_rejects_candidate_and_returns_to_clarify():
    memory = InMemoryMemory()
    services = FuzzyPriceServices()
    agent = FuzzyPriceAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "fuzzy_price_no_preference_dialog"

    reply1 = asyncio.run(agent.chat("Сколько стоит общий анлиз крови?", session_id))
    assert "правильно понял" in reply1.text.lower()

    reply2 = asyncio.run(agent.chat("Любой", session_id))
    assert "уточните" in reply2.text.lower()

    state2 = json.loads(asyncio.run(memory.get_meta_str(session_id, "clinical_dialog_state", "")))
    assert state2["phase"] == "collecting"
    assert state2["flow_kind"] == "clarify"
    assert state2["confirmation_target"] == ""
    assert state2["candidate_entities"] == {}
    assert state2["missing_slots"] == ["service_or_analysis_name"]


def test_mixed_clarify_slot_answer_and_topic_switch_no_reenters_original_flow():
    memory = InMemoryMemory()
    services = DoctorClarifyMixedServices()
    agent = DoctorClarifyMixedAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "doctor_clarify_mixed_no"

    first = asyncio.run(agent.chat("Расскажи про врача", session_id))
    assert "уточните" in first.text.lower()

    confirm = asyncio.run(agent.chat("Суворов, а сколько стоит прием?", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    reply = asyncio.run(agent.chat("нет", session_id))
    assert reply.tool_name == "doctors_info"
    assert "суворов" in reply.text.lower()


def test_mixed_clarify_slot_answer_and_topic_switch_yes_reenters_new_question():
    memory = InMemoryMemory()
    services = DoctorClarifyMixedServices()
    agent = DoctorClarifyMixedAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "doctor_clarify_mixed_yes"

    asyncio.run(agent.chat("Расскажи про врача", session_id))
    asyncio.run(agent.chat("Суворов, а сколько стоит прием?", session_id))

    reply = asyncio.run(agent.chat("да", session_id))
    assert reply.tool_name == "price_info"
    assert "1200" in reply.text


def test_mixed_confirmation_yes_and_topic_switch_no_continues_confirmed_flow():
    memory = InMemoryMemory()
    services = FuzzyPreparePriceServices()
    agent = FuzzyPreparePriceAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "prepare_confirm_mixed_no"

    first = asyncio.run(agent.chat("Как подготовиться к общему анлизу крови?", session_id))
    assert "правильно понял" in first.text.lower()

    confirm = asyncio.run(agent.chat("Да, а сколько стоит общий анализ крови?", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    reply = asyncio.run(agent.chat("нет", session_id))
    assert reply.tool_name == "test_prepare"
    assert "натощак" in reply.text.lower()


def test_mixed_confirmation_yes_and_topic_switch_yes_reenters_new_question():
    memory = InMemoryMemory()
    services = FuzzyPreparePriceServices()
    agent = FuzzyPreparePriceAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "prepare_confirm_mixed_yes"

    asyncio.run(agent.chat("Как подготовиться к общему анлизу крови?", session_id))
    asyncio.run(agent.chat("Да, а сколько стоит общий анализ крови?", session_id))

    reply = asyncio.run(agent.chat("да", session_id))
    assert reply.tool_name == "price_info"
    assert "650" in reply.text


def test_mixed_confirmation_no_correction_and_topic_switch_no_reenters_correction():
    memory = InMemoryMemory()
    services = FuzzyDoctorMixedServices()
    agent = FuzzyDoctorMixedAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "doctor_confirm_mixed_no"

    first = asyncio.run(agent.chat("Расскажи про Дразннина", session_id))
    assert "правильно понял" in first.text.lower()

    confirm = asyncio.run(agent.chat("Нет, Суворов, а сколько стоит прием?", session_id))
    assert "прервать текущий сценарий" in confirm.text.lower()

    reply = asyncio.run(agent.chat("нет", session_id))
    assert reply.tool_name == "doctors_info"
    assert "суворов" in reply.text.lower()


class ScheduleFollowupServices:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def get_catalog_health(self) -> dict[str, object]:
        return {"ok": True}

    async def match_catalog_service(self, raw_text_or_name: str, *, current_service_name: str = "") -> dict[str, str]:
        _ = raw_text_or_name, current_service_name
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def match_catalog_doctor(self, raw_text_or_name: str) -> dict[str, str]:
        probe = str(raw_text_or_name or "").lower()
        if "дразнин" in probe:
            return {
                "status": "exact",
                "canonical": "Дразнин Антон Владимирович",
                "query": raw_text_or_name,
            }
        return {"status": "miss", "canonical": "", "query": raw_text_or_name}

    async def doctors_schedule_week(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query
        self.calls.append(dict(entities))
        return {
            "schedule": [
                {
                    "fio": "Дразнин Антон Владимирович",
                    "schedule": {
                        "г. Самара, пр. Ленина, 5": [
                            {
                                "date": "2026-04-14",
                                "slots": ["09:00", "09:30", "10:00"],
                                "start": "09:00",
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

    async def doctors_info(self, query: str, entities: dict[str, object]) -> dict[str, object]:
        _ = query, entities
        return {
            "doctors": [{"fio": "Дразнин Антон Владимирович", "specialization": "Эндоскопист"}],
            "note": "doctors_info",
        }

    def tool_handlers(self, *, include_meili_tools: bool) -> dict[str, object]:
        _ = include_meili_tools
        return {
            "doctors_schedule_week": self.doctors_schedule_week,
            "doctors_info": self.doctors_info,
        }


class ScheduleFollowupAgent(FreeTalkAgent):
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
        if "распис" in text or "дразнин" in text:
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
            confidence=0.2,
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


def test_contextual_schedule_followup_reuses_doctor_and_filters():
    memory = InMemoryMemory()
    services = ScheduleFollowupServices()
    agent = ScheduleFollowupAgent(
        config=_cfg(),
        services=services,  # type: ignore[arg-type]
        memory=memory,  # type: ignore[arg-type]
        persist=InMemoryPersist(),  # type: ignore[arg-type]
        system_prompt="FT test",
        web_search=None,
    )
    session_id = "schedule_followup_dialog"

    reply1 = asyncio.run(agent.chat("Подскажи расписание Дразнина", session_id))
    assert reply1.tool_name == "doctors_schedule_week"
    assert "дразнин" in reply1.text.lower()

    reply2 = asyncio.run(agent.chat("А на Ленина утром?", session_id))
    assert reply2.tool_name == "doctors_schedule_week"
    assert len(services.calls) >= 2
    assert services.calls[1]["doctor_name"] == "Дразнин Антон Владимирович"
    assert services.calls[1]["branch_name"] == "Ленина"
    assert services.calls[1]["time"] == "утром"
    assert services.calls[1]["time_from"] == "08:00"
