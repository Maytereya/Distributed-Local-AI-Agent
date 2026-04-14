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

    reply2 = asyncio.run(agent.chat("Иванов", session_id))
    assert "год рождения" in reply2.text.lower()

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
