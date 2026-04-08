import asyncio
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from localragagent.freetalk.agent import FreeTalkAgent
from localragagent.freetalk.clinical_router import ClinicalDecision
from localragagent.freetalk.config import FreeTalkConfig
from localragagent.freetalk.tool_dispatcher import ToolDispatcher


def _cfg() -> FreeTalkConfig:
    return FreeTalkConfig(
        mode_key="Free-talk-Ai",
        redis_url="redis://redis:6379/0",
        redis_prefix="ft",
        session_ttl_sec=86400,
        history_tail_turns=14,
        compaction_trigger_turns=22,
        summary_keep_turns=8,
        max_tool_steps=3,
        llm_timeout_s=40,
        llm_queue_timeout_ms=30000,
        include_meili_tools=False,
        enable_web_search_tool=True,
        web_search_url="http://searxng:8080",
        web_search_timeout_s=12,
        web_search_max_results=5,
        web_search_language="ru-RU",
        web_search_healthcheck_timeout_s=3,
        web_search_healthcheck_ttl_s=30,
        context_window_tokens=24576,
        context_warn_ratio=0.82,
        context_estimate_chars_per_token=4,
        context_response_reserve_tokens=2048,
        persistent_memory_path=Path("app_data/free_talk_memory/dialog_summaries.jsonl"),
        system_prompt_path=Path("src/localragagent/freetalk/system_prompt.txt"),
    )


def test_render_tool_reply_uses_deterministic_payload_render(monkeypatch):
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )

    called = {"llm": False}

    async def fake_llm(self: FreeTalkAgent, _prompt: str) -> str:
        called["llm"] = True
        return "LLM text"

    monkeypatch.setattr(FreeTalkAgent, "_llm_text", fake_llm)

    payload = {
        "doctors": [
            {"fio": "Иванов Иван Иванович", "specialization": "Терапевт"},
        ],
        "note": "doctors_info",
    }
    text = asyncio.run(
        agent._render_tool_reply(
            user_message="Кто принимает?",
            tool_name="doctors_info",
            tool_payload=payload,
        )
    )

    assert "Иванов Иван Иванович" in text
    assert called["llm"] is False


def test_tool_dispatcher_ignores_service_bundle_metadata_without_data():
    async def service_bundle_empty(_query: str, _entities: dict[str, object]) -> dict[str, object]:
        return {
            "service_name": "УЗИ",
            "retail_prices": [],
            "doctors": [],
            "prepare": "",
            "top_n_applied": 5,
            "service_kind": "unknown",
            "note": "service_bundle_info",
            "entities_used": {"service_name_effective": "УЗИ"},
        }

    dispatcher = ToolDispatcher({"service_bundle_info": service_bundle_empty})
    result = asyncio.run(dispatcher.call("service_bundle_info", "цена узи", entities={}))

    assert result.found is False


def test_tool_dispatcher_treats_clarify_text_as_useful_data():
    async def price_with_clarify(_query: str, _entities: dict[str, object]) -> dict[str, object]:
        return {
            "prices": [],
            "clarify_text": "Введите конкретное название услуги.",
            "note": "price_info: generic_uzi_clarify",
        }

    dispatcher = ToolDispatcher({"price_info": price_with_clarify})
    result = asyncio.run(dispatcher.call("price_info", "цена узи", entities={}))

    assert result.found is True


def test_clinic_data_fallback_detector_handles_typo_and_clinic_context():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    assert agent._looks_like_clinic_data_query("Выведи пожалуйста теарапевтов клиники") is True


def test_extract_primary_doctor_name_from_tool_payload():
    payload = {
        "doctors": [
            {"fio": "Трубин Алексей Юрьевич", "specialization": "Уролог"},
        ],
        "note": "doctors_info",
    }
    name = FreeTalkAgent._extract_primary_doctor_name("doctors_info", payload)
    assert name == "Трубин Алексей Юрьевич"


def test_ground_entities_accepts_fuzzy_doctor_match():
    class _StubServices:
        async def match_catalog_service(self, _raw: str, *, current_service_name: str = "") -> dict[str, str]:
            _ = current_service_name
            return {"status": "miss", "canonical": ""}

        async def match_catalog_doctor(self, _raw: str) -> dict[str, str]:
            return {"status": "fuzzy", "canonical": "Дразнин"}

    agent = FreeTalkAgent(
        config=_cfg(),
        services=_StubServices(),  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )

    entities = asyncio.run(agent._ground_entities("врач дразнин"))
    assert entities.get("doctor_name") == "Дразнин"
    assert entities.get("doctor_name_match_status") == "fuzzy"


def test_catalog_resolution_reply_for_missing_specific_doctor():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    reply = agent._catalog_resolution_reply_if_needed(
        tool_plan=["doctors_info", "doctors_schedule_week"],
        entities={
            "_ft_doctor_match_status": "miss",
            "_ft_doctor_match_query": "Дразнин",
        },
        fallback_only=True,
    )
    assert reply is not None
    assert "не найден" in str(reply.text).lower()


def test_catalog_resolution_reply_not_triggered_for_specialty_miss():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    reply = agent._catalog_resolution_reply_if_needed(
        tool_plan=["doctors_info"],
        entities={
            "_ft_doctor_match_status": "miss",
            "_ft_doctor_match_query": "уролог",
        },
    )
    assert reply is None


def test_render_schedule_details_includes_region_dates_and_slots():
    payload = {
        "schedule": [
            {
                "fio": "Дразнин Антон Владимирович",
                "schedule": {
                    "г. Самара, пр. Ленина, 5": [
                        {
                            "date": "2026-04-10",
                            "slots": ["09:00", "09:30", "10:00"],
                            "start": "09:00",
                            "end": "18:00",
                        }
                    ]
                },
            }
        ],
        "note": "doctors_schedule_week: realtime from Nayka API",
    }

    text = FreeTalkAgent._render_schedule_details(payload)
    assert "Дразнин Антон Владимирович" in text
    assert "Самара" in text
    assert "09:00" in text
    assert "10.04" in text


def test_doctor_followup_message_detected_with_pronoun_and_memory():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    assert (
        agent._looks_like_doctor_followup_message(
            user_message="Лучше напиши, чем он занимается",
            remembered_doctor="Дразнин Антон Владимирович",
        )
        is True
    )


def test_apply_intent_entity_policy_drops_service_for_doctor_intent():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    entities = {
        "doctor_name": "Дразнин",
        "service_name": "Оформление медицинского заключения",
    }
    out = agent._apply_intent_entity_policy(
        user_message="информация о враче Дразнине",
        intent="doctor_info",
        entities=entities,
    )
    assert out.get("doctor_name") == "Дразнин"
    assert "service_name" not in out


def test_build_tool_plan_from_decision_uses_router_plan():
    agent = FreeTalkAgent(
        config=_cfg(),
        services=None,  # type: ignore[arg-type]
        memory=None,  # type: ignore[arg-type]
        persist=None,  # type: ignore[arg-type]
        system_prompt="test",
        web_search=None,
    )
    decision = ClinicalDecision(
        intent="doctor_info",
        confidence=0.8,
        entities={"doctor_name": "Дразнин"},
        tool_plan=["doctors_info", "doctors_schedule_week"],
    )
    plan = agent._build_tool_plan_from_decision(
        user_message="инфо о враче",
        decision=decision,
    )
    assert plan[:2] == ["doctors_info", "doctors_schedule_week"]
