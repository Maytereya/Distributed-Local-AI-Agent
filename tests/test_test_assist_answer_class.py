"""Подбор анализов — только по данным хода (BUG-2026-10-21-WALKIN-LLM-FREE-TEXT; принципы 1–3).

Класс: ход TEST_ASSIST, которому не нашлось готового ответа, отвечал свободным текстом LLM поверх строк прайса,
найденных по буквам реплики, и утверждал о клинике то, чего в данных нет: «подготовка обычно включает специальную
диету» к строке «Прото-скрин» (g0783), «анализ на магний доступен» при пустом поиске (g0190), анализы плода как
«комплекс к материнству» (g0266). Под `topic_by_llm` (ветки `rule_test_result`, `rule_prepare`) таких ходов больше.

Инвариант под `test_assist_from_data = on`: свободного текста на ходе TEST_ASSIST нет; в ответе — только названия и
сроки выбранных LLM строк прайса (печатает код, без цены); нет данных, сбой LLM, мусор — честный оффер оператора.
Под off — как было (рендер-подделка здесь «выдумывает», и на базе инвариант не держится).

Через обработчик эндпоинта; LLM — подделка настоящего типа (`GenerateResponse`).
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime, test_assist_answer
from messengers_router.service_answer import NO_DATA_TEXT
from tests.test_llm_primary_rules_fallback_class import _CLASSIFIER_MARK
from tests.test_service_answer_class import _LLM, _REGIONS, _classifier, _number_of, verdict

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_prod" / "snapshots"
_DOCTORS = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"
_TA_MARK = "об анализах и исследованиях ТОЛЬКО по прайсу клиники"
_RENDER_MARK = "Ты помощник клиники в мессенджере"
# Свойство вопроса об услуге из каталога (П1 на ходе `topic_by_llm`) решает LLM — как на проде: «есть ли такой анализ».
_FACET_MARK = "понять, что именно пациент хочет узнать об анализе или услуге"
_FACET_AVAILABILITY = json.dumps({"picks": [2]})
FABRICATED = "Подготовка обычно включает специальную диету и отказ от некоторых лекарств. Анализ доступен в нашей клинике."
OFFER = "Перевести?"
# Ветки правил, у которых тему решает LLM, — как на проде (деплой 16): с ними до свободного текста доходит больше ходов.
_PROD_TOPIC_BY_LLM = (
    "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,rule_website_help,"
    "rule_address_work_hours,rule_test_result,rule_prepare"
)

# Формулировки класса, которые на базе доходят до свободного текста: есть ли анализ (каталог нашёл, поиск не нашёл),
# подбор под цель, вопрос об анализах вообще, условие сдачи.
CLASS_REPLIES = [
    "Есть ли у вас анализ на антитела к кори?",
    "Нужен комплекс анализов для подготовки к беременности, и мужу тоже",
    "Анализ на магний у вас есть?",
    "Какие анализы сдать при постоянной усталости?",
    "Сколько обычно ждать результаты анализов?",
    "Есть ли у вас анализ на кальпротектин в жидком кале?",
]


@pytest.fixture(scope="module")
def ta_world(tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("ta_src")
    for sub in ("price_by_region", "price_units"):
        shutil.copytree(_FIXTURE / sub, source / sub)
    (source / "service_info").mkdir()
    (source / "service_info" / "service_info_20261008.jsonl").write_text("", encoding="utf-8")
    shutil.copy2(_DOCTORS, source / _DOCTORS.name)
    root = ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("ta_world") / "world")
    (root / "captured").mkdir()
    (root / "captured" / "kb_main_index.json").write_text("[]", encoding="utf-8")
    return root


def turns(world_dir, monkeypatch, replies: list[str], *, switch: str, answers: dict[str, Any], cfg_extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Ходы одной сессии; у каждого — текст, телеметрия, вызовы LLM, промпты и висящий оффер оператора."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services
    from messengers_router.services import _prepare_select_llm

    memory = MemoryStore()
    session = f"ta-{switch}-{replies[0]}"
    out: list[dict[str, Any]] = []

    async def run(reply: str) -> dict[str, Any]:
        req = MessengerGenerateRequest(session_id=session, text=reply, debug=True)
        body = json.loads((await messenger_generate_once(req, memory=memory, services=Services())).body)
        return {"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff"))}

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        patch.setattr(cfg, "MR_NLU_ENGINE", "legacy_v2", raising=False)
        patch.setattr(cfg, "MR_TOPIC_BY_LLM", _PROD_TOPIC_BY_LLM, raising=False)
        patch.setattr(cfg, "MR_TEST_ASSIST_FROM_DATA", switch, raising=False)
        for key, value in (cfg_extra or {}).items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setitem(_prepare_select_llm._kb_cache, "memos", ())

        async def regions(_self):
            return list(_REGIONS)

        patch.setattr(Services, "_ensure_regions_loaded", regions)
        for reply in replies:
            fake = _LLM(answers)
            patch.setattr(llm_runtime, "_OLLAMA_CLIENT", fake)
            choice._CACHE.clear()
            result = asyncio.run(run(reply))
            result["telemetry"] = world.take_telemetry() or {}
            result["calls"] = list(fake.calls)
            result["prompts"] = dict(fake.prompts)
            result["pending_offer"] = bool(memory.get(session).last_entities.get("_operator_offer_pending"))
            out.append(result)
    return out


def turn(world_dir, monkeypatch, reply: str, **kwargs: Any) -> dict[str, Any]:
    return turns(world_dir, monkeypatch, [reply], **kwargs)[0]


def _answers(ta_answer: Any) -> dict[str, Any]:
    return {_CLASSIFIER_MARK: _classifier("TEST_ASSIST"), _FACET_MARK: _FACET_AVAILABILITY, _TA_MARK: ta_answer, _RENDER_MARK: FABRICATED}


def _pick(*names: str, outcome: str = "none"):
    return lambda prompt: verdict(outcome, picks=tuple(_number_of(prompt, name) for name in names))


# --------------------------------------------------------------------------- база: класс существует


@pytest.mark.parametrize("reply", CLASS_REPLIES)
def test_off_is_free_text_as_before(reply, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, reply, switch="off", answers=_answers(verdict("no_data")))

    assert out["telemetry"].get("answer_source") == "llm_free", out["telemetry"]
    assert _TA_MARK not in out["calls"]
    assert FABRICATED in out["text"]  # на базе свободный текст печатается как есть — класс жив


# --------------------------------------------------------------------------- инвариант под on


@pytest.mark.parametrize("reply", CLASS_REPLIES)
def test_no_data_is_an_honest_offer_not_free_text(reply, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, reply, switch="on", answers=_answers(verdict("no_data")))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert _RENDER_MARK not in out["calls"]  # свободный текст не писался
    assert FABRICATED not in out["text"]
    assert out["text"].startswith(NO_DATA_TEXT) and OFFER in out["text"], out["text"]
    assert out["pending_offer"] and not out["handoff"]


def test_empty_pick_without_outcome_is_no_data(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[1], switch="on", answers=_answers(verdict()))

    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert out["text"].startswith(NO_DATA_TEXT) and out["pending_offer"]


def test_picked_rows_are_printed_from_the_price(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(_pick("Вирус кори Ig G", "Вирус кори Ig M")))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["text"].startswith(test_assist_answer.PICKED_HEADING), out["text"]
    # Название и срок — из строки прайса (срок по правилу печати сроков), не из текста LLM; цены нет.
    assert "— Вирус кори Ig G — срок готовности 2–3 дня" in out["text"], out["text"]
    assert "— Вирус кори Ig M — срок готовности до 10 дней" in out["text"], out["text"]
    assert "руб" not in out["text"] and FABRICATED not in out["text"]
    assert not out["pending_offer"]


def test_partly_prints_rows_and_offers_the_rest(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[5], switch="on", answers=_answers(_pick("Кальпротектин в кале", outcome="partly")))

    assert out["telemetry"].get("ta_answer") == "partly", out["telemetry"]
    assert "— Кальпротектин в кале — срок готовности 1–9 дней" in out["text"], out["text"]
    assert test_assist_answer.PARTLY_TAIL in out["text"] and OFFER in out["text"]
    assert out["pending_offer"] and FABRICATED not in out["text"]


def test_partly_without_rows_is_no_data(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[5], switch="on", answers=_answers(verdict("partly")))

    assert out["telemetry"].get("ta_answer") == "no_data", out["telemetry"]
    assert out["text"].startswith(NO_DATA_TEXT)


def test_question_about_analyses_in_general_asks_for_names(ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[4], switch="on", answers=_answers(verdict("need_name")))

    assert out["telemetry"].get("ta_answer") == "need_name", out["telemetry"]
    assert out["text"] == test_assist_answer.NEED_NAME_TEXT
    assert not out["pending_offer"]


@pytest.mark.parametrize("failure", [ConnectionError("тест: LLM недоступна"), "не json", verdict(picks=(99,))])
def test_llm_failure_or_garbage_is_an_offer_not_free_text(failure, ta_world, monkeypatch):
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(failure))

    assert out["telemetry"].get("answer_source") == "test_assist_data", out["telemetry"]
    assert out["telemetry"].get("ta_answer") == "failed", out["telemetry"]
    assert _RENDER_MARK not in out["calls"] and FABRICATED not in out["text"]
    assert out["text"].startswith(NO_DATA_TEXT) and out["pending_offer"]


def test_decision_budget_refusal_is_an_offer(ta_world, monkeypatch):
    # Решения хода до ответа — разбор фраз и свойство вопроса об услуге из каталога; на ответ подбора анализов лимита не остаётся.
    out = turn(
        ta_world, monkeypatch, CLASS_REPLIES[0], switch="on", answers=_answers(verdict("no_data")), cfg_extra={"MR_LLM_DECISION_TURN_CALLS": 2}
    )

    assert out["telemetry"].get("ta_answer") == "failed", out["telemetry"]
    assert out["telemetry"].get("ta_fail") == "budget", out["telemetry"]
    assert _RENDER_MARK not in out["calls"] and out["text"].startswith(NO_DATA_TEXT)


# --------------------------------------------------------------------------- кандидаты и контекст


def test_row_the_turn_tool_missed_is_a_candidate(ta_world, monkeypatch):
    # Поиск подбора анализов ищет по всей реплике и «магний» в вопросе не находит; пул выбора строки цены — находит.
    out = turn(ta_world, monkeypatch, CLASS_REPLIES[2], switch="on", answers=_answers(_pick("Mg+ (магний)")))

    assert out["telemetry"].get("ta_tool_rows") == 0, out["telemetry"]
    assert "— Mg+ (магний) — срок готовности 1–2 дня" in out["text"], out["text"]


def test_previous_bot_message_is_the_context_of_a_short_reply(ta_world, monkeypatch):
    answers = _answers(_pick("Вирус кори Ig G", "Вирус кори Ig M"))
    first, second = turns(ta_world, monkeypatch, [CLASS_REPLIES[0], "А есть ли такой анализ для ребёнка?"], switch="on", answers=answers)

    assert "Предыдущее сообщение бота: «(диалог только начался)»" in first["prompts"][_TA_MARK]
    assert "Предыдущее сообщение бота: «В прайсе клиники есть: — Вирус кори" in second["prompts"][_TA_MARK], second["prompts"]
