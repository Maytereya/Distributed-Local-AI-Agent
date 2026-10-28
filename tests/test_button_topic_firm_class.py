"""Тему ответа на вопрос кнопки переклейки после NLU по буквам не перебивают (STEP-2026-10-28-C6-BUTTON-TOPIC-FIRM).

Стенд деплоя 19 (`s19all_btn`): на листе «Подобрать анализы» тему ответа задаёт кнопка (`button_answer_topic = slot`, TEST_ASSIST),
но гард после NLU (`policy_nonbookable_walkin`) спрашивает у LLM другое — без записи ли сама услуга, — получает «да, анализы» и
переклеивает подбор в адреса: 4 хода из 4 — провал («подбор анализов подменён списком филиалов»). Тему от LLM-классификатора
(`topic_by_llm`) гард уже не перебивает (STEP-2026-10-21-B-TOPIC-BY-LLM); тема кнопки — тот же смысл, не буквы.

Та же карточка: когда гард отпустил тему, её следом перебивает П1 — совпадение реплики с каталогом по буквам
(`test_assist_catalog_hit_is_price`, `understanding_list_is_price`) переклеивает подбор в цену ближайшей строки («комплексный
анализ на ИППП» → «TORCH-инфекции — 3 800 руб.», чужая цена); на этом листе П1 и без гарда переклеивал 9 ходов (6 — провал).

Инвариант класса: под `button_topic_firm = on` ход, тему которого задала кнопка, остаётся в теме кнопки — ни гард «без записи», ни
П1 его не переклеивают (в журнале хода `button_topic_firm = walkin | p1`); подбор решает LLM по данным хода; та же реплика без
кнопки и off — как было. Через обработчик эндпоинта, LLM — подделка настоящего типа.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_button_actions import GATEWAY_V1_TITLES
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

_FREE_MARK = "Бот попросил назвать что-то своими словами"
_CLASSIFIER_MARK = "Ты классификатор запросов пациента"
_WALKIN_MARK = "хочет ли пациент сдать анализы или сделать ЭКГ"
_BUTTON = "menu.tests.suggest"
# Подбор анализов словами пациента: детектор «без записи» ловит «сдать … анализ».
_REPLIES = ["подберите анализы, где сдать кровь ребенку", "что сдать на анализ при подозрении на стафилококк"]


def _dialog(world_dir, monkeypatch, turns, *, switch: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"button-firm-{switch}-{json.dumps(turns, ensure_ascii=False)}"
    answers = {
        _FREE_MARK: json.dumps({"picks": [], "outcome": "answer"}),
        _WALKIN_MARK: json.dumps({"picks": [1], "outcome": "none"}),  # «анализы — без записи»
        _CLASSIFIER_MARK: json.dumps({"label": "TEST_ASSIST", "confidence": 0.9, "context_action": "new_topic", "entities": {}}),
        "Сообщение пациента": json.dumps({"picks": []}),
    }

    async def run(world) -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            out.append({"text": str(body.get("text") or ""), "telemetry": world.take_telemetry() or {}})
        return out

    with ev_world.FrozenWorld(world_dir, "day") as world, monkeypatch.context() as patch:
        for key, value in {
            "MR_NLU_ENGINE": "legacy_v2",
            "MR_PENDING_BUTTONS": "on",
            "MR_CHOOSE_CACHE_TTL_S": 0,
            "MR_BUTTON_ANSWER_TOPIC": "slot",
            "MR_WALKIN_BY_LLM": "on",
            "MR_BUTTON_TOPIC_FIRM": switch,
        }.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", _LLM(answers))
        choice._CACHE.clear()
        return asyncio.run(run(world))


def _walkin_relabel(tel: dict) -> bool:
    return any(str(s).startswith("policy_nonbookable_walkin:") for s in tel.get("tu_relabel_src") or [])


@pytest.mark.parametrize("reply", _REPLIES)
def test_off_relabels_button_topic_to_addresses_as_before(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _dialog(mini_prod_world, monkeypatch, [(_BUTTON, GATEWAY_V1_TITLES[_BUTTON]), reply], switch="off")[1]["telemetry"]
    assert tel.get("nlu_topic_by_llm") == "button_answer" and tel.get("tu_label_nlu") == "TEST_ASSIST", tel
    assert _walkin_relabel(tel) and tel.get("tu_label_final") == "ADDRESS", tel


@pytest.mark.parametrize("reply", _REPLIES)
def test_on_walkin_guard_keeps_the_button_topic(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _dialog(mini_prod_world, monkeypatch, [(_BUTTON, GATEWAY_V1_TITLES[_BUTTON]), reply], switch="on")[1]["telemetry"]
    assert tel.get("nlu_topic_by_llm") == "button_answer", tel
    assert not _walkin_relabel(tel) and tel.get("tu_label_final") != "ADDRESS", tel
    assert tel.get("button_topic_firm") == "walkin", tel


@pytest.mark.parametrize("reply", _REPLIES)
def test_typed_question_without_button_is_untouched(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [reply], switch="off")[0]
    on = _dialog(mini_prod_world, monkeypatch, [reply], switch="on")[0]
    assert on["text"] == off["text"]
    assert "button_topic_firm" not in on["telemetry"], on["telemetry"]


def _p1_relabel(tel: dict) -> bool:
    return any(str(s).split(":")[0] in {"test_assist_catalog_hit_is_price", "understanding_list_is_price"} for s in tel.get("tu_relabel_src") or [])


# Подбор, где реплика называет услугу каталога: П1 переклеивал её в цену по совпадению букв.
_NAMED = ["анализ на ферритин", "общий анализ крови и ферритин"]


@pytest.mark.parametrize("reply", _NAMED)
def test_off_p1_relabels_button_topic_to_price_as_before(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _dialog(mini_prod_world, monkeypatch, [(_BUTTON, GATEWAY_V1_TITLES[_BUTTON]), reply], switch="off")[1]["telemetry"]
    assert tel.get("nlu_topic_by_llm") == "button_answer", tel
    assert _p1_relabel(tel) and tel.get("tu_label_final") == "PRICE", tel


@pytest.mark.parametrize("reply", _NAMED)
def test_on_p1_keeps_the_button_topic(reply, mini_prod_world, monkeypatch):  # noqa: F811
    tel = _dialog(mini_prod_world, monkeypatch, [(_BUTTON, GATEWAY_V1_TITLES[_BUTTON]), reply], switch="on")[1]["telemetry"]
    assert not _p1_relabel(tel) and tel.get("tu_label_final") == "TEST_ASSIST", tel
    assert tel.get("button_topic_firm") == "p1", tel


@pytest.mark.parametrize("reply", _NAMED)
def test_typed_named_service_without_button_is_untouched(reply, mini_prod_world, monkeypatch):  # noqa: F811
    off = _dialog(mini_prod_world, monkeypatch, [reply], switch="off")[0]
    on = _dialog(mini_prod_world, monkeypatch, [reply], switch="on")[0]
    assert on["text"] == off["text"]
    assert "button_topic_firm" not in on["telemetry"], on["telemetry"]
