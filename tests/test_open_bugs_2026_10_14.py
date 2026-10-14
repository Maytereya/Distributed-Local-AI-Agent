"""Открытые баги 14.10 — тесты-фиксаторы: правильное поведение под `xfail(strict=True)`, симптом — зелёным «как сейчас».

Код бота не менялся. Когда баг починят, `xfail` станет XPASS и уронит прогон (strict) — отметку снять, тест «как
сейчас» этого бага заменить проверкой нового поведения (он закрепляет симптом точно, поэтому покраснеет вместе с
починкой).

1. BUG-2026-10-14-CANCEL-EXISTING-LOST. «Хотите отменить или перенести запись?» → «отменить» → «Отменить текущий
   процесс записи?» → «да» → «Процесс записи отменён». Отмена СУЩЕСТВУЮЩЕЙ записи теряется без оператора. Решение
   владельца (25.09, 04.10/08.10): отмена существующей записи → оператор (перевод или оффер оператора), процесс записи
   этим не «отменяется». Корень: предпроверка записи (`appointment_flow_guard.run_appointment_precheck`) пропускает
   к обработчику выбора действия (`router._handle_appointment_action_pending`) только «да/нет»; «отменить» узнаёт
   `_APPOINTMENT_CANCEL_OR_RESTART_RE` (`отмен\\w*`) и ставит вопрос «прервать оформление» раньше обработчика.

2. BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE. «сколько стоит общий анализ крови» → список семейства с подсказкой
   «напишите: "все"» → «все» → список раскрывается верно, но поиск по каталогу (`router.py`, `catalog_service_exact`)
   берёт реплику «все» запросом услуги и кладёт в `service_name` «УЗИ периферических лимфоузлов комплексно - все зоны
   (детское)»; следующее «а как подготовиться?» отвечает про это УЗИ, которого пациент не видел.

Ходы — через обработчик эндпоинта (`messenger_generate_once`) в мире стенда на мини-проде (`FrozenWorld`: день, сеть
закрыта, LLM гейта недоступна), диалоги из трёх ходов; наборы выключателей `default` и `deploy`.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import expects
from tests.test_price_one_or_many_class import _DEPLOY
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_SETS = {"default": {}, "deploy": _DEPLOY}


def _play(turns: list[str], config: str, monkeypatch, world_dir: Path, session: str) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта в одной сессии; после хода — ответ, решение и состояние сессии."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)

    with ev_world.FrozenWorld(world_dir, "day") as world:

        async def run() -> list[dict[str, Any]]:
            memory, services = MemoryStore(), Services()
            out: list[dict[str, Any]] = []
            for text in turns:
                req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id="")
                body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
                decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
                entities = memory.get(session).last_entities
                out.append(
                    {
                        "text": body.get("text", ""),
                        "handoff": bool(body.get("handoff")),
                        "label": decision.get("label"),
                        "flags": set(decision.get("flags") or []),
                        "service_name": entities.get("service_name"),
                        "expects": copy.deepcopy(entities.get(expects.EXPECTS_KEY)),
                    }
                )
            return out

        record = asyncio.run(run())
        assert list(world.net.attempts) == []
    return record


# --------------------------------------------------------------------------- 1. отмена существующей записи

_ACTION_QUESTION = "Хотите отменить или перенести запись?"
_ABORT_QUESTION = "Отменить текущий процесс записи?"
_ABORTED = "Процесс записи отменён"
# Реплики, которые доводят бота до вопроса о действии с записью (отмена и перенос в одной фразе — действие не ясно).
_AMBIGUOUS_ACTION = ("отменить или перенести запись к терапевту", "запись к терапевту — перенести или отменить?")
# Ответ на вопрос — выбор «отменить» (слово из самого вопроса и его формы).
_CANCEL_PICKS = ("отменить", "Отменить запись", "хочу отменить")


def _cancel_dialog(phrase: str, pick: str, config: str, monkeypatch, world_dir: Path) -> list[dict[str, Any]]:
    return _play([phrase, pick, "да"], config, monkeypatch, world_dir, f"cancel-{config}-{abs(hash((phrase, pick)))}")


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("phrase", _AMBIGUOUS_ACTION)
def test_cancel_existing_reaches_the_action_question(phrase, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Предусловие обоих тестов ниже: фраза с отменой и переносом сразу — вопрос «отменить или перенести?»."""

    first, *_ = _cancel_dialog(phrase, "отменить", config, monkeypatch, mini_prod_world)
    assert first["text"] == _ACTION_QUESTION
    assert first["handoff"] is False and first["label"] == "APPOINTMENT"
    assert first["expects"]["type"] == "appointment_action" and "отменить" in first["expects"]["options"]


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("pick", _CANCEL_PICKS)
@pytest.mark.parametrize("phrase", _AMBIGUOUS_ACTION)
def test_cancel_existing_is_lost_as_now(phrase, pick, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Как сейчас (симптом): выбор «отменить» прерывает оформление — «да» и всё, оператора нет."""

    first, second, third = _cancel_dialog(phrase, pick, config, monkeypatch, mini_prod_world)
    assert first["text"] == _ACTION_QUESTION
    assert second["text"].startswith(_ABORT_QUESTION) and "appointment_flow_cancel_requested" in second["flags"]
    assert second["expects"]["type"] == "appointment_abort"  # ждёт «прервать оформление?», не выбор действия
    assert third["text"].startswith(_ABORTED) and "appointment_flow_cancelled" in third["flags"]
    assert not any(turn["handoff"] for turn in (first, second, third))
    assert "оператор" not in (second["text"] + third["text"]).lower()


@pytest.mark.xfail(
    strict=True,
    reason="BUG-2026-10-14-CANCEL-EXISTING-LOST: предпроверка записи узнаёт «отмен…» раньше обработчика выбора "
    "действия и спрашивает «Отменить текущий процесс записи?»; «да» — «Процесс записи отменён», оператора нет",
)
@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("pick", _CANCEL_PICKS)
@pytest.mark.parametrize("phrase", _AMBIGUOUS_ACTION)
def test_cancel_existing_goes_to_operator(phrase, pick, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс: на вопрос «Хотите отменить или перенести запись?» пациент выбрал отмену (любой формой) — отмена
    СУЩЕСТВУЮЩЕЙ записи → оператор (перевод или оффер оператора с текстом про отмену), не «прервать оформление»."""

    _first, second, third = _cancel_dialog(phrase, pick, config, monkeypatch, mini_prod_world)
    assert _ABORT_QUESTION not in second["text"]
    assert not (second["expects"] or {}).get("type") == "appointment_abort"
    to_operator = second["handoff"] or (second["expects"] or {}).get("type") == "operator_offer"
    assert to_operator and "оператор" in second["text"].lower() and "отмен" in second["text"].lower()
    assert _ABORTED not in third["text"]  # «да» на оффер — перевод, а не «процесс записи отменён»
    assert second["handoff"] or third["handoff"]


# --------------------------------------------------------------------------- 2. «все» как услуга каталога

_FAMILY_QUESTION = "сколько стоит общий анализ крови"
_PREPARE = "а как подготовиться?"
_QUERY = re.compile(r"По запросу «(?P<query>[^»]+)»")
_NUMBERED = re.compile(r"^\d+\. (?P<name>.+?) — \d[\d\s]* руб", re.M)
_HINT = re.compile(r"Чтобы показать их, напишите: \"(?P<word>[^\"]+)\"")
_LEAKED = "УЗИ периферических лимфоузлов комплексно - все зоны (детское)"


def _show_all_dialog(reply: str, config: str, monkeypatch, world_dir: Path) -> tuple[list[dict[str, Any]], set[str], str]:
    """Три хода: семейство с подсказкой «напишите: "все"» → «показать все» → подготовка.

    :return: ходы, названия из показанных списков (обоих ходов) и запрос семейства из ответа бота
    """

    turns = [_FAMILY_QUESTION, reply, _PREPARE]
    record = _play(turns, config, monkeypatch, world_dir, f"show-all-{config}-{abs(hash(tuple(turns)))}")
    first, second, _third = record
    hint = _HINT.search(first["text"])
    assert hint and hint.group("word") == "все", first["text"]
    shown = {m.group("name") for m in _NUMBERED.finditer(first["text"])} | {m.group("name") for m in _NUMBERED.finditer(second["text"])}
    assert second["label"] == "PRICE" and "rule_price_family_show_all" in second["flags"]
    assert len(_NUMBERED.findall(second["text"])) > len(_NUMBERED.findall(first["text"]))  # раскрыт весь список
    return record, shown, _QUERY.search(first["text"]).group("query")


def _seen(name: Any, shown: set[str], query: str) -> bool:
    """Пациент видел эту услугу: её нет, она из показанных списков или это его же запрос семейства."""

    return not name or name in shown or str(name).strip().lower() == query.strip().lower()


def _assert_context_is_what_was_shown(record: list[dict[str, Any]], shown: set[str], query: str) -> None:
    """Правильное поведение: после раскрытия и после «а как подготовиться?» в контексте — показанное (или пусто);
    в показанных списках УЗИ нет — значит, и ответ о подготовке не про УЗИ."""

    _first, second, third = record
    assert _seen(second["service_name"], shown, query), second["service_name"]
    assert _seen(third["service_name"], shown, query), third["service_name"]
    assert not any("узи" in name.lower() for name in shown)
    assert "узи" not in third["text"].lower()


# Узнаваемые формы «показать все» (одно определение — `_prices_helpers._is_price_show_all_request`). Все они уходят в
# поиск по каталогу запросом-репликой (`match_catalog_service("все", context_text="все")`): слово из подсказки и формы
# без глагола находят «УЗИ … все зоны (детское)»; «покажи/показать все» и «все варианты» поиск сводит к текущей услуге
# (строке показанного списка) — граница класса случайная, держится на ответе поиска, а не на понимании реплики.
_LEAKING = ("все", "Всё!", "все услуги?", "ВСЕ АНАЛИЗЫ")
_NOT_LEAKING = ("покажи все", "показать все.", "все варианты")


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("reply", _LEAKING)
def test_show_all_puts_unseen_service_in_context_as_now(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Как сейчас (симптом): список раскрыт верно, но в `service_name` — УЗИ «все зоны», и подготовка — про него."""

    (first, second, third), shown, _query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world)
    assert first["label"] == "PRICE" and first["service_name"] in shown
    assert "catalog_service_exact" in second["flags"] and second["service_name"] == _LEAKED
    assert _LEAKED not in shown
    assert third["label"] == "PREPARE" and third["service_name"] == _LEAKED
    assert "узи периферических лимфоузлов" in third["text"].lower()


@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("reply", _NOT_LEAKING)
def test_show_all_with_verb_keeps_context_on_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Граница класса как сейчас: эти формы уже ведут себя правильно — та же проверка, что у xfail ниже, зелёная."""

    record, shown, query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world)
    _assert_context_is_what_was_shown(record, shown, query)


@pytest.mark.xfail(
    strict=True,
    reason="BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE: поиск по каталогу берёт реплику «все» запросом услуги "
    "(`catalog_service_exact`) — в `service_name` «УЗИ … все зоны (детское)», подготовка отвечает про это УЗИ",
)
@pytest.mark.parametrize("config", sorted(_SETS))
@pytest.mark.parametrize("reply", _LEAKING)
def test_show_all_keeps_context_on_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс: «показать все» после списка семейства — не название услуги. Контекст после раскрытия — то, что пациент
    видел (или пусто); «а как подготовиться?» — не про услугу, которой не было ни в одном показанном списке."""

    record, shown, query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world)
    _assert_context_is_what_was_shown(record, shown, query)
