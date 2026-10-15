"""Открытые баги 14.10 — тесты-фиксаторы: правильное поведение под `xfail(strict=True)`, симптом — зелёным «как сейчас».

Код бота не менялся. Когда баг починят, `xfail` станет XPASS и уронит прогон (strict) — отметку снять, тест «как
сейчас» этого бага заменить проверкой нового поведения (он закрепляет симптом точно, поэтому покраснеет вместе с
починкой).

1. BUG-2026-10-14-CANCEL-EXISTING-LOST (ИСПРАВЛЕНО 14.10: xfail снят, «как сейчас» заменён на «так больше не
   бывает»). «Хотите отменить или перенести запись?» → «отменить» → «Отменить текущий
   процесс записи?» → «да» → «Процесс записи отменён». Отмена СУЩЕСТВУЮЩЕЙ записи теряется без оператора. Решение
   владельца (25.09, 04.10/08.10): отмена существующей записи → оператор (перевод или оффер оператора), процесс записи
   этим не «отменяется». Корень: предпроверка записи (`appointment_flow_guard.run_appointment_precheck`) пропускает
   к обработчику выбора действия (`router._handle_appointment_action_pending`) только «да/нет»; «отменить» узнаёт
   `_APPOINTMENT_CANCEL_OR_RESTART_RE` (`отмен\\w*`) и ставит вопрос «прервать оформление» раньше обработчика.

2. BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE. «сколько стоит общий анализ крови» → список семейства с подсказкой
   «напишите: "все"» → «все» → список раскрывается верно, но поиск по каталогу (`router.py`, `catalog_service_exact`)
   берёт реплику «все» запросом услуги и кладёт в `service_name` «УЗИ периферических лимфоузлов комплексно - все зоны
   (детское)»; следующее «а как подготовиться?» отвечает про это УЗИ, которого пациент не видел.
   ИСПРАВЛЕНО под `[MESSENGER_ROUTER] context_shown = on` (L-14, «контекст = показанное»): на ходе раскрытия
   семейства роутер каталог по реплике не ищет (`shown_context.expands_shown`, флаг `show_all_context_kept`), а ответ,
   раскрывший список из контекста, оставляет в `service_name` услугу контекста до хода, если она среди показанных
   (`shown_context._keep_service_on_expand`). Под `off` — как было: симптом и `xfail(strict=True)` остаются для
   наборов `default`/`deploy`; те же проверки под наборами `*_shown` — зелёные.

Ходы — через обработчик эндпоинта (`messenger_generate_once`) в мире стенда на мини-проде (`FrozenWorld`: день, сеть
закрыта, LLM гейта недоступна), диалоги из двух-трёх ходов; наборы выключателей `default` и `deploy`, для бага 2 —
ещё `default_shown`, `deploy_shown`, `next_shown` (`context_shown = on` поверх набора, как в
`test_l14_two_turn_consumers.py`).
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
from messengers_router import shown_context
from tests.test_l14_two_turn_consumers import _SETS as _L14_SETS
from tests.test_price_one_or_many_class import _DEFAULT, _DEPLOY
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_SETS = {"default": _DEFAULT, "deploy": _DEPLOY}
# Баг 2 исправлен под `context_shown = on`: его проверки — ещё и под наборами с выключателем поверх набора (как
# `*_shown` в `test_l14_two_turn_consumers.py`). Баг 1 выключатель не трогает — его тесты только под `_SETS`.
_SHOWN_SETS = {
    "default_shown": {**_DEFAULT, "MR_CONTEXT_SHOWN": "on"},
    "deploy_shown": _L14_SETS["deploy_shown"],
    "next_shown": _L14_SETS["next_shown"],
}
_ALL_SETS = {**_SETS, **_SHOWN_SETS}


def _play(turns: list[str], config: str, monkeypatch, world_dir: Path, session: str) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта в одной сессии; после хода — ответ, решение и состояние сессии."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    for key, value in _ALL_SETS[config].items():
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
                        "shown": copy.deepcopy(entities.get(shown_context.SHOWN_KEY)),
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
def test_cancel_existing_is_not_lost_anymore(phrase, pick, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Так больше не бывает (исправлено 14.10): выбор «отменить» — сразу перевод тем же путём, что у отмены
    существующей записи (`existing_appointment_change`); вопроса «прервать оформление?» и «Процесс записи отменён» нет,
    оформление записи очищено."""

    from messengers_router.policies import handoff_message

    first, second, third = _cancel_dialog(phrase, pick, config, monkeypatch, mini_prod_world)
    assert first["text"] == _ACTION_QUESTION
    assert second["handoff"] is True and second["text"] == handoff_message("existing_appointment_change")
    assert {"existing_appointment_operator_handoff", "existing_appointment_cancel"} <= second["flags"]
    assert "appointment_flow_cancel_requested" not in second["flags"]
    assert (second["expects"] or {}).get("type") not in {"appointment_abort", "appointment_action"}
    assert "appointment_flow_cancelled" not in third["flags"] and _ABORTED not in third["text"]


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


def _show_all_dialog(
    reply: str, config: str, monkeypatch, world_dir: Path, last: tuple[str, ...] = (_PREPARE,)
) -> tuple[list[dict[str, Any]], set[str], str]:
    """Семейство с подсказкой «напишите: "все"» → «показать все» → `last` (по умолчанию — подготовка; пусто — два хода).

    :return: ходы, названия из показанных списков (обоих ходов) и запрос семейства из ответа бота
    """

    turns = [_FAMILY_QUESTION, reply, *last]
    record = _play(turns, config, monkeypatch, world_dir, f"show-all-{config}-{abs(hash(tuple(turns)))}")
    first, second = record[:2]
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


@pytest.mark.parametrize("config", sorted(_ALL_SETS))
@pytest.mark.parametrize("reply", _NOT_LEAKING)
def test_show_all_with_verb_keeps_context_on_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Граница класса: под `off` эти формы ведут себя правильно случайно (поиск по реплике вернул текущую услугу), под
    `context_shown = on` — потому что каталог по реплике не ищут; та же проверка, что у теста класса ниже, зелёная."""

    record, shown, query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world)
    _assert_context_is_what_was_shown(record, shown, query)


# Под `off` баг открыт: правильное поведение — xfail strict. Под `context_shown = on` — исправлено, тот же тест зелёный.
_OPEN_UNDER_OFF = pytest.mark.xfail(
    strict=True,
    reason="BUG-2026-10-14-SHOW-ALL-AS-CATALOG-SERVICE (под off): поиск по каталогу берёт реплику «все» запросом услуги "
    "(`catalog_service_exact`) — в `service_name` «УЗИ … все зоны (детское)», подготовка отвечает про это УЗИ",
)
_FIXED_UNDER_SHOWN = [*(pytest.param(config, marks=_OPEN_UNDER_OFF) for config in sorted(_SETS)), *sorted(_SHOWN_SETS)]


@pytest.mark.parametrize("config", _FIXED_UNDER_SHOWN)
@pytest.mark.parametrize("reply", _LEAKING)
def test_show_all_keeps_context_on_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс: «показать все» после списка семейства — не название услуги. Контекст после раскрытия — то, что пациент
    видел (или пусто); «а как подготовиться?» — не про услугу, которой не было ни в одном показанном списке."""

    record, shown, query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world)
    _assert_context_is_what_was_shown(record, shown, query)


# --------------------------------------------------------------------------- 2. под `context_shown = on`: как исправлено

_OFF_OF = {"default_shown": "default", "deploy_shown": "deploy"}  # набор под `on` → тот же набор под `off`


@pytest.mark.parametrize("config", sorted(_OFF_OF))
@pytest.mark.parametrize("reply", _LEAKING + _NOT_LEAKING)
def test_show_all_under_shown_answers_the_same_and_holds_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода через эндпоинт, все формы «показать все». Ответ раскрытия под `on` тот же, что под `off`; меняется
    только контекст: каталог по реплике не искали (флаг `show_all_context_kept`, без `catalog_service_exact`),
    `_shown_services` — весь раскрытый список, `service_name` — прежняя показанная услуга (та же, что после первого
    хода), а не находка поиска по реплике."""

    before = _show_all_dialog(reply, _OFF_OF[config], monkeypatch, mini_prod_world, last=())[0]
    (first, second), shown, _query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world, last=())
    assert [(t["text"], t["label"], t["handoff"]) for t in (first, second)] == [(t["text"], t["label"], t["handoff"]) for t in before]
    assert shown_context.SHOW_ALL_KEPT_FLAG in second["flags"] and "catalog_service_exact" not in second["flags"]
    assert second["shown"] == _NUMBERED.findall(second["text"])  # показанное — ровно строки ответа
    assert first["service_name"] in first["shown"] and second["service_name"] == first["service_name"]
    assert shown_context.SHOW_ALL_KEPT_FLAG not in before[1]["flags"]  # под `off` флага нет — ход как был


@pytest.mark.parametrize("config", sorted(_SHOWN_SETS))
@pytest.mark.parametrize("reply", _LEAKING)
def test_show_all_under_shown_then_price_shows_only_what_was_shown(reply, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Три хода: … → «показать все» → «сколько стоит?». Цена — строками, которые пациент видел, под названием
    показанной услуги; УЗИ «все зоны» (было: «По запросу «УЗИ периферических лимфоузлов…»») нет ни в ответе, ни в
    контексте."""

    record, shown, query = _show_all_dialog(reply, config, monkeypatch, mini_prod_world, last=("сколько стоит?",))
    _first, second, third = record
    assert third["label"] == "PRICE" and not third["handoff"]
    priced = _NUMBERED.findall(third["text"])
    assert priced and set(priced) <= shown, third["text"]
    assert "узи" not in third["text"].lower() and _seen(third["service_name"], shown, query)
    assert _seen(second["service_name"], shown, query)


@pytest.mark.parametrize(("mode", "before", "after"), [("on", "B", "B"), ("on", "Z", None), ("on", "", None), ("off", "B", _LEAKED)])
def test_expand_answer_restores_the_service_before_the_turn(mode, before, after, monkeypatch):
    """Флаг раскрытия у ответа (`_price_family_payload_from_context`), а не у решения классификатора: что бы ни
    подставили по реплике (здесь — УЗИ «все зоны»), после раскрытия в `service_name` — услуга контекста до хода, если
    она среди показанных, иначе услуги нет; `_shown_services` — весь список. Под `off` — не трогаем."""

    from messengers_router.mess_types import SessionState
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services._prices_helpers import _price_family_payload_from_context

    monkeypatch.setattr(cfg, "MR_CONTEXT_SHOWN", mode, raising=False)
    family = {"service_name": "запрос", "family_variants": [{"serviceName": n, "cost": 100} for n in "ABC"], "visible_limit": 2}
    state = SessionState(session_id="expand", last_entities={"service_name": before, "_price_family_context": family})
    shown_context.begin_turn(state)
    state.last_entities["service_name"] = _LEAKED  # подстановка по реплике «все»
    shown_context.sync_price(state, _price_family_payload_from_context(state.last_entities, show_all=True), {})
    assert state.last_entities.get("service_name") == after
    assert state.last_entities.get(shown_context.SHOWN_KEY) == (["A", "B", "C"] if mode == "on" else None)
