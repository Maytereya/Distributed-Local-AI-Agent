"""Вопрос walk-in после ответа с показанной услугой — L-14, находка 3 (`[MESSENGER_ROUTER] context_shown = on`).

Было: «сколько стоит ферритин» → «а где сдать?» — гард walk-in (`router._guard_walkin_named_service`, ветка «своей
услуги в реплике нет») ставил в `service_name` класс «анализы», и контекст терял показанную услугу. Под `off`
следующее «сколько стоит?» — «По услуге «сколько стоит»… не удалось… соединяю» (эталон `after_price_where` держит
«анализы»); под `on` после нескольких показанных строк — «По запросу «анализы» нашёл варианты…», а после одной
строки цену спасало только чтение показанного ценой (шаг 4), в состоянии и в сводке для NLU лежал класс.

Стало (под `on`): класс нужен только ответу — инструмент адресов выбирает по нему пункты сдачи анализов, ответ на
«где сдать?» прежний, — а после ответа в контексте снова показанное (`shown_context.sync_address`): одна услуга —
она, несколько — услуга контекста до хода, если она среди показанных; `_shown_services` ответ адреса не меняет.
Кнопка меню «Нужно ли записываться» — отдельный законный случай (класс от кнопки), её не трогаем. Под `off` — как было.

Обратный случай — своя услуга реплики адреса («а где сделать экг?», «а на ттг нужна запись?»): под `on` устаревшее
показанное перебивало её, и «сколько стоит?» следом отвечало ценой ферритина (под `off` — ценой ЭКГ). Теперь ход,
записавший в контекст услугу не из показанного, снимает `_shown_services` — дальше как под `off`.

Инвариант класса: вопрос о свойстве услуги без своей услуги («где сдать», «нужна ли запись») — о показанной; класс
услуги — не услуга и показанную в контексте не заменяет; своя услуга реплики сильнее показанного (принципы 2 и 5;
правило владельца 14.10: в контексте — услуга, которую ответ назвал пациенту одной строкой).
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.services._common import _normalise_catalog_text
from tests.test_l14_two_turn_consumers import BASE_CONFIGS, SHOWN_CONFIGS, _QUOTED, _SETS, _play, _priced
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

_HBA1C = "Гликированный гемоглобин  (HbA1C, Glycated Hemoglobin)"
_TTG = "ТТГ (TSH) тиреотропный гормон"
# Вопросы walk-in без своей услуги: правила понимают их как «живая очередь по контексту анализов».
_ASKS = ("а где сдать?", "нужна ли запись?", "а записываться нужно?")


def _no_class_or_question_words(text: str) -> bool:
    """Ответ не печатает названием услуги ни класс walk-in, ни слова вопроса."""

    return not [q for q in _QUOTED.findall(text) if _normalise_catalog_text(q) in {"анализы", "сколько стоит"}]


# --------------------------------------------------------------------------- одна показанная услуга

# (ответ X, показанная услуга, наборы). У слабых для правил фраз («гликированный гемоглобин», «ттг») под `next_shown`
# строку выбирает LLM (выбор цены и срока — модуль, политика L-13 — on), а LLM гейта недоступна: ответ X услуги не
# показывает, и вопросу walk-in не о чем быть — такие пары только в `deploy_shown`.
_ONE_SHOWN = (
    ("сколько стоит ферритин", "Ферритин", SHOWN_CONFIGS),  # цена: одна строка
    ("сколько готовится ферритин", "Ферритин", SHOWN_CONFIGS),  # срок из МИС называет одну услугу
    ("сколько стоит гликированный гемоглобин", _HBA1C, ("deploy_shown",)),  # другая одиночная услуга
    ("сколько готовится ттг", _TTG, ("deploy_shown",)),
)


def _one_shown_params():
    for first, service, configs in _ONE_SHOWN:
        for config in configs:
            for ask in _ASKS:
                yield pytest.param(first, service, ask, config, id=f"{first}-{ask}-{config}")


@pytest.mark.parametrize(("first", "service", "ask", "config"), list(_one_shown_params()))
def test_walkin_question_keeps_the_one_shown_service(first, service, ask, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода и третий через эндпоинт: после вопроса walk-in в контексте — показанная услуга, а не класс;
    «сколько стоит?» следом — её цена, без «анализы» и слов вопроса в заголовке."""

    shown, walkin, again = _play([first, ask, "сколько стоит?"], f"walkin-one-{first}-{ask}", mini_prod_world, monkeypatch, config)
    assert shown.shown == [service] and shown.service_name == service  # прошлый ответ назвал одну услугу
    assert walkin.label == "ADDRESS"
    assert (walkin.service_name, walkin.shown) == (service, [service]), walkin.text
    assert _priced(again.text) == [_normalise_catalog_text(service)], again.text
    assert _no_class_or_question_words(again.text), again.text


# --------------------------------------------------------------------------- несколько показанных строк


@pytest.mark.parametrize(
    ("first", "config"),
    [
        *((first, config) for first in ("сколько стоит оак", "сколько стоит ТТГ") for config in SHOWN_CONFIGS),
        ("сколько стоит витамин b12", "deploy_shown"),  # под `next_shown` ответ X — выбор LLM, она недоступна
    ],
)
def test_walkin_question_after_several_shown_rows_keeps_them(first, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Показано несколько строк: класс в `service_name` не подставляется (остаётся услуга контекста до хода — она
    среди показанных), показанное сохраняется; «сколько стоит?» — те же строки под прежним названием, не «анализы»."""

    price, walkin, again = _play([first, "а где сдать?", "сколько стоит?"], f"walkin-several-{first}", mini_prod_world, monkeypatch, config)
    assert len(price.shown) >= 2 and price.service_name in price.shown
    assert walkin.label == "ADDRESS"
    assert (walkin.service_name, walkin.shown) == (price.service_name, price.shown), walkin.text
    assert _priced(again.text) == _priced(price.text) and _priced(again.text), again.text
    assert _no_class_or_question_words(again.text), again.text


# --------------------------------------------------------------------------- ответ адреса и граница выключателя


@pytest.mark.parametrize("config", BASE_CONFIGS)
def test_off_keeps_the_class_as_before(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Под `off` — как было (эталон `after_price_where`): класс «анализы» в контексте, показанного нет."""

    _, walkin = _play(["сколько стоит ферритин", "а где сдать?"], "walkin-off", mini_prod_world, monkeypatch, config)
    assert (walkin.label, walkin.service_name, walkin.shown) == ("ADDRESS", "анализы", None)


@pytest.mark.parametrize(("off", "on"), [("deploy", "deploy_shown"), ("next", "next_shown")])
@pytest.mark.parametrize("first", ["сколько стоит ферритин", "сколько стоит оак"])
def test_address_answer_itself_is_unchanged(first, off, on, mini_prod_world, monkeypatch):  # noqa: F811
    """Ответ на «где сдать?» под `on` тот же, что под `off`: меняется только контекст после него."""

    before = _play([first, "а где сдать?"], f"walkin-same-{off}", mini_prod_world, monkeypatch, off)[1]
    after = _play([first, "а где сдать?"], f"walkin-same-{on}", mini_prod_world, monkeypatch, on)[1]
    assert (after.text, after.handoff, after.label, after.plan) == (before.text, before.handoff, before.label, before.plan)


def _gate_world_dialog(turns: list[str], config: str, monkeypatch) -> list[tuple[str, dict]]:
    """Диалог в мире гейта по умолчанию: `/regions` пуст, филиалы — из среза/seed, поэтому ответ адреса — настоящий
    («Анализы выполняются без записи…» и пункты сдачи), а не «Не нашёл филиалы» мини-прода."""

    from messengers_router import router as router_mod
    from messengers_router.memory import MemoryStore
    from messengers_router.mess_types import SessionState
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    state, memory = SessionState(session_id=f"walkin-shown-gate-{config}"), MemoryStore()

    async def run() -> list[tuple[str, dict]]:
        out = []
        for text in turns:
            events = [e async for e in router_mod.patient_routing_stream(text, state, services, memory)]
            out.append(("".join(e.text for e in events if e.text), dict(state.last_entities)))
        return out

    return asyncio.run(run())


def test_walk_in_answer_with_branches_is_unchanged_and_the_context_is_shown(monkeypatch):
    """С пунктами сдачи (мир гейта): текст ответа на «где сдать?» под `on` равен ответу под `off` и говорит
    «без записи»; контекст после него — показанная услуга (под `off` — класс, как было)."""

    turns = ["сколько стоит ферритин", "а где сдать?"]
    (_, off_price), (off_text, off_state) = _gate_world_dialog(turns, "deploy", monkeypatch)
    (_, on_price), (on_text, on_state) = _gate_world_dialog(turns, "deploy_shown", monkeypatch)
    assert "без записи" in on_text and on_text == off_text
    assert off_state.get("service_name") == "анализы"
    assert on_price.get("_shown_services") and on_state.get("service_name") == on_price.get("service_name")
    assert on_state.get("_shown_services") == on_price.get("_shown_services")


def test_menu_button_class_is_left_as_is(mini_prod_world, monkeypatch):  # noqa: F811
    """Кнопка «Нужно ли записываться» (раздел «Сдать анализы») — выбор пациента в меню: класс от кнопки остаётся,
    как было (гард walk-in кнопки не судит, флага «о показанном» нет)."""

    from evalbench import world as ev_world
    from messengers_router.runtime_config import config as cfg
    from tests.test_button_actions import GATEWAY_V1_TITLES
    from tests.test_price_turns_golden import run_dialog

    for key, value in _SETS["deploy_shown"].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    button = ("menu.tests.booking_needed", GATEWAY_V1_TITLES["menu.tests.booking_needed"])
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        price, pressed = run_dialog(["сколько стоит ферритин", button], "walkin-button-shown")
    assert price["service_name"] == "Ферритин"
    assert (pressed["label"], pressed["service_name"]) == ("ADDRESS", "анализы")


# --------------------------------------------------------------------------- своя услуга реплики адреса


_UZI = "Ультразвуковое исследование органов брюшной полости (комплексное)"


@pytest.mark.parametrize(("off", "on"), [("deploy", "deploy_shown"), ("next", "next_shown")])
@pytest.mark.parametrize(
    ("ask", "own"),
    [
        ("а где сделать экг?", "ЭКГ"),  # правило адреса: процедура названа в реплике
        ("где сделать узи брюшной полости?", _UZI),  # то же, услугу нашёл каталог
        ("а на ттг нужна запись?", _TTG),  # walk-in со своей услугой-анализом (каталог МИС)
    ],
)
def test_address_with_its_own_service_ends_the_shown_context(ask, own, off, on, mini_prod_world, monkeypatch):  # noqa: F811
    """Три хода: реплика адреса назвала свою услугу — показанное прошлым ответом («Ферритин») снимается, контекст —
    своя услуга, и «сколько стоит?» отвечает так же, как под `off` (было под `on`: ценой ферритина)."""

    turns = ["сколько стоит ферритин", ask, "сколько стоит?"]
    before = _play(turns, f"own-{off}", mini_prod_world, monkeypatch, off)
    after = _play(turns, f"own-{on}", mini_prod_world, monkeypatch, on)
    assert (after[1].service_name, after[1].shown) == (own, None), after[1].text
    assert after[1].text == before[1].text  # ответ адреса прежний
    assert after[2].text == before[2].text and "ферритин" not in _priced(after[2].text), after[2].text


@pytest.mark.parametrize("config", SHOWN_CONFIGS)
@pytest.mark.parametrize("ask", ["где вы находитесь?", "в каком филиале можно сдать?", "где сдать анализы?"])
def test_address_without_its_own_service_keeps_the_shown(ask, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Адрес без своей услуги (или со словом-классом «анализы») показанное не снимает: «сколько стоит?» — ферритин."""

    _, address, again = _play(["сколько стоит ферритин", ask, "сколько стоит?"], f"keep-{ask}", mini_prod_world, monkeypatch, config)
    assert (address.label, address.service_name, address.shown) == ("ADDRESS", "Ферритин", ["Ферритин"])
    assert _priced(again.text) == ["ферритин"], again.text


# --------------------------------------------------------------------------- граница функции


def test_sync_address_boundaries(monkeypatch):
    """`sync_address` (под `on`): по флагу гарда — только то, что записал walk-in; одна показанная — она, несколько —
    прежняя услуга, если она показана, иначе ничего. Без флага: своя услуга, записанная ходом и не из показанного,
    снимает показанное; класс walk-in («анализы», «ЭКГ» правила, кнопка) и неизменный контекст — нет."""

    from messengers_router import shown_context as sc
    from messengers_router.mess_types import RouteDecision, SessionState
    from messengers_router.runtime_config import config as cfg

    def run(mode: str, shown: list[str], before: str | None, *, held: str = "анализы", flag: bool = True, walkin: bool = True):
        monkeypatch.setattr(cfg, "MR_CONTEXT_SHOWN", mode, raising=False)
        state = SessionState(session_id="sync-address")
        state.last_entities.update({sc.SHOWN_KEY: list(shown), **({"service_name": before} if before else {})})
        sc.begin_turn(state)
        state.last_entities["service_name"] = held  # слияние сущностей хода
        flags = {"rule_nonbookable_walkin"} | (sc.walkin_about_shown() if flag else frozenset())
        sc.sync_address(state, RouteDecision(label="ADDRESS", entities={"service_name": "анализы"}, flags=flags), walkin=walkin)
        return state.last_entities.get("service_name"), state.last_entities.get(sc.SHOWN_KEY)

    ferritin = ["Ферритин"]
    # вопрос walk-in о показанном (флаг гарда)
    assert run("off", ferritin, "Ферритин") == ("анализы", ferritin)  # под off — как было
    assert run("on", [], "Ферритин") == ("анализы", [])  # прошлый ответ ничего не показал
    assert run("on", ferritin, "Ферритин", held="ЭКГ") == ("ЭКГ", ferritin)  # контекст сменил не сам walk-in
    assert run("on", ferritin, "Ферритин") == ("Ферритин", ferritin)
    assert run("on", ["A", "B"], "B") == ("B", ["A", "B"])
    assert run("on", ["A", "B"], "Липидограмма") == (None, ["A", "B"])  # догадка заземления, которой пациент не видел
    assert run("on", ["A", "B"], None) == (None, ["A", "B"])
    # без флага: класс walk-in и неизменный контекст — как было; своя услуга не из показанного — показанное снято
    assert run("on", ferritin, "Ферритин", flag=False) == ("анализы", ferritin)  # кнопка меню, класс из реплики
    assert run("on", ferritin, "Ферритин", held="ЭКГ", flag=False) == ("ЭКГ", ferritin)  # «ЭКГ» правила walk-in — класс
    assert run("on", ferritin, "ЭКГ", held="ЭКГ", flag=False, walkin=False) == ("ЭКГ", ferritin)  # ход контекст не менял
    assert run("on", ferritin, "Ферритин", held="ферритин", flag=False, walkin=False) == ("ферритин", ferritin)  # показанное
    assert run("on", ferritin, "Ферритин", held="ЭКГ", flag=False, walkin=False) == ("ЭКГ", None)  # адрес своей услуги
    assert run("on", ferritin, "Ферритин", held=_TTG, flag=False) == (_TTG, None)  # walk-in со своей услугой
    assert run("off", ferritin, "Ферритин", held="ЭКГ", flag=False, walkin=False) == ("ЭКГ", ferritin)
