"""Контекст сессии = показанное — L-14, шаг 3 (`[MESSENGER_ROUTER] context_shown = on`, по умолчанию off).

Было: услугу в состояние писало только заземление правил по буквам. Строку, которую показал выбор LLM («ггтп» →
«ГГТ гамма-глютамилтрансфераза»), состояние не получало — следующее «сколько стоит?» не знало, о чём речь; «а
цена?» после ответа с услугой в контексте искала в прайсе букву «а» («По услуге «а»…» — цены аллергенов и шерсти);
сводка диалога называла услугу, которую состояние уже сменило или сняло.

Стало (под `on`): ответ с ценами пишет `_shown_services` — строки, которые увидел пациент, — и `service_name`, если
показана одна услуга; срок из МИС — только `_shown_services` (ответ без цен: по решению владельца 13.10 услуги в
контексте нет); корзина снимает `_shown_services`; реплика, в понимании которой услуга не названа, не перебивает
контекст фразой вырезателя; сводка обновляется, как только называет не ту услугу.

Инвариант класса: контекст — то, что пациент видел; короткий вопрос не показывает цен строк, которых пациент не видел,
и не печатает слова вопроса названием услуги. Под `off` — как было.
"""

from __future__ import annotations

import pytest

from messengers_router.services._common import _normalise_catalog_text
from tests.test_l14_two_turn_consumers import _FOUND_GGT, _QUOTED, _play, _priced, _shown
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

SHOWN = ("deploy_shown", "next_shown")
_GGT = "ГГТ гамма-глютамилтрансфераза"


@pytest.mark.parametrize(("raw", "expected"), [(None, "off"), ("off", "off"), ("on", "on"), ("ON", "on"), ("x", "off")])
def test_setting_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router import shown_context
    from messengers_router.runtime_config import config as cfg

    if raw is None:
        monkeypatch.delattr(cfg, "MR_CONTEXT_SHOWN", raising=False)
    else:
        monkeypatch.setattr(cfg, "MR_CONTEXT_SHOWN", raw, raising=False)
    assert shown_context.mode() == expected


def test_config_default_is_off():
    from agent_logic_2 import config

    assert config.MR_CONTEXT_SHOWN == "off"


def _row(name: str, price: int = 100) -> dict:
    return {"serviceName": name, "price": price}


@pytest.mark.parametrize(
    ("payload", "names"),
    [
        ({"clarify_text": "Уточните название", "prices": [_row("A")]}, []),  # уточнение цен не показывает
        ({"prices": [_row("A"), _row("A", 200), _row("B")]}, ["A", "B"]),  # повторы названия — одно
        ({"prices": [_row(str(i)) for i in range(8)]}, ["0", "1", "2", "3", "4"]),  # рендер печатает пять
        ({"service_kind": "family_query", "family_variants": [_row(str(i)) for i in range(12)], "visible_limit": 3}, ["0", "1", "2"]),
        ({"service_kind": "family_query", "family_variants": [_row(str(i)) for i in range(12)], "visible_limit": 3, "showing_all": True}, [str(i) for i in range(12)]),
        ({"prices": []}, []),
    ],
    ids=["clarify", "dedup", "five", "family_visible", "family_all", "empty"],
)
def test_shown_names_are_what_the_renderer_prints(payload, names):
    from messengers_router import shown_context

    assert shown_context.shown_price_names(payload) == names


def test_stale_summary_is_detected_only_when_it_names_another_service():
    from messengers_router.context_summary import _names_stale_context

    summary = "last_turns:\nuser: сколько стоит ферритин\n\nstate:\n_last_label=PRICE, service_name=Ферритин"
    assert not _names_stale_context(summary, {"service_name": "Ферритин"})
    assert _names_stale_context(summary, {})  # услугу сняли
    assert _names_stale_context(summary, {"service_name": "Гастрин"})  # услугу сменили
    assert not _names_stale_context("last_turns:\nuser: привет", {"service_name": "Ферритин"})  # сводка услуги не называет


@pytest.mark.parametrize("config", SHOWN)
def test_llm_picked_row_is_the_context_and_the_next_question_shows_it(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода через эндпоинт: строку выбрала LLM по термину — она в контексте; «сколько стоит?» — её цена, ничего
    другого."""

    first, second = _play(["сколько стоит ггтп", "сколько стоит?"], "shown-ggt", mini_prod_world, monkeypatch, config, llm=_FOUND_GGT())
    assert _priced(first.text) == [_normalise_catalog_text(_GGT)]
    assert (first.service_name, first.shown) == (_GGT, [_GGT])
    assert set(_priced(second.text)) <= _shown(first.text) and _priced(second.text), second.text


def test_off_keeps_the_old_context(mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _play(["сколько стоит ггтп", "сколько стоит?"], "shown-ggt-off", mini_prod_world, monkeypatch, "deploy", llm=_FOUND_GGT())
    assert first.service_name is None and first.shown is None
    assert not _priced(second.text)


@pytest.mark.parametrize("config", SHOWN)
@pytest.mark.parametrize("first", ["сколько стоит ферритин", "сколько стоит оак", "сколько стоит ггтп"])
def test_a_price_after_an_answer_does_not_search_the_letter(first, config, mini_prod_world, monkeypatch):  # noqa: F811
    """«а цена?» без названия услуги — о показанном: ни «По услуге «а»», ни цен строк, которых пациент не видел."""

    llm = _FOUND_GGT() if "ггтп" in first else None
    played = _play([first, "а цена?"], f"shown-a-{first}", mini_prod_world, monkeypatch, config, llm=llm)
    seen = _shown(played[0].text)
    assert set(_priced(played[1].text)) <= seen, played[1].text
    assert not [q for q in _QUOTED.findall(played[1].text) if _normalise_catalog_text(q) in {"а", "а цена"}], played[1].text


@pytest.mark.parametrize("config", SHOWN)
def test_readiness_shows_but_does_not_hold_the_service(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Срок из МИС: показанное записано, а услуги в контексте нет — ответ без цен (решение владельца 13.10)."""

    [turn] = _play(["сколько готовится ферритин"], "shown-readiness", mini_prod_world, monkeypatch, config)
    assert turn.text == "Срок готовности «Ферритин» — 1–2 дня."
    assert (turn.service_name, turn.shown) == (None, ["Ферритин"])


@pytest.mark.parametrize("config", SHOWN)
def test_cart_shows_all_its_rows_and_holds_no_single_service(config, mini_prod_world, monkeypatch):  # noqa: F811
    """Корзина: показанное — все её строки с ценой, одной услуги в контексте нет (позиций несколько)."""

    first, cart = _play(["сколько стоит ферритин", "сколько стоит ферритин, ттг и витамин д"], "shown-cart", mini_prod_world, monkeypatch, config)
    assert first.shown == ["Ферритин"] and first.service_name == "Ферритин"
    assert cart.service_name is None and len(cart.shown or []) >= 2
    assert {_normalise_catalog_text(name) for name in cart.shown} == set(_priced(cart.text)), (cart.shown, cart.text)
