"""Гард П2 у фразы разбора судит LLM (класс A разбора `e99c0eb`, решение владельца 15.10). Единственный путь с 17.10:
выключатель `parse_price_guard` (`rules` — гард по словарю каталога) снят (STEP-2026-10-17-ONE-PATH-PARSE-PRICE-GUARD).

g0757 «Сколько стоит анализ крови на концентрацию карбамазепина»: LLM разбора верно выбирает одну услугу, но гард П2
(`query_has_unsatisfiable_qualifier`) сверяет слова реплики со словарём прайса только дословно и считает
«карбамазепина» (родительный падеж) уточнением, которого в прайсе нет, — название сбрасывается, LLM цены не зовётся,
ответ «Уточните». Инвариант 3 класса `guard_over_blocks_silently`: гард не вправе объявлять пациентское слово
несуществующим по словарю каталога.

Фразу, выбранную LLM разбора одной слабой услугой, судит выбор строки (`select_price_rows`, правило 3 промпта отсекает «по ОМС»); гард П2 — страховка при сбое LLM.

Инварианты класса (через обработчик эндпоинта, мини-прод):
- выбор строки находит услугу в другой словоформе; LLM недоступна — ответ правил, без цены; прежнее значение
  `parse_price_guard = rules` в конфиге ничего не меняет;
- реплика с уточнением П2, которую разбор не отдаёт LLM («приём терапевта по ОМС» — одна фраза), — как было;
- два хода: короткий вопрос после ответа не получает цену того, чего пациент не видел.
"""

from __future__ import annotations

import pytest

from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _Oracle, _assert_follow_up_stays_on_what_was_shown, _dialog

_REPLY = "Сколько стоит анализ крови на концентрацию ферритина"
_PHRASE = "анализ крови на концентрацию ферритина"
_FERRITIN = "Ферритин — 530"


def test_the_guard_reads_the_word_form_as_an_absent_qualifier(mini_prod_world):  # noqa: F811
    """Предусловие класса: на мини-прайсе гард П2 считает «ферритина» уточнением (дословно в прайсе — «ферритин»)."""

    from agent_logic_2.nayka_api import api_price

    from evalbench import world as ev_world
    from messengers_router.services import _prices_helpers as H

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        assert H.query_has_unsatisfiable_qualifier(_REPLY, rows)
        assert not H.query_has_unsatisfiable_qualifier("Сколько стоит ферритин", rows)


@pytest.mark.parametrize("old_key", [None, "rules"])
def test_llm_finds_the_service_named_in_another_word_form(old_key, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    if old_key is not None:  # прод держит ключ до деплоя 6 — он не читается
        monkeypatch.setattr(cfg, "MR_PARSE_PRICE_GUARD", old_key, raising=False)
    oracle = _Oracle((_PHRASE,), ("Ферритин",))
    [turn], rows = _dialog([_REPLY], mini_prod_world, monkeypatch, f"guard-llm-{old_key}", oracle)
    assert _FERRITIN in turn["text"], turn["text"]
    assert rows[-1].get("price_guard") == "llm"


def test_llm_down_keeps_the_rules_answer_without_a_price(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = _Oracle((_PHRASE,))  # выбор строки — «недоступна»
    [turn], _rows = _dialog([_REPLY], mini_prod_world, monkeypatch, "guard-llm-down", oracle)
    assert "руб" not in turn["text"], turn["text"]


@pytest.mark.parametrize("reply", ["Сколько стоит приём терапевта по ОМС"])
def test_a_qualifier_the_parse_does_not_hand_to_the_llm_stays_unpriced(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """П2 (решение владельца): уточнение, которого клиника не оказывает, — честное «не нашёл», не платная цена.
    Выбор строки, который «нашёл бы» терапевта, здесь не спрашивается: разбор не выбирает фразу LLM."""

    oracle = _Oracle(("приём терапевта по ОМС",), ("Прием", "Консультация"))
    [turn], _rows = _dialog([reply], mini_prod_world, monkeypatch, f"guard-p2-{reply[-4:]}", oracle)
    assert "руб" not in turn["text"], turn["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_found_word_form_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    first, second = _dialog([_REPLY, follow_up], mini_prod_world, monkeypatch, f"guard-two-{follow_up}", _Oracle((_PHRASE,), ("Ферритин",)))[0]
    _assert_follow_up_stays_on_what_was_shown(first, second)


# --------------------------------------------------------------------------- открытые классы разбора (15.10)


@pytest.mark.xfail(strict=True, reason="класс B открыт: позиция без разделителя, которую LLM разбора не выбрала, теряется молча (g0255); B1 откачен — шум «Не распознал» и g0512")
def test_an_unpicked_position_without_a_separator_is_not_lost_silently(mini_prod_world, monkeypatch):  # noqa: F811
    [turn], _rows = _dialog(["Подскажите стоимость ХГЧ вместе с забором крови"], mini_prod_world, monkeypatch, "open-b", _Oracle(("ХГЧ",)))
    assert "забор" in turn["text"] or "Взятие крови" in turn["text"], turn["text"]


def _options(world_dir, monkeypatch, mode: str, reply: str) -> list[str]:
    from agent_logic_2.nayka_api import api_price

    from evalbench import world as ev_world
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _turn_phrases as TP

    monkeypatch.setattr(cfg, "MR_PARSE_FRAGMENT_WINDOWS", mode, raising=False)
    with ev_world.FrozenWorld(world_dir, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        return [option.raw.lower() for option in TP.phrase_candidates(reply, facet="price", catalog=rows).options]


_GLUED = "сколько стоят ферритин, ттг и витамин д"


def test_fragment_windows_off_keeps_the_glued_fragment(mini_prod_world, monkeypatch):  # noqa: F811
    """Как было (`parse_fragment_windows = off`): форма «стоят» вне обвязки — кандидат «стоят ферритин», «ферритин» нет."""

    options = _options(mini_prod_world, monkeypatch, "off", _GLUED)
    assert "стоят ферритин" in options and "ферритин" not in options, options


def test_fragment_windows_on_offers_the_strong_window(mini_prod_world, monkeypatch):  # noqa: F811
    """`on`: сильное по данным МИС окно внутри слабого фрагмента — вариант для LLM; слов в обвязку не дописано."""

    options = _options(mini_prod_world, monkeypatch, "on", _GLUED)
    assert "ферритин" in options and "стоят ферритин" in options, options
    assert "стоят" not in options  # промежуток между окнами — не вариант


@pytest.mark.parametrize("reply", ["ферритин, ттг, витамин д", "Стоимость анализов: ферритин, кальпротектин, лактозная непереносимость"])
def test_fragment_windows_change_nothing_when_fragments_are_strong_or_unknown(reply, mini_prod_world, monkeypatch):  # noqa: F811
    assert _options(mini_prod_world, monkeypatch, "on", reply) == _options(mini_prod_world, monkeypatch, "off", reply)


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_window_answer_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PARSE_FRAGMENT_WINDOWS", "on", raising=False)
    oracle = _Oracle(("ферритин", "ттг", "витамин д"), ("Ферритин", "ТТГ", "Витамин D"))
    first, second = _dialog([_GLUED, follow_up], mini_prod_world, monkeypatch, f"windows-two-{follow_up}", oracle)[0]
    assert "Ферритин — 530" in first["text"], first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
