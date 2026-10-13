"""Память разбора на ход (L-15b, шаг 1): одинаковое вычисление в ходе — один раз, ответ прежний.

Ценовая реплика за ход разбирается до пяти раз (понимание, корзина, срок ×2, подбор анализов), и
почти всегда с тем же входом; разбор списка на настоящем прайсе — секунды CPU. Замер 13.10 — журнал
STEP-2026-10-13-L15B-1-PARSE-CACHE. Память хода (`turn_parse`) снимает только повторы: ключ —
функция, точный текст и отпечаток среза прайса.

Инварианты:
- с памятью и без неё результат один и тот же; из памяти — копия, правка потребителя не протекает;
- память не переживает ход: новый ход считает заново, другой срез — другой ключ;
- повторный вызов в ходе не пересчитывает (резолвер не зовётся), вне хода — считается как раньше;
- работает из `asyncio.to_thread` и дочерних задач (копия контекста держит тот же объект);
- в ходе-списке через эндпоинт корзина берёт разбор понимания, а не считает его заново.

Эталон 45 диалогов off / shadow / on (`test_price_turns_golden.py`) и политика L-13
(`test_price_match_policy.py`) с памятью не меняются — это главное покрытие равенства поведения.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import turn_parse, turn_understanding
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

# Реплики-списки мини-прод: через запятую, «и», сплошной, с подстановкой синонима кода, с заголовком;
# и две не-списка — у них тоже есть ключ, и память не должна превращать их в список.
REPLIES = (
    "сколько стоит ферритин, глюкоза, холестерин",
    "оак, оам",
    "ферритин и глюкоза",
    "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",
    "сколько стоит ттг, т4 свободный",
    "сколько стоит ферритин, забор крови из вены, глюкоза",
    "Нужны цены на список анализов: ТТГ, ферритин, витамин D, глюкоза",
    "сколько стоит Общий анализ крови (Le, Er, Hb, СОЭ)",
    "когда будет готов анализ на ферритин",
)
LIST = REPLIES[0]


def _helpers():
    from messengers_router.services import _prices_helpers

    return _prices_helpers


def _parses(text: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    helpers = _helpers()
    return {
        "resolve": helpers._resolve_multi_price_items(text, rows),
        "basket": helpers._multi_price_items(text, rows),
        "understanding": helpers.price_list_services(text, rows=rows),
    }


@pytest.fixture
def resolver_calls(monkeypatch) -> list[str]:
    """Сколько раз разбор дошёл до резолвера — значит, считал, а не взял из памяти."""

    helpers = _helpers()
    real = helpers.resolve_price_service_name_from_catalog
    calls: list[str] = []

    def spy(query_text: str, **kwargs: Any) -> str | None:
        calls.append(query_text)
        return real(query_text, **kwargs)

    monkeypatch.setattr(helpers, "resolve_price_service_name_from_catalog", spy)
    return calls


@pytest.fixture
def turn_memos(monkeypatch) -> list[turn_parse._TurnParse]:
    """Каждый держатель, созданный `begin_turn()`, — чтобы посмотреть его счётчики после хода."""

    made: list[turn_parse._TurnParse] = []

    class Recording(turn_parse._TurnParse):
        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    monkeypatch.setattr(turn_parse, "_TurnParse", Recording)
    return made


def test_parse_with_turn_memory_equals_parse_without(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = _helpers()._catalog_rows(None)
        assert turn_parse.current() is None
        plain = {text: _parses(text, rows) for text in REPLIES}

        async def turn():
            turn_understanding.begin_turn()
            first = {text: _parses(text, rows) for text in REPLIES}
            again = {text: _parses(text, rows) for text in REPLIES}
            return first, again, turn_parse.current()

        first, again, memo = asyncio.run(turn())
    assert first == plain
    assert again == plain
    assert memo.reused >= 2 * len(REPLIES)
    assert sum(1 for names, _ in (p["understanding"] for p in plain.values()) if names) >= 6, "мало списков — тест пуст"


def test_parse_from_memory_is_a_copy(mini_prod_world):  # noqa: F811
    """Корзина и подбор анализов правят свои строки — из памяти каждый получает свою копию."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        helpers = _helpers()
        rows = helpers._catalog_rows(None)
        plain = helpers._multi_price_items(LIST, rows)

        async def turn():
            turn_understanding.begin_turn()
            items, unrecognized = helpers._multi_price_items(LIST, rows)
            items[0]["prices"][0]["cost"] = -1
            items.pop()
            unrecognized.append("чужое")
            reused = helpers._multi_price_items(LIST, rows)
            reused[0][0]["service_name"] = "чужое"
            return helpers._multi_price_items(LIST, rows)

        got = asyncio.run(turn())
    assert got == plain


def test_repeated_parse_in_a_turn_does_not_recompute(mini_prod_world, resolver_calls):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        helpers = _helpers()
        rows = helpers._catalog_rows(None)

        async def turn():
            turn_understanding.begin_turn()
            helpers._multi_price_items(LIST, rows)
            first = len(resolver_calls)
            helpers._multi_price_items(LIST, rows)  # корзина после понимания
            helpers._resolve_multi_price_items(LIST, rows)  # подбор анализов
            helpers.price_list_services(LIST, rows=rows)  # срок
            return first, len(resolver_calls), turn_parse.current()

        first, after, memo = asyncio.run(turn())
    assert first > 0
    assert after == first
    assert (memo.computed, memo.reused) == (2, 3)


def test_outside_a_turn_every_call_computes(mini_prod_world, resolver_calls):  # noqa: F811
    """Скрипты и тесты функций зовут разбор вне хода — памяти нет, поведение прежнее."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        helpers = _helpers()
        rows = helpers._catalog_rows(None)
        helpers._multi_price_items(LIST, rows)
        once = len(resolver_calls)
        helpers._multi_price_items(LIST, rows)
    assert turn_parse.current() is None
    assert len(resolver_calls) == 2 * once


def test_turn_memory_does_not_outlive_its_turn(mini_prod_world, resolver_calls):  # noqa: F811
    """Тот же текст в новом ходе считается заново; другой срез прайса — другой ключ и свой ответ."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        helpers = _helpers()
        rows = helpers._catalog_rows(None)
        without_glucose = [row for row in rows if row.get("serviceName") != "Глюкоза"]

        async def two_turns():
            turn_understanding.begin_turn()
            first = turn_parse.current()
            helpers._multi_price_items(LIST, rows)
            other_slice = [item["service_name"] for item in helpers._multi_price_items(LIST, without_glucose)[0]]
            turn_understanding.begin_turn()
            second = turn_parse.current()
            before = len(resolver_calls)
            same_text = helpers._multi_price_items(LIST, rows)
            return first, second, other_slice, len(resolver_calls) > before, same_text

        first, second, other_slice, recomputed, same_text = asyncio.run(two_turns())
        plain = helpers._multi_price_items(LIST, rows)
    assert second is not first
    assert recomputed
    assert same_text == plain
    assert "Глюкоза" in [item["service_name"] for item in plain[0]]
    assert "Глюкоза" not in other_slice


def test_turn_memory_is_shared_with_threads_and_child_tasks(mini_prod_world, resolver_calls):  # noqa: F811
    """`asyncio.to_thread` и дочерние задачи `gather` работают в копии контекста — с тем же держателем:
    посчитанное в потоке видно ходу, и наоборот."""

    other = "оак, оам"
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        helpers = _helpers()
        rows = helpers._catalog_rows(None)
        plain = helpers._multi_price_items(LIST, rows)
        plain_other = helpers._resolve_multi_price_items(other, rows)

        async def child() -> Any:
            return helpers._multi_price_items(LIST, rows)

        async def turn():
            turn_understanding.begin_turn()
            in_thread = await asyncio.to_thread(helpers._multi_price_items, LIST, rows)
            before = len(resolver_calls)
            here = helpers._multi_price_items(LIST, rows)
            from_child, *pair = await asyncio.gather(
                child(),
                asyncio.to_thread(helpers._resolve_multi_price_items, other, rows),
                asyncio.to_thread(helpers._resolve_multi_price_items, other, rows),
            )
            after_children = len(resolver_calls)
            again_other = helpers._resolve_multi_price_items(other, rows)
            return in_thread, here, from_child, pair, again_other, before, after_children, len(resolver_calls), turn_parse.current()

        in_thread, here, from_child, pair, again_other, before, after_children, end, memo = asyncio.run(turn())
    assert in_thread == here == from_child == plain
    assert pair == [plain_other, plain_other] and again_other == plain_other
    assert after_children > before  # «оак, оам» посчитан в потоке …
    assert end == after_children  # … и ходу виден без пересчёта
    assert memo.reused >= 3


@pytest.mark.parametrize("mode", ["shadow", "on"])
def test_list_turn_through_the_endpoint_parses_the_reply_once(mode, mini_prod_world, turn_memos, monkeypatch):  # noqa: F811
    """Через эндпоинт, два хода подряд в одном цикле: у каждого хода свой держатель и свой разбор
    (второй ход не берёт память первого); под `on` корзина берёт разбор понимания из памяти хода."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", mode, raising=False)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([LIST, LIST], f"turn-parse-{mode}")
    computing = [memo for memo in turn_memos if memo.computed]
    assert len(computing) == 2
    if mode == "on":
        assert all(memo.reused >= 1 for memo in computing)
        assert all("руб" in turn["text"] for turn in record)
