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
- в ходе-списке через эндпоинт у каждого хода свой держатель и свой разбор (корзина пути прода читает фразы разметки);
- слот разбора на фразы (L-15b, шаг 4): одна задача на ход, вне хода и под `off` слота нет; результат
  виден без ожидания только готовым; опоздавшую тень конец хода отменяет (`late`); тень память хода
  читает, но не пополняет.

Эталон 45 диалогов off / shadow / on (`test_price_turns_golden.py`) и политика L-13
(`test_price_match_policy.py`) с памятью не меняются — это главное покрытие равенства поведения.
"""

from __future__ import annotations

import asyncio
import time
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
    # Разбор списка посчитан по разу на функцию; внутри первого расчёта в памяти хода оседают и позиции
    # (`_resolve_fragment_service`, `_select_patient_price_rows` — шаг 6, O4), их повторов здесь нет.
    parses = [key for key in memo._results if str(key[0]).endswith("_multi_price_items")]
    assert (len(parses), memo.reused) == (2, 3)


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


@pytest.mark.parametrize("mode", ["shadow", "on"])  # понимание хода; разбор реплики — единственный режим
def test_list_turn_through_the_endpoint_parses_the_reply_once(mode, mini_prod_world, turn_memos, monkeypatch):  # noqa: F811
    """Через эндпоинт, два хода подряд в одном цикле: у каждого хода свой держатель и свой разбор
    (второй ход не берёт память первого); под `on` оба хода — корзина с ценами.

    Разметка фраз — оракул (`turn_spans_llm = on`, прод с деплоя 17): при лежащей LLM цена списка — «Уточните название»
    (`price_by_parse = on`), и до корзины ход не доходит. Корзина пути прода читает фразы разметки, а не память разбора
    правил (`_multi_price_items`) — переиспользование её памяти корзиной было свойством пути off."""

    from messengers_router import llm_runtime
    from messengers_router.runtime_config import config as cfg
    from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", mode, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(("ферритин", "глюкоза", "холестерин")))
    keep_llm_breaker_closed(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([LIST, LIST], f"turn-parse-{mode}")
    computing = [memo for memo in turn_memos if memo.computed]
    assert len(computing) == 2
    if mode == "on":
        assert all("руб" in turn["text"] for turn in record)


# --------------------------------------------------------------------------- слот разбора на фразы (L-15b, шаг 4)
#
# Разбор реплики на фразы — одна задача на ход в держателе хода (`start_phrases`); ход ждёт её сразу (единственный
# режим с T7 шага 3б), недождавшуюся конец хода отменяет (`cancel_unfinished`). Вне хода слота нет — читатели
# получают None.


def _parsed(status: str = "picked") -> turn_parse.PhraseParse:
    return turn_parse.PhraseParse(reply="ферритин, глюкоза", status=status, source="llm")


def test_phrase_slot_once_per_turn():
    calls: list[int] = []

    async def factory() -> turn_parse.PhraseParse:
        calls.append(1)
        return _parsed()

    async def two_turns():
        turn_understanding.begin_turn()
        first, again = turn_parse.start_phrases(factory), turn_parse.start_phrases(factory)
        result = await turn_parse.phrases()
        turn_understanding.begin_turn()
        next_turn = turn_parse.start_phrases(factory)
        await turn_parse.phrases()
        return first, again, next_turn, result

    first, again, next_turn, result = asyncio.run(two_turns())
    assert first is again and next_turn is not first
    assert len(calls) == 2 and result.status == "picked"


def test_no_slot_outside_a_turn():
    async def factory() -> turn_parse.PhraseParse:
        raise AssertionError("разбор не должен запускаться")

    async def outside():
        turn_parse._CURRENT.set(None)
        turn_parse.cancel_unfinished()  # слота нет — ничего не делает
        return (turn_parse.start_phrases(factory), turn_parse.phrases_now(), await turn_parse.phrases(), turn_parse.acting_for("ферритин, глюкоза"), turn_parse.debug())

    assert asyncio.run(outside()) == (None,) * 5


def test_phrases_now_is_none_until_done():
    async def turn(status: str):
        turn_understanding.begin_turn()
        gate = asyncio.Event()

        async def factory() -> turn_parse.PhraseParse:
            await gate.wait()
            return _parsed(status)

        turn_parse.start_phrases(factory)
        await asyncio.sleep(0)
        before = (turn_parse.phrases_now(), turn_parse.debug())
        gate.set()
        result = await turn_parse.phrases()
        return before, result, turn_parse.phrases_now(), turn_parse.acting_for(result.reply), turn_parse.debug()

    before, result, now, acting, debug = asyncio.run(turn("picked"))
    assert before == (None, {"status": "pending"})
    assert result is now and result.status == "picked" and debug["status"] == "picked"
    assert acting.status == "picked"
    assert asyncio.run(turn("unavailable"))[3] is None  # сбой без разреза читателю не отдаётся


def test_end_of_turn_cancels_an_unawaited_parse():
    """Разбор, который ход не дождался (сбой между запуском и ожиданием), конец хода отменяет — LLM после ответа не
    тратится; результата у отменённого нет (статуса `late` больше нет)."""

    cancelled: list[bool] = []

    async def slow() -> turn_parse.PhraseParse:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return _parsed()

    async def turn():
        turn_understanding.begin_turn()
        task = turn_parse.start_phrases(slow, reply="ферритин, глюкоза")
        await asyncio.sleep(0)
        started = time.monotonic()
        turn_parse.cancel_unfinished()
        await asyncio.wait({task})
        return task, time.monotonic() - started, turn_parse.phrases_now()

    task, waited, result = asyncio.run(turn())
    assert task.cancelled() and cancelled == [True] and waited < 1.0 and result is None


def test_slot_is_shared_with_threads_and_child_tasks():
    calls: list[int] = []

    async def factory() -> turn_parse.PhraseParse:
        calls.append(1)
        return _parsed()

    async def child():
        return turn_parse.start_phrases(factory)

    async def turn():
        turn_understanding.begin_turn()
        first, second = await asyncio.gather(child(), child())
        result = await turn_parse.phrases()
        in_thread = await asyncio.to_thread(turn_parse.phrases_now)
        return first, second, result, in_thread

    first, second, result, in_thread = asyncio.run(turn())
    assert first is second and len(calls) == 1 and in_thread is result
