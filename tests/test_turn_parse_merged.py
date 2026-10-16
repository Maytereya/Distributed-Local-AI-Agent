"""Выбор фраз под `turn_parse = on`: бюджет ответа и слияние списка (L-15b, шаг 6; разбор 95 ответов).

Класс 1 — бюджет ответа LLM растёт с числом вариантов (`phrases_task`): 48 токенов обрезали ответ на
15–16 номерах (все 3 INVALID замера шага 6). Инвариант: у списка из N вариантов бюджет ≥ 16 + 4·N, у
коротких (≤8 вариантов) — прежние 48.

Класс 2 — LLM слила список в одну фразу (G4: «витамин D + паратгормон», «Комплекс анализов кортизол, ТТГ,
Т3, Т4» — 6 «хуже» из 8): выбрала всю реплику W, она слабая по данным, а разделители режут реплику на ≥2
фразы и одного названия прайса на неё нет. Это не решение «одна услуга», а сбой (`merged`): читатели
берут разрез по разделителям (политика сбоя шага 5). Инварианты: (а) слитый ответ не идёт путём одной
услуги — каждая позиция получает свои строки (оракул корзины), чужой цены нет; (б) W, сильная по данным,
или W без разделителей — по-прежнему решение LLM; (в) следующий короткий вопрос («а сколько?») не берёт
услугу, которой пациент не видел (контекст = показанное).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import pytest
from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_parse
from messengers_router.services import _turn_phrases as TP
from tests.test_price_one_or_many_class import CASES, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля
from tests.test_turn_phrases import _PhrasesLLM, _llm_parse

_LIST = "сколько стоит ферритин, ттг и витамин д"
_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)


@pytest.fixture(scope="module")
def catalog(mini_prod_world) -> list[dict[str, Any]]:  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        from messengers_router.services import _prices_helpers as H

        return H._catalog_rows(None)


@pytest.fixture
def world(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as frozen:
        yield frozen


# --------------------------------------------------------------------------- класс 1: бюджет ответа


@pytest.mark.parametrize("n", [1, 3, 8, 9, 12, 16])
def test_answer_budget_grows_with_the_number_of_options(n):
    task = TP.phrases_task(n)
    assert task.num_predict >= TP.PHRASES_TASK.num_predict
    assert task.num_predict >= 16 + 4 * n
    if n <= 8:
        assert task.num_predict == TP.PHRASES_TASK.num_predict  # короткие списки — как раньше
    assert TP.phrases_task(n).queue_timeout_ms == TP.PHRASES_TASK.queue_timeout_ms


def test_long_list_request_has_room_for_every_number(catalog, world, monkeypatch):
    words = ("ферритин", "ттг", "витамин д", "глюкоза", "фсг", "пролактин", "эстрадиол", "кортизол", "инсулин")
    reply = "сколько стоит " + ", ".join([*words, "магний", "спирометрия", "оам", "железо", "кальций"])
    llm = _PhrasesLLM(numbers=[])
    _llm_parse(reply, catalog, llm, monkeypatch)
    request = llm.calls[-1]
    n = len(request["format"]["properties"]["picks"]["items"]["enum"])
    assert n >= 12
    assert request["options"]["num_predict"] == max(48, 16 + 4 * n)


# --------------------------------------------------------------------------- класс 2: слияние списка


def test_whole_weak_reply_picked_over_delimiters_is_a_merge_failure(catalog, world, monkeypatch):
    parse = _llm_parse(_LIST, catalog, _PhrasesLLM(("ферритин, ттг и витамин д",)), monkeypatch)
    assert (parse.status, parse.source, parse.items, parse.decided) == ("merged", "llm", (), False)
    assert [item.text for item in parse.split] == ["ферритин", "ттг", "витамин д"]
    assert [item.strength for item in parse.split] == ["strong", "weak", "weak"]


def test_merge_failure_acts_by_the_delimiter_cut(catalog, world, monkeypatch):
    parse = _llm_parse(_LIST, catalog, _PhrasesLLM(("ферритин, ттг и витамин д",)), monkeypatch)

    async def turn():
        turn_parse.begin()
        memo = turn_parse.current()
        memo.phrases_result = parse
        return turn_parse.acting_for(_LIST), memo.phrases_result.decided

    acting, decided = asyncio.run(turn())
    assert decided is False  # не решение
    assert acting is not None and [item.text for item in acting.items] == ["ферритин", "ттг", "витамин д"]


@pytest.mark.parametrize(
    ("reply", "pick"),
    [
        # Без разделителей W — решение LLM «одна услуга» (g0542).
        ("сколько стоит мазок на микрофлору с поверхности задней стенки глотки", "мазок на микрофлору с поверхности задней стенки глотки"),
        # Разрез есть, но ни одна его часть не сильная по данным — список не подтверждён (g0270).
        ("сколько стоит спирометрия в покое и с нагрузкой", "спирометрия в покое и с нагрузкой"),
        # Одно название прайса покрывает реплику — разреза для сбоя нет («Билирубин и его фракции …»).
        ("сколько стоит Билирубин и его фракции (общий, прямой и непрямой)", "Билирубин и его фракции (общий, прямой и непрямой)"),
    ],
)
def test_whole_reply_stays_a_decision_without_a_delimiter_cut(reply, pick, catalog, world, monkeypatch):
    parse = _llm_parse(reply, catalog, _PhrasesLLM((pick,)), monkeypatch)
    assert parse.status in ("picked", "strong_whole", "single"), parse
    assert parse.decided


def test_picking_a_part_is_not_a_merge(catalog, world, monkeypatch):
    parse = _llm_parse(_LIST, catalog, _PhrasesLLM(("ттг",)), monkeypatch)
    assert (parse.status, [item.text for item in parse.items]) == ("picked", ["ттг"])


class _MergingOracle(_OracleLLM):
    """Оракул строк корзины, а на выборе фраз — вся реплика одной фразой (слияние G4)."""

    def __init__(self, case, whole: str) -> None:
        super().__init__(case, (whole,))


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


def test_merged_list_through_the_endpoint_prices_every_position(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy")
    oracle = _MergingOracle(case, "ферритин, ттг и витамин д")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([case.text, "а сколько?"], "merged-list-two-turns")
        assert list(world.net.attempts) == []
    assert "phrases" in oracle.calls
    violations = _violations(case, first["text"], "oracle", oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    # Два хода: короткий вопрос не берёт услугу, которой пациент не видел (контекст = показанное).
    shown = _priced(first["text"])
    unseen = sorted(_priced(second["text"]) - shown)
    assert unseen == [], "\n".join([*unseen, "--- первый ответ:", first["text"], "--- второй:", second["text"]])


def test_merge_is_counted_in_turn_telemetry(mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES["list_true_three"]
    _configure(monkeypatch, "deploy")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _MergingOracle(case, "ферритин, ттг и витамин д"))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        run_dialog([case.text], "merged-list-telemetry")
        row = world.take_telemetry()
    assert row.get("tu_parse_status") == "merged"
    assert row.get("tu_rule") == "parse_merged"

