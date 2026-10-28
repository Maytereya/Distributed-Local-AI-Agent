"""Уточнения позиции, этап 2 — условие оплаты (DESIGN-2026-10-28-POSITION-QUALIFIERS; `position_qualifiers = on`) — через
обработчик эндпоинта, мини-прод, день, сеть закрыта.

Разметка хода `turn_spans` отвечает и полем `qualifiers`: у каждой услуги её уточнения (номера слов, вид) или «ни к одной»
(`of = 0`). Класс: «по ОМС / полису / ДМС» у позиции или на весь вопрос — прайс розничный, такой цены он не отвечает.

Инварианты класса (принцип 1: дезинформация ценой хуже уточнения; решение 22.07 — «по ОМС» — честное «не нашёл»):
- позиция с условием оплаты не получает розничную цену — при любой границе её группы (условие внутри группы или LLM его
  оторвала) и в любом месте списка; соседние позиции без условия — своими ценами (BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS);
- условие оплаты ни к одной услуге (на весь список; вопрос без названия после цены — BUG-2026-10-27-OMS-FOLLOWUP-PRICED) — на
  этом ходе цен нет вовсе; показанные раньше строки его не обходят;
- без суда LLM: условие оплаты решает разметка (0 новых решений, выбор строки не зовётся);
- встречные: уточнение другого вида и вопрос без уточнений — цены как были; выключатель off — разметка без поля, ответ прежний;
  разметка недоступна — как было («Уточните название»).

Подделка LLM — оракул разметки (`tests/spans_oracle.py`, поле `qualifiers`) поверх оракула выбора строк (`test_turn_parse_t3._Oracle`:
строки с заданным началом названия), прочие места — «недоступна»; предохранитель LLM не размыкается.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _turn_spans as S
from messengers_router.turn_parse import QualifierKind
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _ROWS_MARKS, _Oracle

PAY = QualifierKind.INSURANCE.value
_CLARIFY = "Уточните, пожалуйста, название услуги или анализа — подскажу стоимость."


class _Rows(_Oracle):
    """Выбор строк — строки с началом названия `rows`; счётчик вызовов выбора (`rows_calls`)."""

    def __init__(self, rows: tuple[str, ...]) -> None:
        super().__init__((), rows)
        self.rows_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if any(mark in seen for mark in _ROWS_MARKS):
            self.rows_calls += 1
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _run(turns, oracle: SpansOracle, world, monkeypatch, session: str, *, on: bool = True) -> tuple[list[dict], list[dict]]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    keep_llm_breaker_closed(monkeypatch)
    monkeypatch.setattr(cfg, "MR_POSITION_QUALIFIERS", "on" if on else "off", raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(world, "day") as frozen:
        record = run_dialog(list(turns), session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return record, rows


def _rub(text: str, name: str) -> bool:
    """В ответе есть строка `name` с ценой («Ферритин — 530 руб.»)."""

    return bool(re.search(rf"{re.escape(name)}[^\n—]*— [\d ]+ руб", text))


def _priced(text: str) -> bool:
    return bool(re.search(r"\d\s*руб", text))


def _sid(*parts: str) -> str:
    return "pq-" + "-".join(re.sub(r"\W+", "_", part)[-16:] for part in parts)


# --------------------------------------------------------------------------- позиция списка с условием оплаты

# (реплика, фразы разметки, уточнение (фраза, слова)) — условие оторвано от названия (как живая LLM в 59 из 82 случаев свипа П2) или
# внутри группы (правило 3 промпта), в начале и в конце списка, ОМС / полис / ДМС.
_LIST_CASES = [
    ("сколько стоит ферритин по омс, глюкоза", ("ферритин", "глюкоза"), ("ферритин", "по омс")),
    ("сколько стоит ферритин по омс, глюкоза", ("ферритин по омс", "глюкоза"), ("ферритин по омс", "по омс")),
    ("глюкоза и ферритин по полису сколько стоят", ("глюкоза", "ферритин"), ("ферритин", "по полису")),
    ("по ДМС ферритин и глюкоза сколько", ("ферритин", "глюкоза"), ("ферритин", "по ДМС")),
]


@pytest.mark.parametrize(("reply", "phrases", "qualifier"), _LIST_CASES)
def test_payment_takes_the_price_from_its_position_only(reply, phrases, qualifier, mini_prod_world, monkeypatch):  # noqa: F811
    of, words = qualifier
    oracle = SpansOracle(phrases, inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[(of, words, PAY)])
    [turn], [row] = _run([reply], oracle, mini_prod_world, monkeypatch, _sid("list", reply, of))
    assert not _rub(turn["text"], "Ферритин"), turn["text"]
    assert _rub(turn["text"], "Глюкоза"), turn["text"]
    assert "Не распознал" in turn["text"] and words.lower() in turn["text"].lower(), turn["text"]
    assert row["tu_qualifiers"] == PAY


def test_payment_on_the_whole_list_leaves_no_price(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[(None, "по омс", PAY)])
    [turn], [row] = _run(["ферритин и глюкоза по омс сколько стоят"], oracle, mini_prod_world, monkeypatch, _sid("whole"))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert row["price_payment"] == "loose" and row["tu_qualifiers_loose"] == PAY
    assert oracle.inner.rows_calls == 0  # без суда: условие оплаты решила разметка


# --------------------------------------------------------------------------- одна услуга и вопрос после цены


def test_one_service_with_payment_is_not_found_without_a_judge(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин",), inner=_Rows(("Ферритин",)), qualifiers=[("ферритин", "по омс", PAY)])
    [turn], [row] = _run(["сколько стоит ферритин по омс"], oracle, mini_prod_world, monkeypatch, _sid("one"))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert row["price_payment"] == "position" and oracle.inner.rows_calls == 0


@pytest.mark.parametrize(("follow_up", "condition"), [("а сколько стоит по омс?", "по омс"), ("а сколько стоит по полису?", "по полису")])
def test_payment_after_a_price_does_not_take_the_shown_row(follow_up, condition, mini_prod_world, monkeypatch):  # noqa: F811
    """BUG-2026-10-27-OMS-FOLLOWUP-PRICED: второй ход — условие без названия; показанная строка («контекст = показанное»)
    его не обходит."""

    oracle = SpansOracle(("ферритин",), inner=_Rows(("Ферритин",)), qualifiers=[(None, condition, PAY)])
    (first, second), rows = _run(["сколько стоит ферритин", follow_up], oracle, mini_prod_world, monkeypatch, _sid("follow", follow_up))
    assert _rub(first["text"], "Ферритин"), first["text"]
    assert not _priced(second["text"]) and "Не нашёл" in second["text"], second["text"]
    assert rows[-1]["price_payment"] == "loose"


# --------------------------------------------------------------------------- встречные


def test_list_without_qualifiers_keeps_both_prices(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[])
    [turn], [row] = _run(["сколько стоит ферритин и глюкоза"], oracle, mini_prod_world, monkeypatch, _sid("plain"))
    assert _rub(turn["text"], "Ферритин") and _rub(turn["text"], "Глюкоза"), turn["text"]
    assert "price_payment" not in row and "tu_qualifiers" not in row


def test_other_kinds_do_not_take_the_price(mini_prod_world, monkeypatch):  # noqa: F811
    """Уточнение не об оплате внутри группы («из вены» у ферритина) — не повод убирать цену: этап 2 решает только оплату."""

    oracle = SpansOracle(
        ("ферритин из вены", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[("ферритин из вены", "из вены", "material")]
    )
    [turn], [row] = _run(["сколько стоит ферритин из вены и глюкоза"], oracle, mini_prod_world, monkeypatch, _sid("material"))
    assert _rub(turn["text"], "Глюкоза") and "price_payment" not in row, turn["text"]
    assert row["tu_qualifiers"] == "material"


def test_off_asks_the_old_prompt_and_answers_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    """Выключатель off: промпт разметки без поля уточнений; ответ — как без выключателя (условие у сильной позиции списка — известный
    дефект пути off, xfail strict `test_p2_strong_phrase_basket_class` / `test_basket_weak_phrase_judge_class`)."""

    prompts: list[str] = []

    class _Seen(SpansOracle):
        async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
            if system:
                prompts.append(system)
            return await super().generate(prompt=prompt, system=system, **kwargs)

    oracle = _Seen(("ферритин", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")))
    [turn], [row] = _run(["сколько стоит ферритин и глюкоза"], oracle, mini_prod_world, monkeypatch, _sid("off"), on=False)
    assert prompts and not any('"qualifiers"' in text for text in prompts)
    assert _rub(turn["text"], "Ферритин") and _rub(turn["text"], "Глюкоза")
    assert "price_payment" not in row


def test_markup_down_is_as_before(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(None, inner=_Rows(("Ферритин", "Глюкоза")))
    [turn], [row] = _run(["сколько стоит ферритин по омс, глюкоза"], oracle, mini_prod_world, monkeypatch, _sid("down"))
    assert turn["text"] == _CLARIFY and row["tu_parse_status"] == "unavailable"


# --------------------------------------------------------------------------- контракт поля (модуль разметки)

_CATALOG = [
    {"serviceName": "Ферритин", "serviceHomecode": "408", "cost": 530},
    {"serviceName": "Глюкоза", "serviceHomecode": "152", "cost": 190},
]


def _parse(monkeypatch, reply: str, answer: dict):
    from messengers_router.runtime_config import config as cfg
    from ollama import GenerateResponse

    class _Fixed:
        async def generate(self, **_kwargs):
            return GenerateResponse(model="m", response=json.dumps(answer), done=True)

    monkeypatch.setattr(cfg, "MR_POSITION_QUALIFIERS", "on", raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Fixed())
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    return asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=_CATALOG))


def test_of_follows_the_group_not_the_sorted_order(monkeypatch):
    """`of` — номер группы в ответе LLM; группы код сортирует по месту в реплике — уточнение остаётся у своей услуги."""

    # «1:глюкоза 2:и 3:ферритин 4:по 5:омс»: LLM назвала ферритин первым
    parse = _parse(monkeypatch, "глюкоза и ферритин по омс", {"phrases": [[3, 3], [1, 1]], "qualifiers": [{"of": 1, "words": [4, 5], "kind": PAY}]})
    glucose, ferritin = parse.items
    assert glucose.qualifiers == () and [q.text for q in ferritin.qualifiers] == ["по омс"]
    assert ferritin.beside == "по омс" and ferritin.full == "ферритин по омс" and not ferritin.qualifiers[0].inside


def test_inside_loose_and_broken_qualifiers(monkeypatch):
    parse = _parse(
        monkeypatch,
        "ферритин по омс и глюкоза срочно",
        {
            "phrases": [[1, 3], [5, 5]],
            "qualifiers": [
                {"of": 1, "words": [2, 3], "kind": PAY},  # внутри своей группы
                {"of": 0, "words": [6, 6], "kind": "condition"},  # ни к одной
                {"of": 2, "words": [2, 3], "kind": PAY},  # слова чужой группы
                {"of": 3, "words": [6, 6], "kind": PAY},  # группы нет
                {"of": 1, "words": [9, 9], "kind": PAY},  # слова нет
                {"of": 1, "words": [6, 6], "kind": "price"},  # вида нет
            ],
        },
    )
    ferritin, glucose = parse.items
    assert parse.status == "picked" and [i.text for i in parse.items] == ["ферритин по омс", "глюкоза"]
    assert [(q.kind, q.text, q.inside) for q in ferritin.qualifiers] == [(QualifierKind.INSURANCE, "по омс", True)]
    assert ferritin.beside == "" and glucose.qualifiers == ()
    assert [(q.kind, q.text) for q in parse.loose] == [(QualifierKind.CONDITION, "срочно")]


def test_answer_without_the_field_is_a_parse_without_qualifiers(monkeypatch):
    parse = _parse(monkeypatch, "ферритин и глюкоза", {"phrases": [[1, 1], [3, 3]]})
    assert parse.status == "picked" and all(item.qualifiers == () for item in parse.items) and parse.loose == ()
