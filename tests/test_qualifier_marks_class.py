"""Уточнения позиции, этап 4 — отметки «есть в строке» и заголовок (DESIGN-2026-10-28-POSITION-QUALIFIERS; `qualifier_marks = on`) —
через обработчик эндпоинта, мини-прод, день, сеть закрыта.

Класс (решение владельца 27.10, п. 5; g0604 «ВПЧ количественным методом», g0388 «антитела к аннексину суммарные», g0811 «ПЦР на
аскаридоз»): заголовок ответа о цене («нашёл следующее / варианты стоимости») пишется без сведения, есть ли в выбранных строках
уточнение пациента, — пациент читает «та самая услуга», хотя метода или материала в строке нет. Выбор строки тем же вызовом LLM
отмечает по каждому уточнению «есть в строках» (`in_rows`); заголовок пишет код по отметкам, слова кодом не сверяются.

Инварианты класса:
- уточнение, которого LLM не нашла в выбранных строках, — заголовок «Точно в таком виде в прайсе нет. Есть:», строки и цены те же;
- все уточнения найдены — заголовок прежний;
- встречные: off — запрос к LLM без блока уточнений, ответ прежний; ответ без отметок (не по контракту) — заголовок прежний;
  реплика без уточнений — запрос к LLM байт в байт прежний (повтор из кэша стенда под off не ломается).
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import choice, llm_runtime
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _ROWS_MARKS

_PARTLY = "Точно в таком виде в прайсе нет. Есть:"
_BLOCK = "Уточнения пациента к услуге:"
_REPLY = "сколько стоит ферритин количественным методом"
_PHRASE = "ферритин количественным методом"
_QUALIFIER = "количественным методом"


class _Rows:
    """Выбор строк — строки с началом названия «Ферритин»; `in_rows` — заданные отметки (None — без поля); `prompts` — запросы выбора."""

    def __init__(self, in_rows: list[bool] | None, prefix: str = "Ферритин") -> None:
        self.in_rows = in_rows
        self.prefix = prefix
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if not any(mark in seen for mark in _ROWS_MARKS):
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.prompts.append(prompt)
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        picks = [n for n, text in options if text.split(" → ", 1)[-1].startswith(self.prefix)]
        answer: dict = {"picks": picks, "reason": "none", "search_terms": []}
        if self.in_rows is not None and _BLOCK in prompt:
            answer["in_rows"] = self.in_rows
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _run(
    rows: _Rows,
    world,
    monkeypatch,
    session: str,
    *,
    marks_on: bool = True,
    qualifiers=((_PHRASE, _QUALIFIER, "method"),),
    reply: str = _REPLY,
    phrase: str = _PHRASE,
) -> tuple[dict, dict]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    keep_llm_breaker_closed(monkeypatch)
    monkeypatch.setattr(cfg, "MR_POSITION_QUALIFIERS", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_QUALIFIER_MARKS", "on" if marks_on else "off", raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle((phrase,), inner=rows, qualifiers=list(qualifiers)))
    with ev_world.FrozenWorld(world, "day") as frozen:
        [turn] = run_dialog([reply], session)
        telemetry = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return turn, telemetry[-1]


def _priced(text: str) -> bool:
    return bool(re.search(r"Ферритин[^\n]*— 530 руб", text))


def test_qualifier_not_in_the_row_gets_the_partly_heading(mini_prod_world, monkeypatch):  # noqa: F811
    rows = _Rows([False])
    turn, row = _run(rows, mini_prod_world, monkeypatch, "qm-absent")
    assert turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    assert any(_BLOCK in prompt and f'"{_QUALIFIER}"' in prompt for prompt in rows.prompts)
    assert row["price_qual_missing"] == 1


def test_qualifier_in_the_row_keeps_the_heading(mini_prod_world, monkeypatch):  # noqa: F811
    turn, row = _run(_Rows([True]), mini_prod_world, monkeypatch, "qm-found")
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    assert "price_qual_missing" not in row


def test_answer_without_marks_keeps_the_heading(mini_prod_world, monkeypatch):  # noqa: F811
    turn, _row = _run(_Rows(None), mini_prod_world, monkeypatch, "qm-nomarks")
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]


def test_off_asks_without_the_block(mini_prod_world, monkeypatch):  # noqa: F811
    rows = _Rows([False])
    turn, _row = _run(rows, mini_prod_world, monkeypatch, "qm-off", marks_on=False)
    assert rows.prompts and not any(_BLOCK in prompt for prompt in rows.prompts)
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]


def test_payment_is_not_marked_here(mini_prod_world, monkeypatch):  # noqa: F811
    """Оплату решает этап 2 (без розничной цены) — в блок отметок она не идёт; реплика без других уточнений — блока нет."""

    rows = _Rows([False])
    _run(rows, mini_prod_world, monkeypatch, "qm-pay", qualifiers=())
    assert rows.prompts and not any(_BLOCK in prompt for prompt in rows.prompts)


def test_addon_without_its_row_is_priced_by_the_operator(mini_prod_world, monkeypatch):  # noqa: F811
    """Сопутствующая услуга («под наркозом») без своей строки: цена процедуры — с заголовком «в таком виде нет», стоимость наркоза —
    у оператора (решение 20.10 `4cec16e`, 27.10 п. 12: наркоз — вторая позиция; строки нет — «уточнит оператор»)."""

    rows = _Rows([False], prefix="Эзофагогастродуоденоскопия")
    phrase = "ФГДС под наркозом"
    turn, _row = _run(
        rows, mini_prod_world, monkeypatch, "qm-addon", reply="сколько стоит ФГДС под наркозом", phrase=phrase,
        qualifiers=((phrase, "под наркозом", "addon"),),
    )
    assert turn["text"].startswith(_PARTLY) and re.search(r"Эзофагогастродуоденоскопия[^\n]*— 5 100 руб", turn["text"]), turn["text"]
    assert "«под наркозом» — стоимость уточнит оператор." in turn["text"], turn["text"]


def test_method_without_its_row_has_no_operator_line(mini_prod_world, monkeypatch):  # noqa: F811
    turn, _row = _run(_Rows([False]), mini_prod_world, monkeypatch, "qm-method-noline")
    assert "стоимость уточнит оператор" not in turn["text"], turn["text"]


# --------------------------------------------------------------------------- контракт `choose()`

_TASK = choice.Task(name="price_service_general", reasons=("other_service",), max_terms=3)
_OPTIONS = [choice.Option(id="a", text="Ферритин — 530 руб."), choice.Option(id="b", text="Глюкоза — 190 руб.")]


def test_without_qualifiers_the_request_is_byte_for_byte_the_same():
    assert choice.render(_TASK, _REPLY, _OPTIONS) == choice.render(_TASK, _REPLY, _OPTIONS, qualifiers=())
    assert choice.answer_schema(_TASK, _OPTIONS) == choice.answer_schema(_TASK, _OPTIONS, ())
    assert "in_rows" not in choice.answer_schema(_TASK, _OPTIONS)["properties"]


def test_qualifiers_go_to_the_data_part_not_the_instruction():
    system, user = choice.render(_TASK, _REPLY, _OPTIONS, qualifiers=("количественным «методом»",))
    plain_system, _plain_user = choice.render(_TASK, _REPLY, _OPTIONS)
    assert system == plain_system and _BLOCK in user and '1. "количественным "методом""' in user
    schema = choice.answer_schema(_TASK, _OPTIONS, ("а", "б"))["properties"]["in_rows"]
    assert schema["minItems"] == schema["maxItems"] == 2


@pytest.mark.parametrize(("marks", "count", "expected"), [([True, False], 2, (True, False)), ([True], 2, ()), (["да"], 1, ()), (None, 1, ()), ([True], 0, ())])
def test_marks_off_contract_are_no_marks(marks, count, expected):
    assert choice._in_rows({"in_rows": marks} if marks is not None else {}, count) == expected
