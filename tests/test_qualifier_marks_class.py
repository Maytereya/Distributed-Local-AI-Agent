"""Уточнения позиции, этап 4 — отметки «есть в строке» и заголовок (DESIGN-2026-10-28-POSITION-QUALIFIERS; `qualifier_marks = on`) —
через обработчик эндпоинта, мини-прод, день, сеть закрыта.

Класс (решение владельца 27.10, п. 5; g0604 «ВПЧ количественным методом», g0388 «антитела к аннексину суммарные», g0811 «ПЦР на
аскаридоз»): заголовок ответа о цене («нашёл следующее / варианты стоимости») пишется без сведения, есть ли в выбранных строках
уточнение пациента, — пациент читает «та самая услуга», хотя метода или материала в строке нет. После выбора строки отдельный вопрос
LLM (`choose_qualifier_in_rows`) отмечает, что из названного пациентом есть в выбранных строках: сама услуга и каждое уточнение;
заголовок пишет код по ответу, слова кодом не сверяются. Отметка тем же вызовом, что выбор (первая версия этапа 4), защищала выбор —
карточка кнопок цены 28.10 (`c3kb*`): «суммарные» «есть» в IgG и IgM, «ПЦР» — в «Аскарида (p1)», спермограмма — на «пероксидазный тест».

Инварианты класса:
- уточнение, которого проверка не нашла в выбранных строках, — заголовок «Точно в таком виде в прайсе нет. Есть:», строки и цены те же;
- самой услуги в выбранных строках нет (выбор и проверка разошлись) — «не нашёл», цены нет (принцип 1);
- всё найдено — заголовок прежний; проверка не ответила — выбор как есть, заголовок прежний;
- встречные: off — проверки нет, запрос выбора строки байт в байт как без выключателя (он и под on тот же); реплика без уточнений
  (или только оплата) — проверки нет; решение хода проверка не тратит.
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
_CHECK = "Слова пациента (номер. слова):"
_REPLY = "сколько стоит ферритин количественным методом"
_PHRASE = "ферритин количественным методом"
_QUALIFIER = "количественным методом"


class _Rows:
    """Выбор строк — строки с началом названия `prefix`; проверка выбора — слова пациента из `present` (None — проверка недоступна);
    `prompts` — запросы выбора, `checks` — запросы проверки."""

    def __init__(self, present: set[str] | None, prefix: str = "Ферритин") -> None:
        self.present = present
        self.prefix = prefix
        self.prompts: list[str] = []
        self.checks: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _CHECK in prompt:
            self.checks.append(prompt)
            if self.present is None:
                raise ConnectionError("тест: проверка недоступна")
            words = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.split(_CHECK, 1)[1].splitlines()) if dot and head.isdigit()]
            return GenerateResponse(model="m", response=json.dumps({"picks": [n for n, text in words if text in self.present]}), done=True)
        seen = f"{system}\n\n{prompt}" if system else prompt
        if not any(mark in seen for mark in _ROWS_MARKS):
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.prompts.append(prompt)
        picks = [n for n, text in options if text.split(" → ", 1)[-1].startswith(self.prefix)]
        return GenerateResponse(model="m", response=json.dumps({"picks": picks, "reason": "none", "search_terms": []}, ensure_ascii=False), done=True)


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


_SERVICE = "ферритин"  # название услуги без уточнения — первый вариант проверки (`_qualifier_pool.named`)


def test_qualifier_not_in_the_row_gets_the_partly_heading(mini_prod_world, monkeypatch):  # noqa: F811
    rows = _Rows({_SERVICE})
    turn, row = _run(rows, mini_prod_world, monkeypatch, "qm-absent")
    assert turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    [check] = rows.checks
    assert "Ферритин" in check.split(_CHECK)[0] and f"1. {_SERVICE}" in check and f"2. {_QUALIFIER}" in check
    assert row["price_qual_missing"] == 1 and row["price_qual_check"] == "missing"


def test_qualifier_in_the_row_keeps_the_heading(mini_prod_world, monkeypatch):  # noqa: F811
    turn, row = _run(_Rows({_SERVICE, _QUALIFIER}), mini_prod_world, monkeypatch, "qm-found")
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    assert "price_qual_missing" not in row and row["price_qual_check"] == "found"


def test_service_not_in_the_chosen_rows_is_not_found(mini_prod_world, monkeypatch):  # noqa: F811
    """Выбор и проверка разошлись: в выбранных строках нет самой услуги (карточка `c3kb*`, g0817: спермограмма на «пероксидазный
    тест в эякуляте», строку привёл пул по материалу) — «не нашёл», цены нет (принцип 1: уточнение, не чужая цена)."""

    turn, row = _run(_Rows({_QUALIFIER}), mini_prod_world, monkeypatch, "qm-service-absent")
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert row["price_qual_check"] == "service_absent"


def test_check_down_keeps_the_choice(mini_prod_world, monkeypatch):  # noqa: F811
    turn, row = _run(_Rows(None), mini_prod_world, monkeypatch, "qm-check-down")
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    assert row["price_qual_check"] == "unavailable"


def test_the_row_choice_request_is_the_same_on_and_off(mini_prod_world, monkeypatch):  # noqa: F811
    """Запрос выбора строки под on — байт в байт как под off (повтор из кэша стенда); off — проверки нет."""

    on, off = _Rows({_SERVICE}), _Rows({_SERVICE})
    _run(on, mini_prod_world, monkeypatch, "qm-same-on")
    turn, row = _run(off, mini_prod_world, monkeypatch, "qm-same-off", marks_on=False)
    assert on.prompts == off.prompts and on.checks and not off.checks
    assert not turn["text"].startswith(_PARTLY) and _priced(turn["text"]) and "price_qual_check" not in row


def test_payment_is_not_checked_here(mini_prod_world, monkeypatch):  # noqa: F811
    """Оплату решает этап 2 (вердикт оплаты) — в проверку она не идёт; реплика без других уточнений — проверки нет."""

    rows = _Rows({_SERVICE})
    _run(rows, mini_prod_world, monkeypatch, "qm-pay", qualifiers=())
    assert rows.prompts and not rows.checks


def test_addon_without_its_row_is_priced_by_the_operator(mini_prod_world, monkeypatch):  # noqa: F811
    """Сопутствующая услуга («под наркозом») без своей строки: цена процедуры — с заголовком «в таком виде нет», стоимость наркоза —
    у оператора (решение 20.10 `4cec16e`, 27.10 п. 12: наркоз — вторая позиция; строки нет — «уточнит оператор»)."""

    rows = _Rows({"ФГДС"}, prefix="Эзофагогастродуоденоскопия")
    phrase = "ФГДС под наркозом"
    turn, _row = _run(
        rows, mini_prod_world, monkeypatch, "qm-addon", reply="сколько стоит ФГДС под наркозом", phrase=phrase,
        qualifiers=((phrase, "под наркозом", "addon"),),
    )
    assert turn["text"].startswith(_PARTLY) and re.search(r"Эзофагогастродуоденоскопия[^\n]*— 5 100 руб", turn["text"]), turn["text"]
    assert "«под наркозом» — стоимость уточнит оператор." in turn["text"], turn["text"]


def test_method_without_its_row_has_no_operator_line(mini_prod_world, monkeypatch):  # noqa: F811
    turn, _row = _run(_Rows({_SERVICE}), mini_prod_world, monkeypatch, "qm-method-noline")
    assert "стоимость уточнит оператор" not in turn["text"], turn["text"]


def test_check_spends_no_turn_decision(mini_prod_world, monkeypatch):  # noqa: F811
    """Проверка — вопрос о принятом решении (`Task.verdict`): решений хода не тратит и идёт, даже когда выбор израсходовал лимит
    (ход подбора анализов: классификатор, разметка, свойство вопроса, выбор — проверка молча не шла)."""

    from messengers_router.runtime_config import config as cfg

    _turn, before = _run(_Rows({_SERVICE}), mini_prod_world, monkeypatch, "qm-spent")
    on, off = before["tu_decisions_spent"], _run(_Rows({_SERVICE}), mini_prod_world, monkeypatch, "qm-spent-off", marks_on=False)[1]
    assert on == off["tu_decisions_spent"], (before, off)
    monkeypatch.setattr(cfg, "MR_LLM_DECISION_TURN_CALLS", on, raising=False)  # после выбора строки лимит исчерпан
    rows = _Rows({_SERVICE})
    turn, row = _run(rows, mini_prod_world, monkeypatch, "qm-spent-limit")
    assert rows.checks and row["price_qual_check"] == "missing" and turn["text"].startswith(_PARTLY), (row, turn["text"])


# --------------------------------------------------------------------------- контракт `choose()`: данные задачи (`context`)

_TASK = choice.Task(name="qualifier_in_rows")
_OPTIONS = [choice.Option(id="service", text="ферритин"), choice.Option(id="q0", text="количественным «методом»")]


def test_without_context_the_request_is_byte_for_byte_the_same():
    price = choice.Task(name="price_service_general", reasons=("other_service",), max_terms=3)
    assert choice.render(price, _REPLY, _OPTIONS) == choice.render(price, _REPLY, _OPTIONS, context="")


def test_context_goes_to_the_data_part_line_by_line():
    system, user = choice.render(_TASK, _REPLY, _OPTIONS, context="1. Ферритин — 530 руб.\n2. Глюкоза «натощак» — 190 руб.")
    plain_system, _user = choice.render(_TASK, _REPLY, _OPTIONS)
    assert system == plain_system and "1. Ферритин — 530 руб.\n2. Глюкоза \"натощак\" — 190 руб." in user and _CHECK in user


@pytest.mark.parametrize("answer_topic", ["on", "slot"])
def test_reply_to_the_price_button_is_checked_too(answer_topic, mini_prod_world, monkeypatch):  # noqa: F811
    """Ответ на вопрос кнопки цены анализа (карточка этапа 5 `c3kb*`, g0388, g0811): проверка выбора и здесь — заголовок «в таком
    виде нет», когда уточнения в строках нет."""

    from messengers_router.runtime_config import config as cfg
    from tests.test_button_actions import GATEWAY_V1_TITLES

    rows = _Rows({_SERVICE})
    _deploy(monkeypatch)
    keep_llm_breaker_closed(monkeypatch)
    for key, value in {
        "MR_POSITION_QUALIFIERS": "on", "MR_QUALIFIER_MARKS": "on", "MR_PENDING_BUTTONS": "on", "MR_BUTTON_ANSWER_TOPIC": answer_topic,
    }.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle((_PHRASE,), inner=rows, qualifiers=[(_PHRASE, _QUALIFIER, "method")]))
    with ev_world.FrozenWorld(mini_prod_world, "day") as frozen:
        _press, turn = run_dialog([("menu.price.test", GATEWAY_V1_TITLES["menu.price.test"]), _REPLY], f"qm-button-{answer_topic}")
        telemetry = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
    assert rows.checks, rows.prompts[:1]
    assert turn["text"].startswith(_PARTLY) and _priced(turn["text"]), turn["text"]
    assert telemetry[-1]["price_qual_missing"] == 1
