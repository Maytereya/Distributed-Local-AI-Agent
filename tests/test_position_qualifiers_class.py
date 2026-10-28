"""Уточнения позиции, этап 2 — условие у позиции (DESIGN-2026-10-28-POSITION-QUALIFIERS; `position_qualifiers = on`) — через
обработчик эндпоинта, мини-прод, день, сеть закрыта.

Разметка хода `turn_spans` отвечает и полем `qualifiers`: у каждой услуги её уточнения (номера слов, вид) или «ни к одной»
(`of = 0`). Уточнение, которое LLM отнесла к позиции, — в `ParsedPhrase.beside` (раньше его заполнял только генератор окон по
соседству слов), дальше — прежний механизм «слово — повод спросить, вердикт — LLM»: гард П2 по словарю прайса и суд условия в том же
вызове корзины (правило 5 её промпта). Прямого «оплата → без цены» нет: карточка 28.10 (`c3ton_*`) — живая LLM разметки звала
«оплатой» и «цену», и «платно», и «сертификат» (−20 верных цен).

Инварианты класса (принцип 1; решение 22.07 — «по ОМС» — честное «не нашёл»):
- позиция с условием, которого нет в прайсе, не получает розничную цену — при любой границе её группы (условие внутри группы или
  LLM его оторвала) и в любом месте списка; соседние позиции — своими ценами (BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS);
- условие ни к одной услуге — у каждой позиции; вопрос без названия после цены («а сколько стоит по омс?») не берёт показанную строку —
  судья П2 по услуге контекста (BUG-2026-10-27-OMS-FOLLOWUP-PRICED);
- слова вопроса о цене, которые LLM отметила уточнением («сколько стоит», «какая цена»), цену не снимают (гард их не видит);
- встречные: вопрос без уточнений — цены как были; выключатель off — разметка без поля, ответ прежний; разметка недоступна — как было.

Подделка LLM — оракул разметки (`tests/spans_oracle.py`, поле `qualifiers`) поверх суда: корзина отвергает позицию с «[добавлено:
«… омс / полис / дмс»]» (правило 5, как `test_p2_strong_phrase_basket_class._RuleFive`), выбор одной строки на реплике с таким условием —
«условия нет» (как `test_qualifier_round_two_class._Judge`); прочие строки — с заданным началом названия; остальное — «недоступна».
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _turn_spans as S
from messengers_router.turn_parse import QualifierKind
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _ROWS_MARKS

PAY = QualifierKind.INSURANCE.value
_CLARIFY = "Уточните, пожалуйста, название услуги или анализа — подскажу стоимость."
_CONDITION = re.compile(r"омс|полис|дмс", re.I)
_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"


class _Rows:
    """Суд и выбор строк: корзина — варианты с началом названия `rows`, кроме позиций с «[добавлено: «… условие»]»; одна строка — на
    реплике с условием пусто и «условия нет», иначе строки с началом `rows`. `rows_calls` — вызовы выбора."""

    def __init__(self, rows: tuple[str, ...]) -> None:
        self.rows = rows
        self.rows_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if not any(mark in seen for mark in _ROWS_MARKS):
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.rows_calls += 1
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _BASKET_MARK in seen:
            picks = [
                n for n, text in options
                if text.split(" → ", 1)[-1].startswith(self.rows)
                and not re.search(r"\[добавлено: «[^»]*(омс|полис|дмс)", text.split(" → ", 1)[0], re.I)
            ]
            answer: dict = {"picks": picks}
        else:
            question = re.search(r"пациента: «(.*?)»", prompt, re.S)
            if question and _CONDITION.search(question.group(1)):
                answer = {"picks": [], "reason": "qualifier_absent", "search_terms": []}
            else:
                answer = {"picks": [n for n, text in options if text.split(" → ", 1)[-1].startswith(self.rows)], "reason": "none", "search_terms": []}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


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


# --------------------------------------------------------------------------- позиция списка с условием

# (реплика, фразы разметки, уточнение (фраза, слова)) — условие оторвано от названия (как живая LLM в 59 из 82 случаев свипа П2) или
# внутри группы (правило 3 промпта), в начале и в конце списка, ОМС / полис / ДМС.
_LIST_CASES = [
    ("сколько стоит ферритин по омс, глюкоза", ("ферритин", "глюкоза"), ("ферритин", "по омс")),
    ("сколько стоит ферритин по омс, глюкоза", ("ферритин по омс", "глюкоза"), ("ферритин по омс", "по омс")),
    ("глюкоза и ферритин по полису сколько стоят", ("глюкоза", "ферритин"), ("ферритин", "по полису")),
    ("по ДМС ферритин и глюкоза сколько", ("ферритин", "глюкоза"), ("ферритин", "по ДМС")),
    # Слабая позиция (не дословное название: «ТТГ» ≠ «ТТГ (TSH) тиреотропный гормон»), которую резолвер разрешает, — условие рядом с ней
    # не теряется (свип 28.10, lc_c03: путь слабой позиции `beside` не читал); и внутри группы.
    ("сколько стоит ТТГ по омс, глюкоза", ("ТТГ", "глюкоза"), ("ТТГ", "по омс")),
    ("сколько стоит ТТГ по омс, глюкоза", ("ТТГ по омс", "глюкоза"), ("ТТГ по омс", "по омс")),
]


@pytest.mark.parametrize(("reply", "phrases", "qualifier"), _LIST_CASES)
def test_condition_takes_the_price_from_its_position_only(reply, phrases, qualifier, mini_prod_world, monkeypatch):  # noqa: F811
    of, words = qualifier
    oracle = SpansOracle(phrases, inner=_Rows(("Ферритин", "Глюкоза", "ТТГ")), qualifiers=[(of, words, PAY)])
    [turn], [row] = _run([reply], oracle, mini_prod_world, monkeypatch, _sid("list", reply, of))
    named = "ТТГ" if "ТТГ" in reply else "Ферритин"
    assert not _rub(turn["text"], named), turn["text"]
    assert _rub(turn["text"], "Глюкоза"), turn["text"]
    assert "Не распознал" in turn["text"] and words.lower() in turn["text"].lower(), turn["text"]  # словами пациента
    assert row["tu_qualifiers"] == PAY


def test_condition_on_the_whole_list_reaches_every_position(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[(None, "по омс", PAY)])
    [turn], [row] = _run(["ферритин и глюкоза по омс сколько стоят"], oracle, mini_prod_world, monkeypatch, _sid("whole"))
    assert not _priced(turn["text"]), turn["text"]
    assert row["tu_qualifiers_loose"] == PAY


# --------------------------------------------------------------------------- одна услуга и вопрос после цены


def test_one_service_with_a_condition_is_not_found(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин",), inner=_Rows(("Ферритин",)), qualifiers=[("ферритин", "по омс", PAY)])
    [turn], _rows = _run(["сколько стоит ферритин по омс"], oracle, mini_prod_world, monkeypatch, _sid("one"))
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]


@pytest.mark.parametrize(("follow_up", "condition"), [("а сколько стоит по омс?", "по омс"), ("а сколько стоит по полису?", "по полису")])
def test_condition_after_a_price_does_not_take_the_shown_row(follow_up, condition, mini_prod_world, monkeypatch):  # noqa: F811
    """BUG-2026-10-27-OMS-FOLLOWUP-PRICED: второй ход — условие без названия; показанная строка («контекст = показанное») его не
    обходит — судья П2 по услуге контекста."""

    oracle = SpansOracle(("ферритин",), inner=_Rows(("Ферритин",)), qualifiers=[(None, condition, PAY)])
    (first, second), rows = _run(["сколько стоит ферритин", follow_up], oracle, mini_prod_world, monkeypatch, _sid("follow", follow_up))
    assert _rub(first["text"], "Ферритин"), first["text"]
    assert not _priced(second["text"]) and "Не нашёл" in second["text"], second["text"]
    assert rows[-1]["price_condition_followup"] is True and rows[-1]["p2_llm_judge"] == "not_found"


# --------------------------------------------------------------------------- встречные


@pytest.mark.parametrize(("first", "row"), [("ферритин", "Ферритин"), ("ТТГ", "ТТГ")])  # сильная и слабая позиция
@pytest.mark.parametrize("attached", [True, False])
@pytest.mark.parametrize("words", ["сколько стоит", "какая цена"])
def test_price_words_marked_as_a_qualifier_keep_the_prices(words, attached, first, row, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс карточки 28.10: живая LLM разметки звала «оплатой» слова вопроса о цене — цену это не снимает."""

    reply = f"{words} {first} и глюкоза" if words == "сколько стоит" else f"{first} и глюкоза {words}"
    oracle = SpansOracle((first, "глюкоза"), inner=_Rows((row, "Глюкоза")), qualifiers=[(first if attached else None, words, PAY)])
    [turn], _rows = _run([reply], oracle, mini_prod_world, monkeypatch, _sid("words", words, first, str(attached)))
    assert _rub(turn["text"], row) and _rub(turn["text"], "Глюкоза"), turn["text"]


def test_list_without_qualifiers_keeps_both_prices(mini_prod_world, monkeypatch):  # noqa: F811
    oracle = SpansOracle(("ферритин", "глюкоза"), inner=_Rows(("Ферритин", "Глюкоза")), qualifiers=[])
    [turn], [row] = _run(["сколько стоит ферритин и глюкоза"], oracle, mini_prod_world, monkeypatch, _sid("plain"))
    assert _rub(turn["text"], "Ферритин") and _rub(turn["text"], "Глюкоза"), turn["text"]
    assert "tu_qualifiers" not in row and "price_condition_followup" not in row


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
    assert "tu_qualifiers" not in row


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
    assert ferritin.beside == "срочно" and glucose.beside == "срочно" and glucose.qualifiers == ()  # ни к одной — у каждой позиции
    assert [(q.kind, q.text) for q in parse.loose] == [(QualifierKind.CONDITION, "срочно")]


def test_answer_without_the_field_is_a_parse_without_qualifiers(monkeypatch):
    parse = _parse(monkeypatch, "ферритин и глюкоза", {"phrases": [[1, 1], [3, 3]]})
    assert parse.status == "picked" and all(item.qualifiers == () for item in parse.items) and parse.loose == ()


@pytest.mark.parametrize(("words", "asked"), [(16, True), (17, False)])
def test_long_list_is_marked_without_qualifiers(words, asked, monkeypatch):
    """Перечень длиннее 16 слов — прежним промптом: ответ с уточнениями растёт со списком и не укладывается в остаток секунд хода
    (карточка 28.10, g0711 — 28 слов, тайм-аут разметки)."""

    from ollama import GenerateResponse as _Response

    from messengers_router.runtime_config import config as cfg

    systems: list[str] = []

    class _Spy:
        async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
            systems.append(system or "")
            return _Response(model="m", response=json.dumps({"phrases": [[1, 1]], "qualifiers": []}), done=True)

    monkeypatch.setattr(cfg, "MR_POSITION_QUALIFIERS", "on", raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Spy())
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    reply = " ".join(["ферритин", *(f"глюкоза{k}" for k in range(words - 1))])
    asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=_CATALOG))
    assert systems and ('"qualifiers"' in systems[0]) is asked
