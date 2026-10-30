"""Класс «одна позиция или семейство» (BUG-2026-10-31-PRICE-FAMILY-ONE-ROW, DESIGN-2026-10-31-PRICE-FAMILY-SCOPE; выключатель
`price_scope_llm`, off).

Пациент назвал услугу, у которой в прайсе несколько вариантов с разной ценой, и не сказал, какой («ОАК», «приём уролога»). Выбор
строки цены сводил их к одной: промпт «ИМЕННО та услуга» и его правило 4, затем `_prefer_base_variants` по буквам названия
(к.м.н., повторный). Карточка d23on_btn: 21 из 26 «одна цена вместо списка» — выбор LLM. Решение владельца 22.07: при
неоднозначности — список; 30.10 (В1): варианты врача (к.м.н., повторный, категория) — в списке, варианты условия (cito, на дому,
капиллярная кровь, для ребёнка) — только если пациент о них сказал; (В3) видимых строк — 8.

Инварианты (через обработчик эндпоинта, мини-прод, оракул LLM настоящего типа):
- выбор вернул одну строку, вопрос «что просит» ответил «семейство» — в ответе все варианты этой услуги с ценами, варианты врача
  есть, вариантов условия нет;
- «одна позиция» — ответ как без выключателя, второго выбора нет;
- вопрос «что просит» не ответил — ответ как без выключателя;
- shadow — вопрос задан, ответ как без выключателя;
- off — путь прода: одна цена (красное, `xfail strict`).
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.test_qualifier_round_two_class import _run
from tests.test_turn_parse_t3 import _Oracle

_SCOPE_MARK = "об одной конкретной позиции или об услуге"
_FAMILY_MARK = "ВСЕХ позиций, которые являются вариантами"
_PRICED = re.compile(r"^\s*(?:\d+[.)]|[-—•])?\s*(?P<name>.+?) — \d[\d\s]* руб", re.M)

# Реплика · фраза разбора · строка первого выбора · начала названий вариантов семейства (оракул второго выбора) ·
# варианты, которые должны быть в ответе · варианты, которых быть не должно.
_CASES = {
    "oak": (
        "сколько стоит ОАК",
        "ОАК",
        "Общий анализ крови (Le, Er, Hb, СОЭ)",
        ("Общий анализ крови", "ОАК (без", "Cito Общий анализ крови"),
        ("Общий анализ крови (Le, Er, Hb, СОЭ)", "Общий анализ крови (полный)", "ОАК (без лейкоцитарной формулы и СОЭ)"),
        ("Cito", "капиллярная"),
    ),
    "urologist": (
        "сколько стоит приём уролога",
        "приём уролога",
        "Прием (осмотр, консультация) врача-уролога",
        ("Прием (осмотр, консультация) врача-уролога", "Прием (осмотр, консультация) врача- уролога, к.м.н."),
        ("Прием (осмотр, консультация) врача-уролога", "Прием (осмотр, консультация) врача- уролога, к.м.н."),
        (),
    ),
}


class _Scope(_Oracle):
    """Разбор — фраза случая; первый выбор строки — одна строка `first`; вопрос «что просит» — исход `scope` (пусто — мусор);
    второй выбор — строки, чьё название начинается с одного из `family`."""

    def __init__(self, phrase: str, first: str, family: tuple[str, ...], scope: str) -> None:
        super().__init__((phrase,), rows=(first,))
        self.first, self.family, self.scope = first, family, scope
        self.scope_asked = 0
        self.family_asked = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _SCOPE_MARK in seen:
            self.scope_asked += 1
            body = json.dumps({"picks": [], "outcome": self.scope}) if self.scope else "не JSON"
            return GenerateResponse(model="m", response=body, done=True)
        if _FAMILY_MARK in seen:
            self.family_asked += 1
            picks = [n for n, text in options if text.startswith(self.family)]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks, "reason": "none"}), done=True)
        response = await super().generate(prompt=prompt, system=system, **kwargs)
        if self.rows and options and any(text.startswith(self.first) for _n, text in options):
            # Первый выбор — ровно одна строка `first` (дефект класса: семейство сведено к одной), а не все с этим началом.
            exact = [n for n, text in options if text.split(" — ", 1)[0] == self.first]
            if exact:
                return GenerateResponse(model="m", response=json.dumps({"picks": exact[:1], "reason": "none", "search_terms": []}), done=True)
        return response


def _names(text: str) -> list[str]:
    return [match.group("name").strip() for match in _PRICED.finditer(text)]


def _turn(case: str, scope: str, mode: str, world, monkeypatch):
    reply, phrase, first, family, _want, _not = _CASES[case]
    llm = _Scope(phrase, first, family, scope)
    record, rows = _run([reply], world, monkeypatch, llm=llm, session=f"pfs-{case}-{scope or 'junk'}-{mode}", MR_PRICE_SCOPE_LLM=mode)
    return str(record[-1].get("answer") or record[-1].get("text") or ""), rows[-1], llm


@pytest.mark.parametrize(
    "mode",
    ["on", pytest.param("off", marks=pytest.mark.xfail(strict=True, reason="BUG-2026-10-31-PRICE-FAMILY-ONE-ROW: путь прода (`price_scope_llm = off`)"))],
)
@pytest.mark.parametrize("case", _CASES)
def test_family_shows_every_variant_without_unasked_conditions(case, mode, mini_prod_world, monkeypatch):  # noqa: F811
    answer, row, llm = _turn(case, "family", mode, mini_prod_world, monkeypatch)
    names = _names(answer)
    _reply, _phrase, _first, _family, want, never = _CASES[case]
    for prefix in want:
        assert any(name.startswith(prefix) for name in names), (prefix, answer)
    for word in never:
        assert not any(word in name for name in names), (word, answer)
    assert row.get("price_family_scope") == "family" and row.get("price_policy") == "family", row
    assert 2 <= len(names) <= 8 + 1, answer


@pytest.mark.parametrize("case", _CASES)
def test_one_position_keeps_the_single_row(case, mini_prod_world, monkeypatch):  # noqa: F811
    answer, row, llm = _turn(case, "one", "on", mini_prod_world, monkeypatch)
    off, _off_row, _ = _turn(case, "one", "off", mini_prod_world, monkeypatch)
    assert llm.scope_asked == 1 and llm.family_asked == 0
    assert row.get("price_family_scope") == "one"
    assert _names(answer) == _names(off), (answer, off)


@pytest.mark.parametrize("case", _CASES)
def test_scope_failure_and_shadow_answer_as_off(case, mini_prod_world, monkeypatch):  # noqa: F811
    off, _row, _ = _turn(case, "family", "off", mini_prod_world, monkeypatch)
    junk, junk_row, junk_llm = _turn(case, "", "on", mini_prod_world, monkeypatch)
    shadow, shadow_row, shadow_llm = _turn(case, "family", "shadow", mini_prod_world, monkeypatch)
    assert _names(junk) == _names(off) and junk_row.get("price_family_scope") == "unavailable" and junk_llm.family_asked == 0
    assert _names(shadow) == _names(off) and shadow_row.get("price_family_scope") == "family" and shadow_llm.family_asked == 0


def test_scope_question_sees_the_picked_row_beyond_the_names_cap(monkeypatch) -> None:
    """Выбранная строка дальше потолка названий в пуле (`_SCOPE_NAMES`) — вопрос «что просит» всё равно видит её первой."""

    import asyncio

    from messengers_router.choice import Choice, Status
    from messengers_router.services import _price_select_llm as sel

    pool = [{"serviceName": f"Удаление {i}", "cost": 1000 + i} for i in range(20)]
    picked = {"serviceName": "Геморроидэктомия (осложненная), 3-4 стадии", "cost": 50000}
    pool.append(picked)
    seen: dict[str, str] = {}

    async def fake_choose(task, question, options, **kw):
        seen["context"] = kw.get("context", "")
        return Choice(Status.PICKED, outcome="one")

    monkeypatch.setattr(sel, "choose", fake_choose)
    monkeypatch.setattr(sel._cfg, "MR_PRICE_SCOPE_LLM", "on", raising=False)
    assert asyncio.run(sel._family_rows("удаление геморроя 3-4 степени, цены", [picked], pool)) is None
    lines = seen["context"].splitlines()
    assert lines[0] == picked["serviceName"] and len(lines) == sel._SCOPE_NAMES


def test_prompts_are_the_same_in_both_locations() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("choose_price_scope", "choose_price_service_family"):
        assert (root / "messengers_router/prompts" / f"{name}.txt").read_bytes() == (root / "app_data/prompts" / f"mr_{name}.txt").read_bytes(), name
