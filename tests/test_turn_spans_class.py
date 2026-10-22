"""Разбор реплики без списков слов: LLM размечает номера слов (DESIGN-2026-10-23-NO-STOPWORDS, этап 1, `turn_spans_llm`).

Инварианты класса:
- код не выкидывает из реплики ни одного слова: LLM видит все слова с номерами, в том числе «сколько», «анализы», «услуги» —
  что услуга, решает она (решение владельца 22.10, BUG-2026-10-22-CATALOG-CONFIRM-BY-LETTERS: «анализ» / «анализы» различал список);
- фраза — подстрока реплики по номерам слов; ответ вне контракта (номер вне списка, пропуск в группе, пересечение) — INVALID;
- без LLM действует только реплика-название или код клиники целиком и одно слово; сбой LLM — UNAVAILABLE без разреза;
- модуль разбора не читает ни одного из списков стоп-слов.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import pytest

from messengers_router import llm_runtime, turn_understanding
from messengers_router.services import _turn_spans as S

_CATALOG = [
    {"serviceName": "Ферритин", "serviceHomecode": "408", "cost": 530},
    {"serviceName": "Глюкоза", "serviceHomecode": "152", "cost": 190},
    {"serviceName": "Общий анализ крови (Le, Er, Hb, СОЭ)", "serviceHomecode": "502", "cost": 390},
]


def _parse(monkeypatch, reply: str, answer) -> tuple:
    seen: dict = {}

    async def fake(**kw):
        seen.update(kw)
        if isinstance(answer, Exception):
            raise answer
        return json.dumps(answer) if not isinstance(answer, str) else answer, {}

    monkeypatch.setattr(S.llm_runtime, "generate_structured", fake)
    return asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=_CATALOG)), seen


@pytest.mark.parametrize("reply", ["Сколько стоит анализ на ферритин", "Сколько стоят анализы на ферритин", "Цена анализов на ферритин"])
def test_every_word_reaches_llm_whatever_its_form(monkeypatch, reply: str) -> None:
    _, seen = _parse(monkeypatch, reply, {"phrases": [[len(reply.split())]]})
    numbered = seen["user"].split("Слова сообщения с номерами:")[1]
    assert all(f"{k}:{w}" in numbered for k, w in enumerate(reply.split(), 1))


def test_picked_span_is_substring_with_data_rows(monkeypatch) -> None:
    parse, _ = _parse(monkeypatch, "Сколько стоит анализ на ферритин", {"phrases": [[5]]})
    assert parse.status == "picked" and parse.source == "llm"
    item = parse.items[0]
    assert item.text == "ферритин" and item.rows and item.rows[0]["serviceHomecode"] == "408"


@pytest.mark.parametrize("group", [[3, 5], [3, 4, 5]])
def test_range_or_full_run_is_one_phrase(monkeypatch, group) -> None:
    """Группа — `[первое, последнее]` (так отвечает модель: карточка 22.10) или перечень подряд — одна фраза."""

    parse, _ = _parse(monkeypatch, "Сколько стоит анализ на ферритин", {"phrases": [group]})
    assert parse.status == "picked" and [i.text for i in parse.items] == ["анализ на ферритин"]


def test_several_groups_are_a_list(monkeypatch) -> None:
    parse, _ = _parse(monkeypatch, "ферритин, глюкоза — сколько?", {"phrases": [[1], [2]]})
    assert [i.text for i in parse.items] == ["ферритин", "глюкоза"]


def test_empty_answer_is_none_fit(monkeypatch) -> None:
    parse, _ = _parse(monkeypatch, "Сколько стоят анализы?", {"phrases": []})
    assert parse.status == "none_fit" and not parse.items


@pytest.mark.parametrize(
    "answer",
    [{"phrases": [[0]]}, {"phrases": [[9]]}, {"phrases": [[3, 1]]}, {"phrases": [[1, 2, 4]]}, {"phrases": [[1, 2], [2, 2]]}, {"x": 1}, "не json"],
)
def test_contract_violation_is_invalid(monkeypatch, answer) -> None:
    parse, _ = _parse(monkeypatch, "ферритин и глюкоза сколько", answer)
    assert parse.status == "invalid" and not parse.items and not parse.split


@pytest.mark.parametrize("exc", [TimeoutError("llm"), llm_runtime.LLMBudgetExhaustedError("budget")])
def test_llm_failure_gives_no_letter_cut(monkeypatch, exc) -> None:
    parse, _ = _parse(monkeypatch, "ферритин, глюкоза — сколько?", exc)
    assert parse.status == "unavailable" and not parse.items and not parse.split


@pytest.mark.parametrize(
    ("reply", "status"),
    [("", "empty"), ("???", "empty"), ("Ферритин", "strong_whole"), ("408", "strong_whole"), ("хгч", "single")],
)
def test_without_llm_only_whole_name_code_or_one_word(monkeypatch, reply: str, status: str) -> None:
    async def boom(**_kw):
        raise AssertionError("LLM не нужна")

    monkeypatch.setattr(S.llm_runtime, "generate_structured", boom)
    parse = asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=_CATALOG))
    assert parse.status == status


def test_question_words_around_a_name_still_go_to_llm(monkeypatch) -> None:
    """«Сколько стоит ферритин» — не дословное название: слова вопроса код не снимает, решает LLM."""

    parse, seen = _parse(monkeypatch, "Сколько стоит ферритин", {"phrases": [[3]]})
    assert seen and parse.status == "picked" and parse.items[0].text == "ферритин"


def test_too_long_reply_is_too_many(monkeypatch) -> None:
    async def boom(**_kw):
        raise AssertionError("LLM не нужна")

    monkeypatch.setattr(S.llm_runtime, "generate_structured", boom)
    parse = asyncio.run(S.parse_turn_spans(" ".join(["слово"] * 41), facet=None, catalog=_CATALOG))
    assert parse.status == "too_many"


_STOPWORD_NAMES = {
    "_PRICE_QUERY_STOPWORDS", "_PRICE_SPEECH_ACT_STOPWORDS", "_MULTI_PRICE_NOISE_WORDS", "_MULTI_PRICE_NOISE_RE",
    "_PRICE_GENERIC_SERVICE_TOKENS", "_PRICE_QUERY_SERVICE_NOISE_TOKENS", "_GENERIC_SERVICE_PREFIXES", "_SPEECH_WRAPPER",
    "_WRAPPER", "_FILLER", "_READINESS_LEADING_FILLER", "_DATETIME_FILLER_RE", "SERVICE_GENERIC_STOPWORDS",
}


def test_module_reads_no_stopword_list() -> None:
    tree = ast.parse(Path(S.__file__).read_text(encoding="utf-8"))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
    }
    assert not names & _STOPWORD_NAMES


def test_switch_routes_understanding_to_spans(monkeypatch) -> None:
    called: list[str] = []

    async def spans(text, **_kw):
        called.append(text)
        return S._result(text, S.ParseStatus.NONE_FIT, "llm", 3)

    monkeypatch.setattr(turn_understanding._turn_spans, "parse_turn_spans", spans)
    monkeypatch.setattr(turn_understanding._turn_spans, "enabled", lambda: True)
    monkeypatch.setattr(turn_understanding, "_catalog_rows", lambda _r: _CATALOG)
    parse = asyncio.run(turn_understanding._parse_phrases("сколько стоят анализы", facet=None, tu=None, agree=False))
    assert called == ["сколько стоят анализы"] and parse.status == "none_fit"


def test_long_list_gets_answer_budget_and_time(monkeypatch) -> None:
    """Перечень из 15 позиций: ответ — 15 групп по 7–8 токенов; бюджет и ожидание растут со словами (карточка 2, 22.10)."""

    reply = ", ".join(["ферритин", "глюкоза", "ттг"] * 5)
    parse, seen = _parse(monkeypatch, reply, {"phrases": [[k] for k in range(1, 16)]})
    assert parse.status == "picked" and len(parse.items) == 15
    assert seen["options"]["num_predict"] >= 8 * 15 and seen["timeout_s"] > 5.0
