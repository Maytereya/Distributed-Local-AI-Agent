"""Реквизиты результатов анализов: слот не берёт команду как данные (BUG-2026-10-24-RESULT-SLOT-COMMAND, `result_slots_llm = on`).

Инварианты класса:
- код анализа (`filial`) — только форма (полный порядок «фамилия, год, код, номер», «код Бг») или сущность LLM-классификатора хода;
  ни служебное слово в любой форме и регистре, ни фамилия, ни город, ни «да / нет» без формы и LLM кодом не становятся — бот
  переспросит (решение владельца 22.10 (а));
- сущность LLM — только слова реплики подряд и не слово другого реквизита хода (одно слово — один реквизит);
- путь `on` не зовёт прежнюю догадку (`policies._guess_result_code`).
ПДн — только выдуманные (репо публичный).
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from messengers_router import classifier, policies, result_slots
from messengers_router import router as router_mod
from messengers_router.entity_grounder import ground_decision_entities
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services

_ALL = ["surname", "year", "filial", "number"]


def _switch(monkeypatch, value: str = "on") -> None:
    monkeypatch.setattr(result_slots._cfg, "MR_RESULT_SLOTS_LLM", value, raising=False)


def _fields(text: str, missing: list[str]) -> dict:
    out: dict = {}
    policies._fill_test_result_entities(text, missing, out)
    return out


# --- без формы и LLM код не угадывается ------------------------------------------------------------------------------

# Служебные слова в разных формах и регистрах, фамилии, ответы «да / нет», города, фразы о результатах, код без значения.
_NOT_A_CODE = [
    "Сидорова", "сидорова", "СИДОРОВА", "Петрова-Водкина",
    "Результаты", "результатов", "РЕЗУЛЬТАТЫ АНАЛИЗОВ", "Анализы", "анализом",
    "Хочу", "Подскажите", "Пожалуйста", "да", "Да", "нет", "ок", "Стоп", "Отмена",
    "Самара", "в Тольятти", "Хочу результаты, я из Самары", "филиал Ленина, хочу узнать результат",
    "Хочу получить результаты анализов", "код не помню", "код 5437", "Бг",
]


@pytest.mark.parametrize("missing", [_ALL, ["filial"], ["filial", "number"]], ids=["all", "code", "code_number"])
@pytest.mark.parametrize("text", _NOT_A_CODE)
def test_no_form_no_llm_no_code(monkeypatch, text: str, missing: list[str]) -> None:
    """«Бг» тоже: голое слово — код только по LLM хода (без LLM — переспрос), иначе кодом становится и «да»."""

    _switch(monkeypatch)
    assert "filial" not in _fields(text, missing)


@pytest.mark.parametrize("text", ["Сидорова", "Результаты", "да", "Самара", "код Бг"])
def test_off_as_before(monkeypatch, text: str) -> None:
    _switch(monkeypatch, "off")
    assert "filial" in _fields(text, _ALL)


# --- форма даёт код -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Сидорова, 1985, Бг, 123", "Бг"),
        ("сидорова 1985 бг 123", "бг"),
        ("код Бг", "Бг"),
        ("Код анализа: бг", "бг"),
        ("КОД АНАЛИЗА БГ", "БГ"),
        ("мой код — ву", "ву"),
    ],
)
def test_form_gives_code(monkeypatch, text: str, code: str) -> None:
    _switch(monkeypatch)
    assert _fields(text, _ALL)["filial"] == code


def test_code_marker_with_number(monkeypatch) -> None:
    """«код ву 276» (как в прод #659): код — слово после маркера, номер — число."""

    _switch(monkeypatch)
    assert _fields("код ву 276", ["filial", "number"]) == {"filial": "ву", "number": 276}


# --- одно число — один реквизит ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "missing", "expected"),
    [
        ("1985", _ALL, {"year": 1985}),
        ("1985", ["filial", "number"], {"number": 1985}),  # год уже есть — это номер, известный год не перетирается
        ("1985", ["surname"], {}),  # ни год, ни номер не ждём — не перетирать ни один
        ("номер 1985", _ALL, {"number": 1985}),
        ("номер анализа: 2001", _ALL, {"number": 2001}),
        ("год рождения 1985", _ALL, {"year": 1985}),
        ("Сидорова 1985", _ALL, {"year": 1985}),
        ("1985 123", _ALL, {"year": 1985, "number": 123}),
    ],
)
def test_one_number_is_one_field(monkeypatch, text: str, missing: list[str], expected: dict) -> None:
    _switch(monkeypatch)
    out = _fields(text, missing)
    assert {k: out[k] for k in ("year", "number") if k in out} == expected


def test_order_number_is_not_also_the_year(monkeypatch) -> None:
    _switch(monkeypatch)
    out = _fields("заказ 1985", _ALL)
    assert (out.get("number"), out.get("order_id"), "year" in out) == (1985, "1985", False)


def test_one_number_off_as_before(monkeypatch) -> None:
    _switch(monkeypatch, "off")
    assert _fields("1985", _ALL) == {"year": 1985, "number": 1985}


# --- сущность LLM: слова реплики, одно слово — один реквизит -------------------------------------------------------


def _ground(text: str, entities: dict) -> tuple[dict, set]:
    services = Services()
    services.ensure_background_refresh_started = lambda: None
    decision = RouteDecision(label="TEST_RESULT", confidence=0.9, entities=dict(entities))
    result = asyncio.run(ground_decision_entities(decision, text, SessionState(session_id="result-slots-ground"), services))
    return dict(result.entities), set(result.flags)


@pytest.mark.parametrize(("text", "said"), [("Бг", "Бг"), ("бг", "Бг"), ("код Бг", "код Бг"), ("мой код бг, номер 123", "бг")])
def test_llm_code_in_reply_is_kept(monkeypatch, text: str, said: str) -> None:
    _switch(monkeypatch)
    entities, _ = _ground(text, {"filial": said})
    assert entities["filial"].casefold() == "бг"


@pytest.mark.parametrize("text", ["Сидорова", "сидорова", "СИДОРОВА"])
def test_llm_surname_word_is_not_also_the_code(monkeypatch, text: str) -> None:
    _switch(monkeypatch)
    entities, flags = _ground(text, {"surname": "Сидорова", "filial": text})
    assert entities == {"surname": "Сидорова"}
    assert "entity_dropped_result_code" in flags


def test_llm_number_is_not_also_the_code(monkeypatch) -> None:
    _switch(monkeypatch)
    entities, _ = _ground("ву 276", {"filial": "276", "number": 276})
    assert "filial" not in entities


@pytest.mark.parametrize(("text", "said"), [("Сидорова", "Бг"), ("Хочу получить результаты анализов", "Бг")])
def test_llm_code_not_in_reply_is_dropped(monkeypatch, text: str, said: str) -> None:
    """Код, которого в реплике нет (модель взяла его из примера или памяти), не принимается."""

    _switch(monkeypatch)
    assert "filial" not in _ground(text, {"filial": said})[0]


def test_llm_code_off_as_before(monkeypatch) -> None:
    _switch(monkeypatch, "off")
    assert _ground("Сидорова", {"surname": "Сидорова", "filial": "Сидорова"})[0]["filial"] == "Сидорова"


# --- диалог через роутер ----------------------------------------------------------------------------------------------


def _dialog(monkeypatch, second: str, llm_entities: dict | None) -> dict:
    if llm_entities is not None:
        async def llm(prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
            entities = llm_entities if second in prompt else {}
            return {"label": "TEST_RESULT", "confidence": 0.9, "context_action": "continue", "entities": entities}

        monkeypatch.setattr(classifier, "ollama_classify_json", llm)
    state = SessionState(session_id=f"result-slots-{second}-{bool(llm_entities)}")
    memory = MemoryStore()
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def go() -> None:
        for text in ("Хочу получить результаты анализов", second):
            [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]

    asyncio.run(go())
    return dict(state.last_entities)


@pytest.mark.parametrize("llm", [None, {"surname": "Сидорова", "filial": "Сидорова"}], ids=["no_llm", "llm_both"])
@pytest.mark.parametrize("second", ["Сидорова", "сидорова"])
def test_dialog_surname_is_not_the_code(monkeypatch, second: str, llm: dict | None) -> None:
    _switch(monkeypatch)
    if llm is not None:
        llm = {"surname": "Сидорова", "filial": second}
    assert "filial" not in _dialog(monkeypatch, second, llm)


def test_dialog_code_named_by_llm(monkeypatch) -> None:
    _switch(monkeypatch)
    assert _dialog(monkeypatch, "Бг", {"filial": "Бг"}).get("filial") == "Бг"


# --- путь on не зовёт прежнюю догадку (AST) --------------------------------------------------------------------------

_POLICIES = Path(__file__).resolve().parents[1] / "messengers_router" / "policies.py"


def _is_on(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "on"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "result_slots"
    )


def test_on_path_does_not_call_the_guess() -> None:
    tree = ast.parse(_POLICIES.read_text(encoding="utf-8"))
    callers = [
        (func, call)
        for func in ast.walk(tree)
        if isinstance(func, ast.FunctionDef)
        for call in ast.walk(func)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_guess_result_code"
    ]
    assert [func.name for func, _ in callers] == ["_fill_test_result_entities"]
    func, call = callers[0]
    branch = next(
        node for node in ast.walk(func)
        if isinstance(node, ast.If) and _is_on(node.test) and any(call is n for s in node.orelse for n in ast.walk(s))
    )
    assert not any(call is n for s in branch.body for n in ast.walk(s))


def test_switch_is_a_config_field_off_by_default() -> None:
    from agent_logic_2 import config

    assert config.MR_RESULT_SLOTS_LLM == "off"
