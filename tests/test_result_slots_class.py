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
import json
from pathlib import Path

import pytest

from messengers_router import classifier, policies, result_slots
from messengers_router import router as router_mod
from messengers_router.entity_grounder import ground_decision_entities
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.services import Services
from tests.test_booking_slots_by_meaning import _LLM
from tests.test_pending_buttons_class import _CLASSIFIER_MARK, _FREE_MARK, _ON, _free, _in_world
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура

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


@pytest.mark.parametrize("buttons", ["off", "on"])
@pytest.mark.parametrize("code", ["ЯВ", "Тд", "уб"])
def test_code_question_answer_is_code_by_llm(monkeypatch, code: str, buttons: str) -> None:
    """Встречный свип 24.10: на «Укажите код анализа» пациенты отвечают голым кодом из двух букв (13 из 24 ответов) — путь on
    принимает его по LLM хода (и при одной функции ответа на висящий вопрос), без LLM — переспрос."""

    _switch(monkeypatch)
    monkeypatch.setattr(result_slots._cfg, "MR_PENDING_BUTTONS", buttons, raising=False)
    by_reply = {"Сидорова 1985 123": {"surname": "Сидорова", "year": 1985, "number": 123}, code: {"filial": code}}

    async def llm(prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
        entities = by_reply.get(prompt.rsplit("Текст пациента:", 1)[-1].strip(), {})
        return {"label": "TEST_RESULT", "confidence": 0.9, "context_action": "continue", "entities": entities}

    monkeypatch.setattr(classifier, "ollama_classify_json", llm)
    (_, ask, link), entities = _say(monkeypatch, ["Хочу получить результаты анализов", "Сидорова 1985 123", code])
    assert ask == "Укажите код анализа (например: Бг)."
    assert entities.get("filial") == code and "getanaliz.php?" in link


# --- переспрос называет недостающее; ответ в том же порядке — остаток формы ----------------------------------------

_WHOLE_QUESTION = "фамилия, год рождения, код анализа, номер анализа"


@pytest.mark.parametrize(
    ("missing", "asks", "have"),
    [
        (["year", "filial", "number"], "Укажите ещё год рождения, код анализа и номер анализа", "Фамилия уже есть."),
        (["surname", "filial", "number"], "Укажите ещё фамилию, код анализа и номер анализа", "Год рождения уже есть."),
        (["filial", "number"], "Укажите ещё код анализа и номер анализа", "Фамилия и год рождения уже есть."),
    ],
)
def test_partial_clarify_names_only_the_rest(monkeypatch, missing: list[str], asks: str, have: str) -> None:
    _switch(monkeypatch)
    text = policies.clarification_question("TEST_RESULT", missing)
    assert text.startswith(asks) and text.endswith(have)
    assert _WHOLE_QUESTION not in text


def test_clarify_without_progress_and_single_field_as_before(monkeypatch) -> None:
    _switch(monkeypatch)
    assert _WHOLE_QUESTION in policies.clarification_question("TEST_RESULT", list(_ALL))
    assert policies.clarification_question("TEST_RESULT", ["filial"]) == "Укажите код анализа (например: Бг)."


def test_partial_clarify_off_as_before(monkeypatch) -> None:
    _switch(monkeypatch, "off")
    assert _WHOLE_QUESTION in policies.clarification_question("TEST_RESULT", ["year", "filial", "number"])


@pytest.mark.parametrize(
    ("text", "missing", "expected"),
    [
        ("1985, Бг, 123", ["year", "filial", "number"], {"year": 1985, "filial": "Бг", "number": 123}),
        ("1985 бг 123.", ["year", "filial", "number"], {"year": 1985, "filial": "бг", "number": 123}),
        ("ву 276", ["filial", "number"], {"filial": "ву", "number": 276}),
        ("сидорова, Бг, 123", ["surname", "filial", "number"], {"surname": "Сидорова", "filial": "Бг", "number": 123}),
        ("Сидорова; 1985", ["surname", "year"], {"surname": "Сидорова", "year": 1985}),
    ],
)
def test_rest_of_form_fills_the_rest(monkeypatch, text: str, missing: list[str], expected: dict) -> None:
    _switch(monkeypatch)
    assert {k: v for k, v in _fields(text, missing).items() if k in _ALL} == expected


@pytest.mark.parametrize(
    ("text", "missing"),
    [
        ("Сидорова, Бг", ["surname", "filial"]),  # без числа — не форма: слова решает LLM
        ("Бг", ["filial", "number"]),
        ("1985, Бг", ["year", "filial", "number"]),  # не все недостающие — не форма
        ("Результаты анализов", ["year", "filial", "number"]),
        ("код 5437", ["filial", "number"]),  # подпись графы — не значение
        ("номер 123", ["filial", "number"]),
        ("Код 123", ["filial", "number"]),
    ],
)
def test_not_the_rest_of_form_no_code(monkeypatch, text: str, missing: list[str]) -> None:
    _switch(monkeypatch)
    assert "filial" not in _fields(text, missing)


def _say(monkeypatch, texts: list[str]) -> tuple[list[str], dict]:
    monkeypatch.setattr(result_slots._cfg, "MR_NAMES_RESULT_SURNAME", "off", raising=False)
    state = SessionState(session_id="result-slots-say-" + "-".join(texts))
    memory = MemoryStore()
    services = Services()
    services.ensure_background_refresh_started = lambda: None

    async def go() -> list[str]:
        answers = []
        for text in texts:
            out = [env async for env in router_mod.patient_routing_stream(text, state, services, memory)]
            answers.append("".join(env.text for env in out if env.text))
        return answers

    return asyncio.run(go()), dict(state.last_entities)


@pytest.mark.parametrize(
    "turns",
    [
        ["Хочу получить результаты анализов", "Сидорова", "1985, Бг, 123"],
        ["Результаты анализов", "сидорова", "1985 бг 123"],
        ["Хочу получить результаты анализов", "1985", "Сидорова, Бг, 123"],
    ],
)
def test_dialog_progress_asks_the_rest_then_gives_link(monkeypatch, turns: list[str]) -> None:
    """Без LLM (сеть закрыта): реквизит принят — бот просит остальные, без оффера оператора; остаток формы — ссылка."""

    _switch(monkeypatch)
    (_, rest, link), entities = _say(monkeypatch, turns)
    assert rest.startswith("Укажите ещё") and "оператор" not in rest
    assert "getanaliz.php?" in link and "year=1985" in link and "nom2=123" in link
    assert str(entities.get("filial")).casefold() == "бг"


def test_dialog_off_progress_repeats_question(monkeypatch) -> None:
    _switch(monkeypatch, "off")
    (_, again, _), _ = _say(monkeypatch, ["Хочу получить результаты анализов", "1985", "Сидорова, Бг, 123"])
    assert _WHOLE_QUESTION in again and "Перевести на оператора?" in again


# --- ответ на вопрос об одном реквизите — решение одной функции ответа на висящий вопрос (LLM) ----------------------

# Стенд 24.10: живой классификатор код в `filial` не кладёт (кладёт город, всю реплику, код — в `order_id`), а одна функция ответа
# на висящий вопрос «ЯВ» на «Укажите код анализа» признаёт ответом, «да» и «не знаю код» — «нет».
_ASK_CODE = "Укажите код анализа (например: Бг)."
_BEFORE_CODE = ["Хочу получить результаты анализов", "Сидорова 1985 123"]


def _classifier(prompt: str) -> str:
    reply = prompt.rsplit("Текст пациента:", 1)[-1].strip()
    entities = {"surname": "Сидорова", "year": 1985, "number": 123} if reply == "Сидорова 1985 123" else {}
    return json.dumps({"label": "TEST_RESULT", "confidence": 0.9, "context_action": "continue", "entities": entities})


def _code_dialog(world, monkeypatch, reply: str, *, free: str | None, result_slots_llm: str = "on", buttons: dict | None = None):
    answers = {_CLASSIFIER_MARK: _classifier, **({_FREE_MARK: _free(free)} if free else {})}
    switches = {**(buttons or _ON), "MR_RESULT_SLOTS_LLM": result_slots_llm}
    return _in_world(world, monkeypatch, [*_BEFORE_CODE, reply], switches=switches, llm=_LLM(answers))


@pytest.mark.parametrize("code", ["ЯВ", "Тд", "уб"])
def test_code_answer_by_the_one_function_gives_the_link(code: str, mini_prod_world, monkeypatch) -> None:  # noqa: F811
    _, ask, link = _code_dialog(mini_prod_world, monkeypatch, code, free="answer")
    assert ask["text"] == _ASK_CODE
    assert link["state"].get("filial") == code and "getanaliz.php?" in link["text"], link


@pytest.mark.parametrize("reply", ["да", "не знаю код"])
def test_not_an_answer_is_not_the_code(reply: str, mini_prod_world, monkeypatch) -> None:  # noqa: F811
    from messengers_router.router import TEST_RESULT_DATA_HINT_TEXT

    turn = _code_dialog(mini_prod_world, monkeypatch, reply, free="no")[-1]
    assert "filial" not in turn["state"] and turn["text"] == TEST_RESULT_DATA_HINT_TEXT, turn


@pytest.mark.parametrize("reply", ["Сидорова 1985 123", "код анализа", "не знаю какой"])
def test_answer_of_the_wrong_kind_is_not_the_code(reply: str, mini_prod_world, monkeypatch) -> None:  # noqa: F811
    """Модель сочла ответом, но это не код по виду (несколько слов, подпись графы) — кода нет, бот переспросит."""

    turn = _code_dialog(mini_prod_world, monkeypatch, reply, free="answer")[-1]
    assert "filial" not in turn["state"] and "getanaliz.php?" not in turn["text"], turn


def test_code_answer_without_the_one_function_asks_again(mini_prod_world, monkeypatch) -> None:  # noqa: F811
    """LLM ответа на висящий вопрос нет (сбой или `pending_buttons = off`), классификатор кода не назвал — переспрос."""

    turn = _code_dialog(mini_prod_world, monkeypatch, "ЯВ", free=None)[-1]
    assert "filial" not in turn["state"] and turn["text"].startswith(_ASK_CODE), turn


def test_code_answer_off_as_before(mini_prod_world, monkeypatch) -> None:  # noqa: F811
    turn = _code_dialog(mini_prod_world, monkeypatch, "ЯВ", free=None, result_slots_llm="off")[-1]
    assert turn["state"].get("filial") == "ЯВ" and "getanaliz.php?" in turn["text"], turn


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


# --- промпты: часть реквизитов — ответ; код анализа — не город (24.10, решения владельца) ------------------------------

_ROOT = Path(__file__).resolve().parents[1]


def _prompt(name: str) -> str:
    text = (_ROOT / "messengers_router" / "prompts" / f"{name}.txt").read_text(encoding="utf-8")
    assert text == (_ROOT / "app_data" / "prompts" / f"mr_{name}.txt").read_text(encoding="utf-8"), "промпт в двух локациях разошёлся"
    return text


def test_free_reply_prompt_counts_part_of_the_data_as_an_answer() -> None:
    """Голая фамилия на вопрос о четырёх реквизитах — ответ, не «нет» (карточка 24.10: «Сидорова» → подсказка и оффер)."""

    assert "часть из них — тоже \"answer\"" in _prompt("choose_pending_free_reply")


def test_classifier_prompt_two_locations_equal() -> None:
    """Правка промпта классификатора 24.10 (пояснение filial) откатана по карточке: dev — 19 смен темы по всему боту."""

    _prompt("classifier_patient")


# --- правило лаб-кода: lab_code_by_llm = on — правило молчит, ход решает LLM ---------------------------------------


def _rule(text: str):
    return asyncio.run(classifier.deterministic_rule_decision(text, {}, allow_refine=False, attach_secondary=False))


@pytest.mark.parametrize("text", ["Са-125", "CA 19-9", "Ву 148", "Тк 404"])
def test_lab_code_rule_silent_on(monkeypatch, text: str) -> None:
    monkeypatch.setattr(result_slots._cfg, "MR_LAB_CODE_BY_LLM", "on", raising=False)
    decision = _rule(text)
    assert decision is None or "rule_test_assist_lab_code" not in decision.flags


@pytest.mark.parametrize("text", ["Са-125", "Ву 148"])
def test_lab_code_rule_off_as_before(monkeypatch, text: str) -> None:
    monkeypatch.setattr(result_slots._cfg, "MR_LAB_CODE_BY_LLM", "off", raising=False)
    assert "rule_test_assist_lab_code" in _rule(text).flags


# --- ответ на вопрос о реквизитах — остаток формы: ход остаётся в результатах (прод #659: «ву 276») -----------------

_WAITS_CODE_NUMBER = {"_pending": {"label": "TEST_RESULT", "missing": ["filial", "number"]}}


def _rule_in(text: str, last_entities: dict):
    return asyncio.run(classifier.deterministic_rule_decision(text, last_entities, allow_refine=False, attach_secondary=False))


@pytest.mark.parametrize(("text", "code", "number"), [("ву 276", "ву", 276), ("Ву 148", "Ву", 148), ("ТМ, 192", "ТМ", 192)])
def test_rest_of_form_answer_stays_in_results(monkeypatch, text: str, code: str, number: int) -> None:
    _switch(monkeypatch)
    decision = _rule_in(text, _WAITS_CODE_NUMBER)
    assert decision.label == "TEST_RESULT" and "rule_test_result_rest_of_form" in decision.flags
    assert decision.entities == {"filial": code, "number": number}


@pytest.mark.parametrize(
    ("text", "last_entities"),
    [
        ("Са-125", _WAITS_CODE_NUMBER),  # маркер посреди сбора реквизитов — не остаток формы: правило лаб-кода как было
        ("ву 276", {}),  # реквизитов не ждём — как было
        ("ву 276", {"_pending": {"label": "TEST_RESULT", "missing": ["surname", "year", "filial", "number"]}}),
    ],
)
def test_not_rest_of_form_lab_code_rule_as_before(monkeypatch, text: str, last_entities: dict) -> None:
    _switch(monkeypatch)
    assert "rule_test_assist_lab_code" in _rule_in(text, last_entities).flags


def test_rest_of_form_rule_off_as_before(monkeypatch) -> None:
    _switch(monkeypatch, "off")
    assert "rule_test_assist_lab_code" in _rule_in("ву 276", _WAITS_CODE_NUMBER).flags


def test_dialog_code_and_number_after_surname_and_year_gives_link(monkeypatch) -> None:
    """Прод #659: «Сидорова 1985» → «Укажите ещё код анализа и номер анализа…» → «ву 276» — ссылка, а не подбор анализов."""

    _switch(monkeypatch)
    by_reply = {"Сидорова 1985": {"surname": "Сидорова", "year": 1985}}

    async def llm(prompt: str, *, queue_timeout_ms: int = 30000) -> dict:
        entities = by_reply.get(prompt.rsplit("Текст пациента:", 1)[-1].strip(), {})
        return {"label": "TEST_RESULT", "confidence": 0.9, "context_action": "continue", "entities": entities}

    monkeypatch.setattr(classifier, "ollama_classify_json", llm)
    (_, ask, link), entities = _say(monkeypatch, ["Хочу получить результаты анализов", "Сидорова 1985", "ву 276"])
    assert ask.startswith("Укажите ещё код анализа и номер анализа")
    assert (entities.get("filial"), entities.get("number")) == ("ву", 276) and "getanaliz.php?" in link
