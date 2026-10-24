"""Подготовка читает фразы разбора хода (`prepare_by_parse`, STEP-2026-10-24-PREPARE-BY-PARSE; DESIGN-2026-10-23-NO-STOPWORDS, этап 2).

Было: предмет памятки — реплика без слов пяти списков подготовки (`_PREPARE_QUERY_STOPWORDS`,
`_PREPARE_SERVICE_INFO_GENERIC_TOKENS`, `_PREPARE_GENERIC_HEADINGS`, `_GENERIC_PREPARE_QUERY_RE`, `_VARIANT_NOISE_RE`): «анализу»
в списке общих слов есть, «анализам» — нет, и «Подготовка к анализам крови» получала памятку Нечипоренко
(BUG-2026-10-22-PREPARE-GENERAL-BY-LETTERS). Решения владельца 22.10: списки удалить, без LLM услугу не угадывать.

Инварианты класса (путь `on`):
- словоформы одного вопроса — один ответ: фраза разбора («анализу на ферритин», «ферритину», «анализы на ферритин») находит ту
  же памятку, общее слово фразы весит по данным (редкость в памятках МИС), а не по списку;
- ни одной услуги в разборе — без поиска по словам: среди вариантов LLM только общая памятка о крови (если включена), памяток
  процедур нет; не подошла — просьба назвать анализ;
- сбой разбора — «уточните, к какому анализу или процедуре», без выбора памятки и без запасного пути по буквам;
- без LLM выбора «названа» только карточка, которую фраза называет по данным МИС; чужая карточка не выбирается никогда;
- путь `on` не читает ни одного из пяти списков (AST, по графу вызовов); путь `off` — прежний.
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path

import pytest

from agent_logic_2 import config as cfg
from messengers_router import shown_context, turn_understanding
from messengers_router.memory import MemoryStore
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.planner import build_plan
from messengers_router.services import Services
from messengers_router.services import _prepare as P
from messengers_router.services import _prepare_select_llm as select_mod
from messengers_router.services import _strong_match, _turn_spans
from messengers_router.services import prepare as prepare_mod

_FIVE = (
    "_PREPARE_QUERY_STOPWORDS",
    "_PREPARE_SERVICE_INFO_GENERIC_TOKENS",
    "_PREPARE_GENERIC_HEADINGS",
    "_GENERIC_PREPARE_QUERY_RE",
    "_VARIANT_NOISE_RE",
)

_PRICE = [
    {"serviceName": "Ферритин", "serviceHomecode": "408"},
    {"serviceName": "ТТГ (TSH) тиреотропный гормон", "serviceHomecode": "301"},
    {"serviceName": "Антитела к рецепторам ТТГ", "serviceHomecode": "317"},
    {"serviceName": "Анализ мочи по Нечипоренко", "serviceHomecode": "601"},
    {"serviceName": "Общий анализ мочи", "serviceHomecode": "602"},
    {"serviceName": "Гомоцистеин", "serviceHomecode": "455"},  # услуга есть, памятки нет
]

_BLOOD = "Кровь сдают утром строго натощак, после 8–12 часов ночного голодания. Можно пить воду."


def _memo(name: str, body: str) -> dict:
    return {"serviceName": name, "preparation": f"<p>Подготовка к исследованию</p><p>{body}</p>"}


# Срез МИС: у большинства памяток первая строка — подпись поля «Подготовка к исследованию»; «анализ» и «кровь» — почти в каждой.
_ROWS = [
    _memo("Ферритин", _BLOOD + " За сутки исключить жирную пищу."),
    _memo("ТТГ (TSH) тиреотропный гормон", _BLOOD + " Анализ крови на гормоны щитовидной железы сдают до приёма препаратов."),
    _memo("Антитела к рецепторам ТТГ", _BLOOD + " Анализ крови на антитела сдают до приёма препаратов щитовидной железы."),
    _memo("Анализ мочи по Нечипоренко", "Анализ мочи: накануне исключить мочегонные. Анализ крови сдают отдельно, натощак."),
    _memo("Общий анализ крови (Le, Er, Hb, СОЭ)", "Анализ крови сдают натощак. " + _BLOOD),
    _memo("Глюкоза", "Анализ крови на глюкозу: строго натощак, за сутки исключить сладкое. " + _BLOOD),
    _memo("Холестерол", "Анализ крови: натощак, за сутки исключить жирное. " + _BLOOD),
    _memo("Креатинин", "Анализ крови: натощак, за сутки исключить мясо. " + _BLOOD),
    _memo("Общий анализ мочи", "Утром собрать среднюю порцию мочи в стерильный контейнер, накануне исключить мочегонные."),
]


_KNOWN_TITLES = (*(row["serviceName"] for row in _ROWS), prepare_mod._BLOOD_GENERAL_TITLE)


def _options(prompt: str) -> list[tuple[int, str]]:
    """Варианты промпта выбора памятки: «N. название — начало текста» (в названии общей памятки тоже есть « — »; правила
    промпта тоже нумерованы — их названия не известны)."""

    out = []
    for num, rest in re.findall(r"^(\d+)\. (.*)$", prompt, flags=re.M):
        title = next((t for t in _KNOWN_TITLES if rest.startswith(f"{t} — ")), None)
        if title is not None:
            out.append((int(num), title))
    return out


def _synonyms(text: str) -> tuple[str, ...]:
    """Словарь синонимов клиники, как живой, чувствителен к словоформе слов вокруг: «анализ на сахар» знает, «анализу на
    сахар» — нет (обвязка словаря — другой модуль)."""

    return ("Глюкоза",) if str(text).strip().lower() in {"сахар", "на сахар", "анализ на сахар"} else ()


def _spans(reply: str, phrases: list[str]) -> dict:
    """Ответ «LLM разбора» по смыслу: группы номеров слов, из которых состоит каждая фраза."""

    words = [w.text.lower() for w in _turn_spans.words(reply)]
    groups = []
    for phrase in phrases:
        want = [w.text.lower() for w in _turn_spans.words(phrase)]
        start = next(i for i in range(len(words)) if words[i : i + len(want)] == want)
        groups.append([start + 1, start + len(want)])
    return {"phrases": groups}


@pytest.fixture
def world(monkeypatch):
    """Подготовка на синтетическом срезе; LLM разбора и выбора — подделки «по смыслу», которые задаёт тест."""

    state = {"phrases": {}, "want": set(), "select_calls": 0, "spans_down": False, "prompts": []}

    async def spans_llm(**kw):
        if state["spans_down"]:
            raise ConnectionError("llm down")
        reply = re.search(r"Сообщение пациента: «(.*)»", kw["user"]).group(1)
        return json.dumps(_spans(reply, state["phrases"][reply])), {}

    async def select_llm(prompt, **_kw):
        state["select_calls"] += 1
        state["prompts"].append(prompt)
        for num, title in _options(prompt):
            if title in state["want"]:
                return json.dumps({"match": num})
        return json.dumps({"match": None})

    async def same(first, second):
        return {first.title, second.title} <= state["want"]

    monkeypatch.setattr(cfg, "MR_PREPARE_BY_PARSE", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_TURN_SPANS_LLM", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_PREPARE_GENERAL_MEMO_LLM", "on", raising=False)
    monkeypatch.setattr(_turn_spans.llm_runtime, "generate_structured", spans_llm)
    monkeypatch.setattr(select_mod, "generate_text", select_llm)
    monkeypatch.setattr(select_mod, "_same_preparation", same)
    monkeypatch.setattr(turn_understanding, "_catalog_rows", lambda _region=None: _PRICE)
    monkeypatch.setattr(prepare_mod, "_catalog_rows", lambda _region=None: _PRICE)
    monkeypatch.setattr(_strong_match, "mis_synonym_candidates", _synonyms)
    monkeypatch.setattr(prepare_mod, "mis_synonym_candidates", _synonyms)
    monkeypatch.setattr(prepare_mod.api_service_info, "load_service_info", lambda: list(_ROWS))
    monkeypatch.setattr(prepare_mod, "kb_patient_memos", lambda: ())
    return state


def _turn(reply: str, entities: dict | None = None, *, query: str | None = None) -> dict:
    async def go() -> dict:
        turn_understanding.begin_turn()
        await turn_understanding.observe_prepare_turn("PREPARE", user_text=reply)
        svc = Services()
        svc.ensure_background_refresh_started = lambda: None
        return await svc.test_prepare(reply if query is None else query, dict(entities or {}))

    return asyncio.run(go())


_FERRITIN = [
    ("Подготовка к анализу на ферритин", ["анализу на ферритин"]),
    ("Как подготовиться к сдаче анализа на ферритин?", ["анализа на ферритин"]),
    ("подготовка к ферритину", ["ферритину"]),
    ("Как сдавать анализы на ферритин?", ["анализы на ферритин"]),
]
_TSH = [
    ("подготовка к ттг", ["ттг"]),
    ("Подготовка к анализу на ТТГ", ["анализу на ТТГ"]),
    ("как сдавать анализ ттг?", ["анализ ттг"]),
]
_BLOOD_GENERAL = [
    ("Подготовка к анализу крови", []),
    ("Подготовка к анализам крови", []),
    ("Как подготовиться к сдаче анализов крови?", []),
]


def _answers(world, group, want: set[str], entities: dict | None = None) -> list[str]:
    world["phrases"] = dict(group)
    world["want"] = want
    return [_turn(question, entities)["prepare"] for question, _ in group]


def test_ferritin_word_forms_one_memo(world) -> None:
    answers = _answers(world, _FERRITIN, {"Ферритин"})
    assert len(set(answers)) == 1, answers
    assert answers[0].startswith("Памятка клиники «Ферритин»:") and "Нечипоренко" not in answers[0]


def test_tsh_word_forms_both_memos_named_by_the_same_word(world) -> None:
    """«анализ на ттг» и «ттг» — одна пара памяток: двойник названия — по слову, которым названа выбранная, а не по фразе
    с общим словом «анализ» (без списка общих слов)."""

    answers = _answers(world, _TSH, {"ТТГ (TSH) тиреотропный гормон", "Антитела к рецепторам ТТГ"})
    assert len(set(answers)) == 1, answers
    assert "«ТТГ (TSH) тиреотропный гормон»:" in answers[0] and "«Антитела к рецепторам ТТГ»:" in answers[0]


_SUGAR = [
    ("подготовка к анализу на сахар", ["анализу на сахар"]),
    ("Как подготовиться к анализу на сахар?", ["анализу на сахар"]),
    ("Подготовка к анализу на сахар крови", ["анализу на сахар"]),
    ("как сдавать анализ на сахар", ["анализ на сахар"]),
]


def test_clinic_synonym_found_whatever_the_form_of_words_before_it(world) -> None:
    """Синоним клиники «сахар» → «Глюкоза» — карточка «названа» при любой словоформе слов перед синонимом: данные ищутся по
    хвостам фразы («анализу на сахар» → «на сахар»), а не по фразе без слов списка общих слов."""

    answers = _answers(world, _SUGAR, {"Глюкоза"})
    assert len(set(answers)) == 1 and answers[0].startswith("Памятка клиники «Глюкоза»:"), answers
    world["select_calls"] = 0
    strict = [_turn(question, {"__runtime_llm_mode": "strict"})["prepare"] for question, _ in _SUGAR]
    assert strict == answers and world["select_calls"] == 0  # без LLM — та же карточка: её назвали данные


_OAM = [
    ("Подготовка к общему анализу мочи", ["общему анализу мочи"]),
    ("Как сдавать общий анализ мочи?", ["общий анализ мочи"]),
    ("Как подготовиться к сдаче общего анализа мочи?", ["общего анализа мочи"]),
]


def test_word_form_unknown_to_data_does_not_block_the_memo(world) -> None:
    """«общему» стеммер сводит к «общем» — такой основы нет ни в памятках, ни в прайсе: это словоформа, а не услуга, и она не
    делает памятку «Общий анализ мочи» негодной (карточка 24.10: «Подготовка к общему анализу мочи» → оффер вместо памятки)."""

    answers = _answers(world, _OAM, {"Общий анализ мочи"})
    assert len(set(answers)) == 1 and answers[0].startswith("Памятка клиники «Общий анализ мочи»:"), answers


def test_general_word_of_the_phrase_does_not_bring_foreign_memos(world) -> None:
    """«анализу крови на гомоцистеин»: памятки, где есть только «анализ» и «кровь», предмет не называют — в кандидаты LLM не
    идут (нужны слова фразы, несущие половину её веса); памятки о гомоцистеине нет — оффер, а не чужая памятка."""

    world["phrases"] = {"Подготовка к анализу крови на гомоцистеин": ["анализу крови на гомоцистеин"]}
    world["want"] = {"Анализ мочи по Нечипоренко", "Общий анализ крови (Le, Er, Hb, СОЭ)", "Глюкоза", "Холестерол", "Креатинин"}
    out = _turn("Подготовка к анализу крови на гомоцистеин")["prepare"]
    assert out.endswith("Перевести на оператора?"), out
    assert all([title for _num, title in _options(prompt)] == [prepare_mod._BLOOD_GENERAL_TITLE] for prompt in world["prompts"])


def test_no_service_in_parse_offers_only_general_memo(world) -> None:
    answers = _answers(world, _BLOOD_GENERAL, {prepare_mod._BLOOD_GENERAL_TITLE})
    assert answers == [prepare_mod._BLOOD_COLLECTION_GUIDANCE] * len(_BLOOD_GENERAL)
    # без поиска по словам: ни одной карточки МИС среди вариантов — только общая памятка
    assert all([title for _num, title in _options(prompt)] == [prepare_mod._BLOOD_GENERAL_TITLE] for prompt in world["prompts"])


def test_no_service_and_no_general_option_asks_for_the_name(world, monkeypatch) -> None:
    monkeypatch.setattr(cfg, "MR_PREPARE_GENERAL_MEMO_LLM", "off", raising=False)
    answers = _answers(world, _BLOOD_GENERAL, set())
    assert len(set(answers)) == 1 and "Уточните полное название анализа или процедуры" in answers[0]
    assert world["select_calls"] == 0  # вариантов нет — выбирать нечего


@pytest.mark.parametrize("group", [_FERRITIN, _TSH, _BLOOD_GENERAL])
def test_parse_failure_asks_to_name_the_subject_without_letters(world, group) -> None:
    world["spans_down"] = True
    answers = _answers(world, group, {"Ферритин", prepare_mod._BLOOD_GENERAL_TITLE})
    assert answers == [prepare_mod._PARSE_FAILED_TEXT] * len(group)
    assert world["select_calls"] == 0


def test_without_llm_choice_only_card_named_by_data(world) -> None:
    """Без LLM выбора (strict) — только карточка, которую фраза называет по данным МИС; карточка, где слова фразы лишь
    упомянуты («анализ», «кровь» у Нечипоренко), не выбирается никогда."""

    world["phrases"] = dict(_FERRITIN)
    for question, _ in _FERRITIN:
        out = _turn(question, {"__runtime_llm_mode": "strict"})["prepare"]
        assert "Нечипоренко" not in out and "Креатинин" not in out
        assert out.startswith("Памятка клиники «Ферритин»:") or out.endswith("Перевести на оператора?")
    assert world["select_calls"] == 0


def test_input_not_the_reply_uses_the_service_of_the_data(world) -> None:
    """«Подготовка» в ответе о цене зовёт подготовку по названию услуги (`service_bundle_info`): разбор хода — о другой
    реплике, предмет — название из данных."""

    world["phrases"] = {"Сколько стоит ферритин и как подготовиться?": ["ферритин"]}
    world["want"] = {"Ферритин"}
    out = _turn("Сколько стоит ферритин и как подготовиться?", {"service_name": "Ферритин"}, query="Ферритин")
    assert out["prepare"].startswith("Памятка клиники «Ферритин»:")


def test_question_without_service_after_one_shown_service_is_about_it(world) -> None:
    world["phrases"] = {"а как подготовиться?": []}
    world["want"] = {"Ферритин"}

    async def go() -> dict:
        state = SessionState(session_id="t")
        state.last_entities["_shown_services"] = ["Ферритин"]
        shown_context.begin_turn(state)
        turn_understanding.begin_turn()
        await turn_understanding.observe_prepare_turn("PREPARE", user_text="а как подготовиться?")
        svc = Services()
        svc.ensure_background_refresh_started = lambda: None
        return await svc.test_prepare("а как подготовиться?", {})

    assert asyncio.run(go())["prepare"].startswith("Памятка клиники «Ферритин»:")


def test_memo_caption_by_data_drops_the_same_lines_as_the_heading_list() -> None:
    """Подпись поля «Подготовка» задаёт срез (большинство памяток начинается одной строкой), а не список заголовков: строки
    ответа те же."""

    corpus = P.memo_corpus(list(_ROWS))
    assert corpus.caption == "подготовка к исследованию"
    for row in _ROWS:
        memo = select_mod.PrepareMemo(select_mod.SOURCE_MIS, row["serviceName"], P._memo_plain_text(row["preparation"]))
        assert prepare_mod._memo_lines_without_caption(memo, corpus.caption) == prepare_mod._memo_lines(memo)


@pytest.mark.parametrize(
    "title, other",
    [("ПАМЯТКА Гастроскопия (без наркоза)", "ПАМЯТКА Гастроскопия с наркозом"), ("ПАМЯТКА ФКС БЕЗ НАРКОЗА ", "ПАМЯТКА ФКС с наркозом ")],
)
def test_anesthesia_variants_by_title_words(title: str, other: str) -> None:
    memos = [select_mod.PrepareMemo(select_mod.SOURCE_KB, t, "текст") for t in (title, other, "ПАМЯТКА  ФКС + ФГДС с наркозом")]
    by_words = prepare_mod._anesthesia_variants_by_words(memos[0], memos)
    assert {m.title for m in by_words.values()} == {title, other}
    assert by_words == prepare_mod._anesthesia_variants(memos[0], memos)


def test_off_does_not_parse_prepare_turns(world, monkeypatch) -> None:
    monkeypatch.setattr(cfg, "MR_PREPARE_BY_PARSE", "off", raising=False)
    world["phrases"] = {"Подготовка к анализу на ферритин": ["анализу на ферритин"]}

    async def go():
        turn_understanding.begin_turn()
        await turn_understanding.observe_prepare_turn("PREPARE", user_text="Подготовка к анализу на ферритин")
        return turn_understanding.parse_status("Подготовка к анализу на ферритин")

    assert asyncio.run(go()) is None


@pytest.mark.parametrize("mode", ["on", "off"])
def test_planner_slot_gate_stays_with_nlu(world, monkeypatch, mode: str) -> None:
    """Гейт слота планировщика («к какому анализу нужна подготовка?», если NLU не извлёк услугу) под `on` прежний: карточка
    24.10 — обход гейта по фразам разбора сработал на одном ходе из 1068, и там хуже (g0673 «в какой день недели приносить
    анализы кала и мочи?» — памятка вместо вопроса). Сущность NLU — тоже решение LLM, не список."""

    world["phrases"] = {"В какой день недели можно приносить анализы кала и мочи?": ["анализы кала", "мочи"]}

    async def go():
        turn_understanding.begin_turn()
        monkeypatch.setattr(cfg, "MR_PREPARE_BY_PARSE", "on", raising=False)
        await turn_understanding.observe_prepare_turn("PREPARE", user_text="В какой день недели можно приносить анализы кала и мочи?")
        monkeypatch.setattr(cfg, "MR_PREPARE_BY_PARSE", mode, raising=False)
        decision = RouteDecision(label="PREPARE", confidence=0.9, entities={}, flags=set())
        return build_plan(decision, SessionState(session_id="t"), "В какой день недели можно приносить анализы кала и мочи?", MemoryStore())

    assert not any(step.tool == "test_prepare" for step in asyncio.run(go()).steps)


# --------------------------------------------------------------------------- путь `on` не читает пяти списков (AST)

_MODULES = {
    "prepare": Path(prepare_mod.__file__),
    "_prepare": Path(P.__file__),
    "_prepare_select_llm": Path(select_mod.__file__),
}


def _module_defs() -> dict[str, tuple[dict[str, ast.AST], dict[str, str]]]:
    """Модуль → (его функции и классы верхнего уровня, имя → модуль, из которого оно импортировано среди трёх)."""

    out = {}
    for name, path in _MODULES.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in _MODULES:
                imports.update({alias.asname or alias.name: node.module for alias in node.names})
        out[name] = (defs, imports)
    return out


def _reached_names(module: str, entry: str) -> set[str]:
    """Все имена, которые читают функции, достижимые из `entry` по вызовам внутри трёх модулей подготовки."""

    defs = _module_defs()
    seen: set[tuple[str, str]] = set()
    names: set[str] = set()
    todo = [(module, entry)]
    while todo:
        mod, fn = todo.pop()
        if (mod, fn) in seen:
            continue
        seen.add((mod, fn))
        local, imports = defs[mod]
        for node in ast.walk(local[fn]):
            ident = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if ident is None:
                continue
            names.add(ident)
            if ident in local:
                todo.append((mod, ident))
            elif ident in imports and ident in defs[imports[ident]][0]:
                todo.append((imports[ident], ident))
    return names


def test_on_path_reads_none_of_the_five_lists() -> None:
    reached = _reached_names("prepare", "_test_prepare_by_parse")
    assert "_phrase_candidates" in reached and "select_prepare_memos_by_parse" in reached  # граф обойдён
    assert not reached & set(_FIVE), sorted(reached & set(_FIVE))


def test_off_path_still_reads_them_walker_sees_lists() -> None:
    """Проверка обходчика: прежний путь (без ветки `on`) эти списки читает — обходчик их видит."""

    reached = _reached_names("prepare", "_prepare_candidates_from_analysis_api_cache") | _reached_names("prepare", "_anesthesia_variants")
    assert {"_PREPARE_QUERY_STOPWORDS", "_PREPARE_SERVICE_INFO_GENERIC_TOKENS", "_PREPARE_GENERIC_HEADINGS", "_VARIANT_NOISE_RE"} <= reached
    assert "_GENERIC_PREPARE_QUERY_RE" in _reached_names("prepare", "_is_generic_prepare_query")
