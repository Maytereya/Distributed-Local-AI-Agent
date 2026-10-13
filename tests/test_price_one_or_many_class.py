"""Класс «одна услуга или несколько» (L-15b, шаг 2) — через обработчик эндпоинта, мини-прод.

Ценовая реплика — это одна услуга или список, и решают это сейчас правила: разделители («,», «и»,
«+»), сегментация сплошного списка по каталогу и защиты понимания хода (`price_list_services`).
Названия прайса сами содержат запятые, «и» и «+» («Прием (осмотр, консультация) врача-терапевта»,
«Билирубин и его фракции (общий, прямой и непрямой)», «УЗИ органов мошонки + ЦДК сосудов»), а фраза
одной услуги без разделителей режется на «позиции» («мазок на» / «микрофлору из» / «зева»). Контракт
C (L-15b, шаг 4) отдаёт этот выбор LLM: фразы-подстроки из перечня кода. Этот тест описывает класс
до него и держит его после.

Инварианты класса (принципы 1–3; решения владельца 13.10):
- (а) одно название из прайса — одна услуга: реплика не режется на позиции корзины, а цену в ответе
  имеют только строки этого названия (сильное по данным — дословное название); фраза одной услуги не
  из прайса тоже не режется, а при живой LLM в ответе нет чужой цены;
- (б) список — каждая позиция своя строка: строки одного раздела корзины — одной позиции, чужой цены
  нет, при живой LLM у каждой позиции с услугой в прайсе есть цена;
- (в) при недоступной LLM в списке — только сильные по данным позиции с ценой (точное название,
  синоним МИС, код — решение владельца 13.10, политика сбоя (ii)), остальные — без цены;
- вопрос о сроке одной услуги — не цена и не список: срок кодом из МИС или честный перевод.

Мир: мини-прод, день, сеть закрыта, синонимы МИС пусты (`_hermetic_mis_synonyms`). Две LLM: гейтовая
«недоступна» (`_LLMDown`) и поддельная LLM-оракул — идеальный выбор модели: на выборе строки прайса,
корзины и срока она отмечает ровно строки разметки случая, остальные места (классификатор,
нормализатор, рендер) — «недоступна». Два набора выключателей: по умолчанию и набор деплоя
(понимание хода, корзина, срок, симптом — on; цена через модуль — off; L-13 — тень). Где текущий код
инвариант не держит — `xfail(strict=True)` с причиной: позеленевший случай краснеет, и его отметку
снимают в том же коммите.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT
from tests.test_price_turns_golden import FIXTURE, run_dialog

_DEPLOY = {
    "MR_TURN_UNDERSTANDING_PRICE": "on",
    "MR_CHOOSE_BASKET": "on",
    "MR_CHOOSE_READINESS": "on",
    "MR_CHOOSE_SYMPTOM": "on",
    "MR_CHOOSE_PRICE": "off",
    "MR_PRICE_MATCH_POLICY": "shadow",
}
CONFIGS = ("default", "deploy")
LLMS = ("down", "oracle")

_SELECT_MARKS = {
    "basket": "ответить на вопрос о ценах нескольких услуг",
    "price": "ответить пациенту на вопрос о цене",
    "readiness": "сколько готовится анализ или исследование",
}
_BASKET_SECTION = "Формат: По услуге «"
_UNRECOGNIZED = "Не распознал"
_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)

_OAK = ("Общий анализ крови (Le, Er, Hb, СОЭ)", "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)", "ОАК (без лейкоцитарной формулы и СОЭ)")
_VITAMIN_D = ("Витамин D суммарный (25-OH витамин D2 и 25-OH витамин D3, общий)", "Витамин D, 25-гидрокси (кальциферол), ВЭЖХ-МС/МС - метод масс-спектрометрия")
_CULTURE = ("Посев на микрофлору (идентификация агента и чувствительность к а\\б)",)


@dataclass(frozen=True)
class Case:
    """Реплика и разметка: позиция → строки прайса, которые и есть эта услуга.

    `kind`: `one` — одна услуга (позиция одна, `exact` — реплика называет её дословным названием
    прайса); `list` — список (`strong` — позиции, сильные по данным); `readiness` — вопрос о сроке.
    """

    text: str
    kind: str
    positions: dict[str, tuple[str, ...]]
    strong: frozenset[str] = frozenset()
    exact: bool = False
    deadline: str = ""
    xfail: dict[tuple[str, str], str] = field(default_factory=dict)


def _one(name: str, **kw) -> Case:
    return Case(f"сколько стоит {name}", "one", {name: (name,)}, exact=True, **kw)


_FAMILY_DOWN = (
    "сбой LLM — ответ правил на дословное название: семейный список с ценами соседних услуг; строки "
    "названия по данным — L-13 (тень в наборе деплоя) или контракт C"
)
_FAMILY_SHORTCUT = (
    "дословное название среди кандидатов — выбор LLM пропущен («правила не ошибаются»), а ответ правил — "
    "семья ОАК с ценой второй услуги"
)
_DOWN = {(c, "down"): _FAMILY_DOWN for c in CONFIGS}

CASES: dict[str, Case] = {
    # --- названия прайса с «,», «и», «+», скобками с запятыми — одна услуга (g0315)
    "name_parens_commas": _one("Прием (осмотр, консультация) врача-терапевта"),
    "name_commas_in_parens": _one(
        "Общий анализ крови (Le, Er, Hb, СОЭ)",
        xfail={**_DOWN, **{(c, "oracle"): _FAMILY_SHORTCUT for c in CONFIGS}},
    ),
    "name_plus_in_parens": _one("Глюкозотолерантный тест (глюкоза+ C-пептид)", xfail=_DOWN),
    "name_plus_outside": _one("УЗИ органов мошонки + ЦДК сосудов"),
    "name_and_outside_parens": _one("Билирубин и его фракции (общий, прямой и непрямой)"),
    "name_and_in_parens": _one("Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)"),
    "name_and_between_numerals": _one("Пепсиногены  I и II с расчетом соотношения", xfail=_DOWN),
    "name_many_commas": _one("Исследование кала на паразиты (лямблии, описторхи, аскариды, острицы, власоглав, карликовый цепень и др.)"),
    # --- сплошная фраза одной услуги (g0542)
    "phrase_smear_throat": Case("сколько стоит мазок на микрофлору с поверхности задней стенки глотки", "one", {"мазок": _CULTURE}),
    "phrase_smear_pharynx": Case(
        "сколько стоит мазок на микрофлору из зева",
        "one",
        {"мазок": _CULTURE},
        xfail={
            ("deploy", "down"): (
                "фраза одной услуги разрезана сегментацией каталога на «мазок на» / «микрофлору из» / «зева»; сбой "
                "LLM — корзина правил с чужими строками («Мазок на степень чистоты влагалища»); разрез — контракт C"
            ),
        },
    ),
    "phrase_oms": Case("сколько стоит приём терапевта по ОМС", "one", {"приём терапевта по омс": ()}),
    # --- вопрос о сроке с лишними словами
    "readiness_extra_words": Case(
        "здравствуйте, подскажите, сколько дней делается анализ на ферритин?",
        "readiness",
        {"ферритин": ("Ферритин",)},
        deadline="Срок готовности «Ферритин» — 1–2 дня.",
        xfail={("default", m): "понимание хода в тени: вопрос о сроке получает цену (READINESS-AS-PRICE)" for m in LLMS},
    ),
    # --- списки
    "list_true_three": Case(
        "сколько стоит ферритин, ттг и витамин д",
        "list",
        {"ферритин": ("Ферритин",), "ттг": ("ТТГ (TSH) тиреотропный гормон",), "витамин д": _VITAMIN_D},
        strong=frozenset({"ферритин"}),
        xfail={(c, "down"): "сбой LLM — слабые позиции («ттг», «витамин д») с ценой строк правил; политика (ii) — шаг 5" for c in CONFIGS},
    ),
    "list_one_unrecognized": Case(
        "сколько стоит ферритин, глюкоза и спирометрия",
        "list",
        {"ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",), "спирометрия": ()},
        strong=frozenset({"ферритин", "глюкоза"}),
        xfail={("default", m): "корзина правил (choose_basket off): вторая строка позиции — «Глюкоза в моче» к глюкозе (limit=2)" for m in LLMS},
    ),
    "list_caps_run_on_priced": Case(
        "сколько стоит ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ",
        "list",
        {"ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",), "холестерол": ("Холестерол",)},
        strong=frozenset({"ферритин", "глюкоза", "холестерол"}),
        xfail={
            **{(c, "down"): "сплошной список из трёх дословных названий не распознан как список — «не нашёл»" for c in CONFIGS},
            **{
                (c, "oracle"): "сплошной список не распознан как список: три услуги — одним ответом «одной услуги» (выбор строки прайса отметил все три)"
                for c in CONFIGS
            },
        },
    ),
    "list_caps_run_on_bare": Case(
        "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",
        "list",
        {"оак": _OAK, "оам": ("Общий анализ мочи",), "ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",)},
        strong=frozenset({"ферритин", "глюкоза"}),
        xfail={
            **{("default", m): "понимание хода в тени: список сведён к одной услуге (ОАК)" for m in LLMS},
            ("deploy", "down"): "сбой LLM — корзина правил: к «ОАМ» строка ОАК (чужая цена), слабые «ОАК»/«ОАМ» с ценой; политика (ii) — шаг 5",
        },
    ),
}


class _OracleLLM:
    """LLM-оракул: на выборе строки прайса, корзины и срока отмечает ровно строки разметки случая.

    Корзина — по позиции: строка выбирается, только если «позиция» — услуга из разметки и строка — её;
    обрывок фразы («мазок на») — не услуга, для него ничего. Цена и срок — любая строка разметки.
    Прочие места — «недоступна», как в мире гейта.
    """

    def __init__(self, case: Case) -> None:
        self.by_fragment = {fragment.lower(): set(names) for fragment, names in case.positions.items()}
        self.names = {name for names in case.positions.values() for name in names}
        self.calls: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        task = next((task for task, mark in _SELECT_MARKS.items() if mark in seen), None)
        if task is None:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.calls.append(task)
        picks = []
        for line in prompt.splitlines():
            head, dot, rest = line.partition(". ")
            if not (dot and head.isdigit()):
                continue
            wanted = self.names
            if task == "basket":
                fragment, _, rest = rest.partition(" → ")
                wanted = self.by_fragment.get(fragment.strip("«»").lower(), set())
            if _norm(rest.rsplit(" — ", 1)[0]) in {_norm(name) for name in wanted}:
                picks.append(int(head))
        answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _norm(name: str) -> str:
    return " ".join(str(name).split())


def _priced_sections(text: str) -> list[list[str]]:
    """Строки с ценой по разделам ответа (раздел корзины — позиция; ответ одной услуги — один раздел)."""

    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        if line.startswith(_BASKET_SECTION):
            sections.append([])
        match = _PRICED.match(line)
        if match:
            sections[-1].append(_norm(match.group("name")))
    return [section for section in sections if section]


def _violations(case: Case, text: str, llm: str) -> list[str]:
    sections = _priced_sections(text)
    priced = [name for section in sections for name in section]
    truth = {fragment: {_norm(n) for n in names} for fragment, names in case.positions.items()}
    every = set().union(*truth.values())
    out: list[str] = []
    if case.kind in ("one", "readiness"):
        if _BASKET_SECTION in text or _UNRECOGNIZED in text:
            out.append("(а) одна услуга разрезана на позиции корзины")
        if case.exact and not priced:
            out.append("(а) дословное название без цены")
        if case.exact and set(priced) - every:
            out.append(f"(а) цена чужих строк при дословном названии: {sorted(set(priced) - every)}")
        if case.kind == "one" and llm == "oracle" and set(priced) - every:
            out.append(f"(а) чужая цена при живой LLM: {sorted(set(priced) - every)}")
    if case.kind == "readiness":
        if priced or re.search(r"\d\s*руб", text):
            out.append("вопрос о сроке получил цену")
        allowed = {case.deadline} if llm == "oracle" else {case.deadline, RESULT_TIMING_OPERATOR_TEXT}
        if text not in allowed:
            out.append(f"вопрос о сроке: не срок из МИС и не честный перевод: {text[:120]!r}")
    if case.kind == "list":
        if set(priced) - every:
            out.append(f"(б) чужая цена: {sorted(set(priced) - every)}")
        for section in sections:
            if not any(set(section) <= names for names in truth.values() if names):
                out.append(f"(б) в разделе строки разных позиций: {section}")
        if llm == "oracle":
            missing = [fragment for fragment, names in truth.items() if names and not names & set(priced)]
            if missing:
                out.append(f"(б) позиции без своей цены при живой LLM: {missing}")
        else:
            strong = set().union(*(truth[fragment] for fragment in case.strong)) if case.strong else set()
            if set(priced) - strong:
                out.append(f"(в) сбой LLM — цена не сильной позиции: {sorted(set(priced) - strong)}")
            missing = [fragment for fragment in sorted(case.strong) if not truth[fragment] & set(priced)]
            if missing:
                out.append(f"(в) сильная позиция без цены: {missing}")
    return out


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


def _params():
    for name, case in sorted(CASES.items()):
        for config in CONFIGS:
            for llm in LLMS:
                reason = case.xfail.get((config, llm))
                marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
                yield pytest.param(name, config, llm, marks=marks, id=f"{name}-{config}-{llm}")


@pytest.mark.parametrize(("name", "config", "llm"), list(_params()))
def test_one_service_or_several(name, config, llm, mini_prod_world, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    case = CASES[name]
    if config == "deploy":
        for key, value in _DEPLOY.items():
            monkeypatch.setattr(cfg, key, value, raising=False)
    if llm == "oracle":
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _OracleLLM(case))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog([case.text], f"one-or-many-{name}-{config}-{llm}")
        attempts = list(world.net.attempts)
    assert attempts == []
    violations = _violations(case, turn["text"], llm)
    assert not violations, "\n".join([*violations, "--- ответ:", turn["text"]])


def test_oracle_answers_the_selection_it_was_asked(mini_prod_world, monkeypatch):
    """Оракул действительно выбирает: на живом наборе деплоя слабые позиции списка получают ровно свои
    строки, а строки другой услуги рядом («Антитела к рецепторам ТТГ», акции витамина D) — нет."""

    from messengers_router.runtime_config import config as cfg

    for key, value in _DEPLOY.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    case = CASES["list_true_three"]
    oracle = _OracleLLM(case)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        [turn] = run_dialog([case.text], "one-or-many-oracle")
    assert oracle.calls == ["basket"]
    assert "Антитела к рецепторам ТТГ" not in turn["text"] and "регулярно" not in turn["text"]
    assert "ТТГ (TSH) тиреотропный гормон — 400" in turn["text"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1. Ферритин — 530 руб..\n2) Розничная цена: Глюкоза — 1 190 руб.. Формат: поликлиника.", [["Ферритин", "Глюкоза"]]),
        ("По запросу:\n\nФормат: По услуге «ОАК»:\n1. А — 1 руб..\n\nФормат: По услуге «ОАМ»:\n1. Б — 2 руб..", [["А"], ["Б"]]),
        ("Срок готовности «Ферритин» — 1–2 дня.", []),
    ],
)
def test_answer_parser_reads_priced_rows_by_section(text, expected):
    assert _priced_sections(text) == expected


def test_violations_catch_a_cut_name_and_a_foreign_row():
    case = CASES["name_parens_commas"]
    cut = "Формат: По услуге «Прием»:\n1. Прием (осмотр, консультация) врача-терапевта — 2 700 руб..\nНе распознал: консультация"
    assert _violations(case, cut, "down")
    oak = CASES["list_caps_run_on_bare"]
    foreign = "Формат: По услуге «ОАМ»:\n1. Общий анализ мочи — 280 руб..\n2. Общий анализ крови (Le, Er, Hb, СОЭ) — 390 руб.."
    assert any(v.startswith("(б) в разделе") for v in _violations(oak, foreign, "oracle"))
