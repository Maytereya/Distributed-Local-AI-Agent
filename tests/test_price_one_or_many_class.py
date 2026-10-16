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
- (в) каждая цена в ответе подписана полным названием строки прайса; одиночная цена без списка —
  только при сильном совпадении по данным (точное название, синоним МИС, код — решение владельца 13.10)
  или при выборе LLM. У сильной позиции — только её строки. Слабая позиция при недоступной LLM —
  несколько вариантов кода (до 5) с полными названиями и ценами или без цены, но не одна строка как «та
  самая» (поправка владельца 13.10 к политике сбоя (ii): слабые не скрываются; форму решит замер шага
  6); при живой LLM все выбранные ею строки — с ценой;
- вопрос о сроке одной услуги — не цена и не список: срок кодом из МИС или честный перевод.

Мир: мини-прод, день, сеть закрыта, синонимы МИС пусты (`_hermetic_mis_synonyms`). Две LLM: гейтовая
«недоступна» (`_LLMDown`) и поддельная LLM-оракул — идеальный выбор модели: на выборе строки прайса,
корзины и срока она отмечает ровно строки разметки случая, на выборе фраз реплики (L-15b) — ровно фразы
разреза случая (`TRUE_CUT`), остальные места (классификатор, нормализатор, рендер) — «недоступна». Два
набора выключателей: умолчания кода до переворота «умолчания = прод» (заданы явно); набор деплоя (понимание хода,
корзина, срок, симптом — on; цена через модуль — off). Разбор реплики на фразы (`turn_parse`, L-15b) — единственный
режим (T7 шаг 3б, 16.10): понимание, корзина, срок и путь одной услуги читают разбор в обоих наборах. Где код
инвариант не держит — `xfail(strict=True)` с причиной: позеленевший случай краснеет, и его отметку снимают в том же
коммите.
"""

from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass, field

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT
from tests.test_price_turns_golden import FIXTURE, run_dialog

# Наборы деплоя 14.10 и `default` — исторические: «контекст = показанное» на них выключен явно (умолчание кода = прод с
# 16.10, деплой 5: `context_shown = on`); прод-наборы и `*_shown` включают его сами. Фраза журнала на ответы не влияет —
# тесты журнала задают её сами. В `default` так же явно — выбор строки прежним вызовом и гарды П2 правил (умолчания кода
# = прод с 17.10, STEP-2026-10-17-DEFAULTS-D: `choose_price = on`, `parse_price_guard = llm`, `p2_llm_judge = on`); набор
# деплоя гарды П2 не задаёт — тесты класса A и судьи П2 ставят их сами до `_deploy`.
_DEPLOY = {
    "MR_TURN_UNDERSTANDING_PRICE": "on",
    "MR_CHOOSE_READINESS": "on",
    "MR_CHOOSE_SYMPTOM": "on",
    "MR_CHOOSE_PRICE": "off",
    "MR_CONTEXT_SHOWN": "off",
}
# Набор `default` — умолчания кода ДО переворота «умолчания = прод» (STEP-2026-10-15-DEFAULTS-A), заданные явно:
# переворот умолчаний в `agent_logic_2/config.py` не меняет смысла отметок этого набора.
_DEFAULT = {
    "MR_TURN_UNDERSTANDING_PRICE": "shadow",
    "MR_CHOOSE_READINESS": "off",
    "MR_CHOOSE_SYMPTOM": "off",
    "MR_PARSE_FAIL_VARIANTS": 5,
    "MR_CONTEXT_SHOWN": "off",
    "MR_CHOOSE_PRICE": "off",
    "MR_PARSE_PRICE_GUARD": "rules",
    "MR_P2_LLM_JUDGE": "off",
}
# Набор выключателей → значения поверх конфига гейта. Разбор реплики (`turn_parse = on`) — единственный режим
# (T7 шаг 3б, 16.10): `deploy` — прод; `deploy_parse` — прежнее имя того же набора для файлов, которые его зовут.
_SETS = {"default": _DEFAULT, "deploy": _DEPLOY, "deploy_parse": _DEPLOY}
CONFIGS = ("default", "deploy")
LLMS = ("down", "oracle")

_SELECT_MARKS = {
    "basket": "ответить на вопрос о ценах нескольких услуг",
    "price": "ответить пациенту на вопрос о цене",
    "readiness": "сколько готовится анализ или исследование",
}
_PHRASES_MARK = "понять, какие услуги назвал пациент"
def _is_basket_section(line: str) -> bool:
    """Заголовок раздела корзины — «По услуге «X»:» (отдельной строкой). Ответ одной услуги начинается
    «По услуге «X» нашёл следующее:» — это не раздел корзины."""

    return line.startswith("По услуге «") and line.endswith("»:")


_UNRECOGNIZED = "Не распознал"
_VARIANTS_MAX = 5  # слабая позиция без LLM — несколько вариантов кода (до 3–5), форму решит замер шага 6
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


# T7 шаг 3б (16.10): под `turn_parse = off` при недоступной LLM дословное название с «и», «+», запятыми давало семью
# соседних строк или вариант B без цены; под `on` (единственный режим) его решает разбор по данным — отметки сняты.

CASES: dict[str, Case] = {
    # --- названия прайса с «,», «и», «+», скобками с запятыми — одна услуга (g0315)
    "name_parens_commas": _one("Прием (осмотр, консультация) врача-терапевта"),
    # Ветка «дословное название — правила не ошибаются» снята (T7 шаг 2): строка по данным, без семьи ОАК.
    "name_commas_in_parens": _one("Общий анализ крови (Le, Er, Hb, СОЭ)"),
    "name_plus_in_parens": _one("Глюкозотолерантный тест (глюкоза+ C-пептид)"),
    "name_plus_outside": _one("УЗИ органов мошонки + ЦДК сосудов"),
    "name_and_outside_parens": _one("Билирубин и его фракции (общий, прямой и непрямой)"),
    "name_and_in_parens": _one("Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)"),
    "name_and_between_numerals": _one("Пепсиногены  I и II с расчетом соотношения"),
    "name_many_commas": _one("Исследование кала на паразиты (лямблии, описторхи, аскариды, острицы, власоглав, карликовый цепень и др.)"),
    # --- сплошная фраза одной услуги (g0542)
    "phrase_smear_throat": Case("сколько стоит мазок на микрофлору с поверхности задней стенки глотки", "one", {"мазок": _CULTURE}),
    "phrase_smear_pharynx": Case(
        "сколько стоит мазок на микрофлору из зева",
        "one",
        {"мазок": _CULTURE},
        # При сбое LLM — реплика целиком (Q2 отменено 16.10, T7 шаг 3а): одна услуга, без нарезки «мазок на / …».
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
        # «корзина правил (choose_basket off): одна строка, выбранная буквами» — выключателя нет с 16.10, LLM выбирает всегда.
    ),
    "list_one_unrecognized": Case(
        "сколько стоит ферритин, глюкоза и спирометрия",
        "list",
        {"ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",), "спирометрия": ()},
        strong=frozenset({"ферритин", "глюкоза"}),
    ),
    "list_caps_run_on_priced": Case(
        "сколько стоит ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ",
        "list",
        {"ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",), "холестерол": ("Холестерол",)},
        strong=frozenset({"ферритин", "глюкоза", "холестерол"}),
        # При сбое LLM — разрез сильными окнами по данным МИС (Q2 отменено 16.10, T7 шаг 3а): три цены.
    ),
    "list_caps_run_on_bare": Case(
        "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",
        "list",
        {"оак": _OAK, "оам": ("Общий анализ мочи",), "ферритин": ("Ферритин",), "глюкоза": ("Глюкоза",)},
        strong=frozenset({"ферритин", "глюкоза"}),
        xfail={
            **{("default", m): "понимание хода в тени: список сведён к одной услуге (ОАК), ферритин и глюкоза без цены" for m in LLMS},
            ("deploy", "down"): (
                "LLM недоступна, сплошной текст: в промежутке между сильными окнами — слова вне прайса («ОАК ОАМ» без "
                "синонимов МИС в мини-проде), без LLM неотличимые от условия П2 («по омс») — реплика целиком, ферритин и "
                "глюкоза без цены (T7 шаг 3, П2 строже покрытия); с синонимами МИС (прод) — разрез данными"
            ),
        },
    ),
}


# Разрез случая — словами реплики (сравнение без регистра и пунктуации): фразы, которые оракул отмечает на
# выборе фраз реплики (L-15b), и разрез, который генератор фраз обязан предложить (`test_turn_phrases.py`).
TRUE_CUT: dict[str, tuple[str, ...]] = {
    "name_parens_commas": ("Прием (осмотр, консультация) врача-терапевта",),
    "name_commas_in_parens": ("Общий анализ крови (Le, Er, Hb, СОЭ)",),
    "name_plus_in_parens": ("Глюкозотолерантный тест (глюкоза+ C-пептид)",),
    "name_plus_outside": ("УЗИ органов мошонки + ЦДК сосудов",),
    "name_and_outside_parens": ("Билирубин и его фракции (общий, прямой и непрямой)",),
    "name_and_in_parens": ("Экспресс-тест для определения сифилиса (антитела IgG и IgM к Treponema pallidum)",),
    "name_and_between_numerals": ("Пепсиногены  I и II с расчетом соотношения",),
    "name_many_commas": ("Исследование кала на паразиты (лямблии, описторхи, аскариды, острицы, власоглав, карликовый цепень и др.)",),
    "phrase_smear_throat": ("мазок на микрофлору с поверхности задней стенки глотки",),
    "phrase_smear_pharynx": ("мазок на микрофлору из зева",),
    "phrase_oms": ("приём терапевта по ОМС",),
    "readiness_extra_words": ("анализ на ферритин",),
    "list_true_three": ("ферритин", "ттг", "витамин д"),
    "list_one_unrecognized": ("ферритин", "глюкоза", "спирометрия"),
    "list_caps_run_on_priced": ("ФЕРРИТИН", "ГЛЮКОЗА", "ХОЛЕСТЕРОЛ"),
    "list_caps_run_on_bare": ("ОАК", "ОАМ", "ФЕРРИТИН", "ГЛЮКОЗА"),
}


class _OracleLLM:
    """LLM-оракул: на выборе строки прайса, корзины и срока отмечает ровно строки разметки случая.

    Корзина — по позиции: строка выбирается, только если «позиция» — услуга из разметки и строка — её;
    обрывок фразы («мазок на») — не услуга, для него ничего. Цена и срок — любая строка разметки. Выбор
    фраз реплики (L-15b) — ровно фразы разреза случая (`cut`; не задан — «недоступна»). Прочие места —
    «недоступна», как в мире гейта.
    """

    def __init__(self, case: Case, cut: tuple[str, ...] = ()) -> None:
        self.by_fragment = {fragment.lower(): set(names) for fragment, names in case.positions.items()}
        self.names = {name for names in case.positions.values() for name in names}
        self.cut = {_normalise_catalog_text(text) for text in cut}
        self.calls: list[str] = []
        self.picked: set[str] = set()

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if self.cut and _PHRASES_MARK in seen:
            self.calls.append("phrases")
            lines = (line.partition(". ") for line in prompt.splitlines())
            picks = [int(head) for head, dot, rest in lines if dot and head.isdigit() and _normalise_catalog_text(rest.strip("«»")) in self.cut]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
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
            name = _norm(rest.rsplit(" — ", 1)[0])
            if name in {_norm(row) for row in wanted}:
                picks.append(int(head))
                self.picked.add(name)
        answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _norm(name: str) -> str:
    return " ".join(str(name).split())


@functools.cache
def _catalog_names() -> frozenset[str]:
    path = FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl"
    return frozenset(_norm(json.loads(line)["serviceName"]) for line in path.open(encoding="utf-8") if line.strip())


def _priced_sections(text: str) -> list[list[str]]:
    """Строки с ценой по разделам ответа (раздел корзины — позиция; ответ одной услуги — один раздел)."""

    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        if _is_basket_section(line):
            sections.append([])
        match = _PRICED.match(line)
        if match:
            sections[-1].append(_norm(match.group("name")))
    return [section for section in sections if section]


def _violations(case: Case, text: str, llm: str, picked: frozenset[str] | set[str] = frozenset()) -> list[str]:
    """Нарушения инвариантов класса в ответе; `picked` — строки, которые выбрала LLM (оракул)."""

    sections = _priced_sections(text)
    priced = [name for section in sections for name in section]
    truth = {fragment: {_norm(n) for n in names} for fragment, names in case.positions.items()}
    every = set().union(*truth.values())
    strong = [truth[fragment] for fragment in sorted(case.strong)]
    out: list[str] = []
    unnamed = sorted(set(priced) - _catalog_names())
    if unnamed:
        out.append(f"(в) цена подписана не полным названием строки прайса: {unnamed}")
    if case.kind in ("one", "readiness"):
        if any(_is_basket_section(line) for line in text.splitlines()) or _UNRECOGNIZED in text:
            out.append("(а) одна услуга разрезана на позиции корзины")
        if case.exact and not priced:
            out.append("(а) дословное название без цены")
        if case.exact and set(priced) - every:
            out.append(f"(а) цена чужих строк при дословном названии: {sorted(set(priced) - every)}")
        if case.kind == "one" and llm == "oracle" and set(priced) - every:
            out.append(f"(а) чужая цена при живой LLM: {sorted(set(priced) - every)}")
        if not case.exact and len(priced) == 1 and priced[0] not in picked:
            out.append(f"(в) одиночная цена без сильного совпадения и без выбора LLM: {priced}")
    if case.kind == "readiness":
        if priced or re.search(r"\d\s*руб", text):
            out.append("вопрос о сроке получил цену")
        allowed = {case.deadline} if llm == "oracle" else {case.deadline, RESULT_TIMING_OPERATOR_TEXT}
        if text not in allowed:
            out.append(f"вопрос о сроке: не срок из МИС и не честный перевод: {text[:120]!r}")
    if case.kind == "list":
        for section in sections:
            own = next((names for names in strong if names & set(section)), None)
            if own is not None and not set(section) <= own:
                out.append(f"(в) у сильной позиции чужая строка: {sorted(set(section) - own)}")
            if own is None and len(section) == 1 and section[0] not in picked:
                out.append(f"(в) одиночная цена без сильного совпадения и без выбора LLM: {section}")
            if own is None and llm == "down" and len(section) > _VARIANTS_MAX:
                out.append(f"(в) у слабой позиции больше {_VARIANTS_MAX} вариантов: {len(section)}")
        missing = [fragment for fragment in sorted(case.strong) if not truth[fragment] & set(priced)]
        if missing:
            out.append(f"(в) сильная позиция без цены: {missing}")
        if llm == "oracle":
            if set(priced) - every:
                out.append(f"(б) чужая цена при живой LLM: {sorted(set(priced) - every)}")
            for section in sections:
                if not any(set(section) <= names for names in truth.values() if names):
                    out.append(f"(б) в разделе строки разных позиций: {section}")
            missing = [fragment for fragment, names in truth.items() if names and not names & set(priced)]
            if missing:
                out.append(f"(б) позиции без своей цены при живой LLM: {missing}")
            if picked - set(priced):
                out.append(f"(б) выбранные LLM строки без цены: {sorted(picked - set(priced))}")
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


def _configure(monkeypatch, config: str) -> None:
    from messengers_router.runtime_config import config as cfg

    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)


def _class_turn(name: str, config: str, llm: str, world_dir, monkeypatch) -> tuple[dict, dict | None, _OracleLLM]:
    case = CASES[name]
    _configure(monkeypatch, config)
    oracle = _OracleLLM(case, TRUE_CUT[name])
    if llm == "oracle":
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([case.text], f"one-or-many-{name}-{config}-{llm}")
        attempts, row = list(world.net.attempts), world.take_telemetry()
    assert attempts == []
    return turn, row, oracle


@pytest.mark.parametrize(("name", "config", "llm"), list(_params()))
def test_one_service_or_several(name, config, llm, mini_prod_world, monkeypatch):
    turn, _row, oracle = _class_turn(name, config, llm, mini_prod_world, monkeypatch)
    violations = _violations(CASES[name], turn["text"], llm, oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", turn["text"]])


@pytest.mark.parametrize("name", sorted(CASES))
def test_parse_on_spends_at_most_two_decisions(name, mini_prod_world, monkeypatch):
    """Под `turn_parse = on` (живая LLM) решений хода не больше двух — лимит не меняли (решение владельца 13.10,
    Q4): выбор фраз — первое, выбор строк корзины, строки цены или срока — второе; бюджет хода не исчерпан."""

    _turn, row, oracle = _class_turn(name, "deploy_parse", "oracle", mini_prod_world, monkeypatch)
    decisions = [(entry["task"], entry["status"]) for entry in row.get("choose") or [] if not entry.get("shadow")]
    assert len(decisions) <= 2, decisions
    assert "phrases" not in oracle.calls or decisions[0] == ("turn_phrases", "picked"), decisions
    assert not [call for call in row.get("llm") or [] if call.get("outcome") == "budget"], row.get("llm")


@pytest.mark.parametrize("name", sorted(CASES))
def test_parse_on_with_llm_down_names_no_lone_weak_price(name, mini_prod_world, monkeypatch):
    """Под `turn_parse = on` при недоступной LLM одиночная цена — только у сильного по данным (дословное
    название, синоним МИС, код): ни одна слабая позиция и ни одна фраза одной услуги не получает одну строку
    как «ту самую», при дословном названии нет цен соседей. Держится во всех 16 случаях, в том числе в двух
    xfail основного теста (там нарушено другое: разрез фразы одной услуги и «не нашёл» сплошного списка)."""

    turn, _row, _oracle = _class_turn(name, "deploy_parse", "down", mini_prod_world, monkeypatch)
    lone = [v for v in _violations(CASES[name], turn["text"], "down") if "одиночная цена" in v or "чужих строк" in v or "вариантов" in v]
    assert lone == [], "\n".join([*lone, "--- ответ:", turn["text"]])


def test_oracle_answers_the_selection_it_was_asked(mini_prod_world, monkeypatch):
    """Оракул действительно выбирает: на живом наборе деплоя слабые позиции списка получают ровно свои
    строки, а строки другой услуги рядом («Антитела к рецепторам ТТГ», акции витамина D) — нет."""

    _configure(monkeypatch, "deploy")
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
        ("1. Ферритин — 530 руб.\n2) Розничная цена: Глюкоза — 1 190 руб. Формат: поликлиника.", [["Ферритин", "Глюкоза"]]),
        ("По запросу:\n\nПо услуге «ОАК»:\n1. А — 1 руб.\n\nПо услуге «ОАМ»:\n1. Б — 2 руб.", [["А"], ["Б"]]),
        ("Срок готовности «Ферритин» — 1–2 дня.", []),
    ],
)
def test_answer_parser_reads_priced_rows_by_section(text, expected):
    assert _priced_sections(text) == expected


def test_violations_catch_a_cut_name_and_a_foreign_row():
    case = CASES["name_parens_commas"]
    cut = "По услуге «Прием»:\n1. Прием (осмотр, консультация) врача-терапевта — 2 700 руб.\nНе распознал: консультация"
    assert _violations(case, cut, "down")
    oak = CASES["list_caps_run_on_bare"]
    foreign = "По услуге «ОАМ»:\n1. Общий анализ мочи — 280 руб.\n2. Общий анализ крови (Le, Er, Hb, СОЭ) — 390 руб."
    assert any(v.startswith("(б) в разделе") for v in _violations(oak, foreign, "oracle"))


def test_lone_weak_price_needs_strong_data_or_the_model_choice():
    case = CASES["list_true_three"]
    lone = "По услуге «Ферритин»:\n1. Ферритин — 530 руб.\n\nПо услуге «ТТГ (TSH) тиреотропный гормон»:\n1. ТТГ (TSH) тиреотропный гормон — 400 руб."
    variants = lone + "\n2. Антитела к рецепторам ТТГ — 1 460 руб."
    assert any("одиночная цена" in v for v in _violations(case, lone, "down"))
    assert not any("одиночная цена" in v for v in _violations(case, lone, "oracle", {"ТТГ (TSH) тиреотропный гормон"}))
    assert not any("одиночная цена" in v for v in _violations(case, variants, "down"))
    assert any("не полным названием" in v for v in _violations(case, "1. ТТГ — 400 руб.", "down"))
