"""Цена, этап 2 «без стоп-слов» — характеристики читателей вырезателя через эндпоинт (DESIGN-2026-10-25-PRICE-STAGE2).

Второй слой характеристик (первый — `tests/test_price_stage2_readers.py`, каждый читатель по отдельности): ходы через
обработчик эндпоинта `messenger_generate_once` с выключателями прода деплоя 17 (`turn_spans_llm`, `turn_understanding_one`,
`prepare_by_parse`, `price_select_rules = general` и др.), мини-прод, день, сеть закрыта. Здесь видно то, чего не видит
читатель по отдельности: какой путь хода выбран (разбор LLM или правила), что пациент прочитал, что ушло клинике в журнал
синонимов (K7) и сколько решений LLM потратил ход.

Два режима LLM на каждый ход:
- `llm` — оракул: разметка слов (`turn_spans`) отмечает ровно фразы случая; выбор строки прайса — строки с заданным началом
  названия (`rows`), пустой выбор (`()`) — «ни одна не подходит»; прочие места — «недоступна» (как в гейте);
- `down` — LLM недоступна везде: разбор `unavailable` без разреза, и ход ведут правила — именно этот путь этап 2 меняет
  по решению владельца 22.10 («без LLM услугу не угадывать: оффер или „уточните“»).

На каждый ход фиксируется: текст ответа, перевод на оператора, метка, план, услуга инструмента и сессии, висящий вопрос,
поля телеметрии хода (статус разбора, путь понимания, политика цены, решения LLM) и строки журнала синонимов. Эталон
`tests/golden/price_stage2_turns.json` — поведение как есть, с известными дефектами: этап 2 меняет его только перезаписью
с разбором диффа (`GOLDEN_UPDATE=1 pytest tests/test_price_stage2_turns.py`).

Код этапа 2 — под выключателем `price_by_parse` (дизайн §5): каждый ход снимается в обоих положениях. Эталон `off` — путь
прода, код этапа 2 его не меняет ни на байт; эталон `on` (`tests/golden/price_stage2_turns_on.json`) — дифф шагов Ш1–Ш5, каждое
отличие разобрано в журнале багов (STEP-2026-10-25-PRICE-STAGE2-CODE).

Реплики — классов, не вопросы эталона стенда (Гудхарт, 25.10).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

GOLDEN = Path(__file__).resolve().parent / "golden" / "price_stage2_turns.json"
GOLDENS = {"off": GOLDEN, "on": GOLDEN.with_name("price_stage2_turns_on.json")}
SWITCH = tuple(GOLDENS)
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"

# Выключатели прода деплоя 17 (24.10), которые читает код этой ветки; снятые веткой снятия выключателей — не нужны.
PROD_D17: dict[str, Any] = {
    "MR_TURN_UNDERSTANDING_PRICE": "on",
    "MR_TURN_UNDERSTANDING_ONE": "on",
    "MR_TURN_SPANS_LLM": "on",
    "MR_CHOOSE_READINESS": "on",
    "MR_CHOOSE_SYMPTOM": "on",
    "MR_CHOOSE_CACHE_TTL_S": 0,
    "MR_PARSE_FAIL_VARIANTS": 3,
    "MR_MIS_HONEST_DEGRADATION": "on",
    "MR_TOPIC_OVERRIDE_LLM": "off",
    "MR_PARSE_FRAGMENT_WINDOWS": "off",
    "MR_PARSE_COMPLETE_WINDOWS": "off",
    "MR_PENDING_BUTTONS": "on",
    "MR_ROLES_CATALOG_LLM": "on",
    "MR_CATALOG_CONFIRM_LLM": "on",
    "MR_TOPIC_BY_LLM": "rule_test_assist,rule_doctor_info_specialty,rule_schedule,rule_analysis_lookup_ambiguous,"
    "rule_website_help,rule_address_work_hours,rule_test_result,rule_prepare",
    "MR_P1_FACET_NONE_OFFER": "on",
    "MR_PRICE_BUDGET_CLARIFY": "on",
    "MR_DOCTORS_BY_SERVICE_DATA": "shadow",
    "MR_PREPARE_GENERAL_MEMO_LLM": "on",
    "MR_BUTTON_PASS_RULES": "on",
    "MR_PRICE_GENERAL_CLARIFY": "on",
    "MR_SYNONYM_WORDFORMS": "on",
    "MR_PRICE_SELECT_RULES": "general",
    "MR_PREPARE_BY_PARSE": "on",
    "MR_GROUNDER_GENERIC_BY_LLM": "on",
    "MR_BIOMATERIAL_BY_MIS": "on",
    "MR_TEST_ASSIST_FROM_DATA": "on",
    # Оракул отвечает только разметке и выбору строки; сбои прочих мест не должны размыкать предохранитель для них.
    "MR_LLM_DECISION_BREAKER_FAILURES": 0,
}

_SPANS_MARK = "Слова сообщения с номерами:"
_ROWS_MARKS = ("ответить на вопрос о ценах нескольких услуг", "ответить пациенту на вопрос о цене")
MODES = ("llm", "down")


@dataclass(frozen=True)
class Case:
    turns: tuple[str, ...]
    phrases: tuple[str, ...]  # что LLM разметки отмечает услугами (пусто — «ни одной»)
    rows: tuple[str, ...] | None = None  # начало названий, которые выбирает LLM строки; () — «ни одна»; None — недоступна
    why: str = field(default="", compare=False)
    spans: str = ""  # ответ разметки как есть (не по контракту — `invalid`); пусто — по фразам случая
    reason: str = ""  # код причины пустого выбора строки (`qualifier_absent` — «той услуги с условием нет»)


CASES: dict[str, Case] = {
    # одна услуга: что пациент назвал, решает разметка; строки — данные или выбор
    "one_strong": Case(("сколько стоит ферритин",), ("ферритин",), why="фраза разметки — дословное название"),
    "one_analysis_on": Case(("сколько стоит анализ на ферритин",), ("ферритин",), rows=("Ферритин",), why="обвязка «анализ на»"),
    "one_analysis_on_grouped": Case(
        ("сколько стоит анализ на ферритин",), ("анализ на ферритин",), rows=("Ферритин",), why="правило 5 промпта разметки: «анализ на» — в группе"
    ),
    "one_analyses_on": Case(("сколько стоят анализы на ферритин",), ("ферритин",), rows=("Ферритин",), why="словоформы вне списка («стоят»)"),
    "one_strong_parse_select_down": Case(("сколько стоят анализы на ферритин",), ("ферритин",), why="сильная фраза разметки, выбор строки недоступен"),
    "one_two_letter": Case(("сколько стоит ЛГ",), ("ЛГ",), rows=("ЛГ (LH)",), why="сокращение в две буквы"),
    "one_two_letter_lower": Case(("сколько стоит лг",), ("лг",), rows=("ЛГ (LH)",), why="строчными: «названа ли» — по разбору, не по заглавным"),
    "one_consult": Case(("сколько стоит прием уролога",), ("прием уролога",), rows=("Прием (осмотр, консультация) врача-уролога",), why="приём врача со специальностью"),
    "one_consult_generic": Case(("сколько стоит прием врача",), ("прием врача",), rows=(), why="приём без специальности"),
    "one_generic_word": Case(("сколько стоят анализы",), (), why="общее слово — разметка: ни одной услуги"),
    # разметка оторвала условие от дословного названия (живая LLM — 59 из 82 свипа П2, карточка Ш7): цена — только по суду условия
    "condition_cut": Case(
        ("сколько стоит ферритин по омс",), ("ферритин",), rows=(), reason="qualifier_absent", why="суд: той услуги с условием нет"
    ),
    "condition_cut_judge_down": Case(("сколько стоит ферритин по омс",), ("ферритин",), why="суд условия недоступен"),
    # выбор строки LLM у слабой фразы
    "weak_select_picks": Case(("сколько стоит кардиограмма",), ("кардиограмма",), rows=("ЭКГ",), why="слово пациента не из прайса"),
    "weak_select_down": Case(("сколько стоит кардиограмма",), ("кардиограмма",), why="выбор строки недоступен — ответ правил"),
    "weak_abbr": Case(("сколько стоит оак",), ("оак",), rows=("Общий анализ крови (Le",), why="сокращение"),
    # журнал синонимов клинике (K7): промах — фраза разметки, а не реплика и не вырезатель
    "journal_miss": Case(("сколько стоит спирометрия",), ("спирометрия",), rows=(), why="услуги нет в прайсе"),
    "journal_wrapped": Case(("сколько стоит флороценоз подскажите пожалуйста",), ("флороценоз",), rows=(), why="обвязка вокруг"),
    "journal_grouped": Case(
        ("сколько стоит анализ на флороценоз",), ("анализ на флороценоз",), rows=(), why="фраза разметки с «анализ на» (правило 5 промпта)"
    ),
    # LLM работает, а разбор не решён или «ни одна фраза не услуга»: строку по-прежнему выбирает LLM (карточка Ш7, 25.10)
    "parse_invalid": Case(
        ("сколько стоит анализ ферритина",), (), rows=("Ферритин",), spans='{"phrases": [[1, 3], [2, 4]]}', why="ответ разметки не по контракту"
    ),
    "parse_none_fit_findable": Case(("сколько стоит анализ ферритина",), (), rows=("Ферритин",), why="разметка: ни одна не услуга, строка есть"),
    # корзина: позиции — фразы разметки
    "basket_commas": Case(("ферритин, глюкоза, холестерин",), ("ферритин", "глюкоза", "холестерин"), rows=("Холестерол",)),
    "basket_run_on": Case(("ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",), ("ОАК", "ОАМ", "ФЕРРИТИН", "ГЛЮКОЗА"), rows=("Общий анализ крови (Le", "Общий анализ мочи")),
    # другие свойства услуги — тот же разбор
    "readiness_named": Case(("когда будет готов анализ на ферритин",), ("ферритин",), why="срок по данным"),
    "prepare_named": Case(("как подготовиться к анализу на глюкозу",), ("глюкозу",), why="подготовка по фразам разбора"),
    "prepare_pick_by_number": Case(
        ("сколько стоит ферритин, глюкоза", "а как подготовиться?", "1"), ("ферритин", "глюкоза"), rows=(), why="номер из показанных — не код услуги"
    ),
}

# Класс характеристик → случаи (на них ссылаются объявления читателей в `test_price_stage2_readers.DECLARED`).
CLASSES: dict[str, tuple[str, ...]] = {
    "price_one": (
        "condition_cut",
        "condition_cut_judge_down",
        "one_strong",
        "one_analysis_on",
        "one_analysis_on_grouped",
        "one_analyses_on",
        "one_strong_parse_select_down",
        "one_two_letter",
        "one_two_letter_lower",
        "one_consult",
        "one_consult_generic",
        "one_generic_word",
    ),
    "price_select": ("weak_select_picks", "weak_select_down", "weak_abbr", "parse_invalid", "parse_none_fit_findable"),
    "journal": ("journal_miss", "journal_wrapped", "journal_grouped"),
    "basket": ("basket_commas", "basket_run_on"),
    "readiness": ("readiness_named",),
    "prepare": ("prepare_named", "prepare_pick_by_number"),
}

# Поля телеметрии хода, которые показывают путь читателей (без текста пациента).
_TELEMETRY = (
    "tu_parse_status",
    "tu_rule",
    "tu_mention_src",
    "tu_mention_strength",
    "tu_acted",
    "price_policy",
    "price_guard",
    "tu_decisions_denied",
)


class _Oracle:
    """Разметка слов — ровно фразы случая; выбор строки — строки с заданным началом названия; прочее — «недоступна»."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self.asked: list[str] = []

    def _spans(self, prompt: str) -> dict[str, Any]:
        numbered = prompt.split(_SPANS_MARK, 1)[1].strip().splitlines()[0].split()
        words = [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]
        groups = []
        for phrase in self.case.phrases:
            want = _normalise_catalog_text(phrase).split()
            for k in range(len(words) - len(want) + 1):
                if words[k : k + len(want)] == want:
                    groups.append([k + 1, k + len(want)])
                    break
        return {"phrases": groups}

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _SPANS_MARK in prompt:
            self.asked.append("spans")
            return GenerateResponse(model="m", response=self.case.spans or json.dumps(self._spans(prompt)), done=True)
        if self.case.rows is not None and any(mark in seen for mark in _ROWS_MARKS):
            self.asked.append("rows")
            options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
            picks = [n for n, text in options if self.case.rows and text.split(" → ", 1)[-1].startswith(self.case.rows)]
            answer = {"match": picks, "search_terms": []} if system is None else {"picks": picks}
            if self.case.reason and not picks:
                answer["reason"] = self.case.reason
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


class _Down:
    async def generate(self, **_kwargs):
        raise ConnectionError("тест: LLM недоступна")


def _journal(world_dir: Path) -> list[dict[str, Any]]:
    from messengers_router.services import _price_select_llm

    path = Path(_price_select_llm._journal_path())
    if not path.exists():
        return []
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{key: entry.get(key) for key in ("phrase", "source", "found") if key in entry} for entry in entries]


def _turns(case_name: str, mode: str, switch: str, world_dir: Path, monkeypatch) -> dict[str, Any]:
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D17, "MR_PRICE_BY_PARSE": switch}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    case = CASES[case_name]
    oracle = _Oracle(case) if mode == "llm" else _Down()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        records = run_dialog(list(case.turns), f"stage2-{case_name}-{mode}")
        telemetry = list(world.telemetry)
        attempts = list(world.net.attempts)
        journal = _journal(world_dir)
    assert attempts == []
    for record, row in zip(records, telemetry):
        record["telemetry"] = {key: row.get(key) for key in _TELEMETRY if row.get(key) not in (None, [], "")}
        record["llm"] = [[entry.get("task"), entry.get("status")] for entry in row.get("choose") or [] if not entry.get("shadow")]
    return {"turns": records, "journal": journal, "asked": list(getattr(oracle, "asked", []))}


def _load(switch: str = "off") -> dict[str, Any]:
    path = GOLDENS[switch]
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _store(key: str, record: dict[str, Any], switch: str) -> None:
    data = _load(switch)
    data[key] = record
    GOLDENS[switch].write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_turn_matches_golden(case_name, mode, switch, mini_prod_world, monkeypatch):  # noqa: F811
    record = json.loads(json.dumps(_turns(case_name, mode, switch, mini_prod_world, monkeypatch), ensure_ascii=False))
    for turn in record["turns"]:
        assert turn["text"].strip(), (case_name, mode, switch, turn)
    key = f"{case_name}-{mode}"
    if UPDATE:
        _store(key, record, switch)
        return
    base = _load(switch).get(key)
    assert base is not None, f"нет эталона для {key} ({switch}): GOLDEN_UPDATE=1"
    assert record == base


@pytest.mark.parametrize("switch", SWITCH)
def test_golden_has_no_orphans(switch):
    assert set(_load(switch)) == {f"{name}-{mode}" for name in CASES for mode in MODES}


def test_classes_cover_every_case():
    assert sorted(name for names in CLASSES.values() for name in names) == sorted(CASES)
