"""Запасной путь правил под `turn_parse = on` — характеризация (golden master) перед T7, шаг 3 (15.10).

Под `turn_parse = on` цену читают из разбора реплики на фразы (`turn_parse.acting_for`). Когда разбор не годится
для действия — статусы `empty`/`too_many`, сбой LLM без разделителей (решение владельца 13.10, Q2: «сплошной текст
без LLM не режем»), — `acting_for` даёт None, и ход ведут прежние правила: защиты `price_list_services` (K1:
«одно название покрывает», «сплошной список — только распознанный»), гард третьего стыка корзины (K3) и
сегментатор сплошного списка S2 (K2). На dev при недоступной LLM это ~20% ценовых ходов, при живой ~3%. Разбор
14.10 (`~/Dev/handoffs/dla/12_t7_heuristics_removal.md`, K1, раздел 3а): ни один тест под `on` от защит K1 не
зависел — корпус не покрывал запасной путь из dev `down`. Этот файл его закрепляет до снятия.

Инварианты класса (через обработчик эндпоинта, мини-прод, день, сеть закрыта, LLM недоступна):
- путь хода — тот, что задуман: `fallback` — ни один читатель не получил разбора (`acting_for` = None на каждом
  вызове), понимание хода — правил (`tu_rule` без `parse_`); `policy` — сбой LLM с разрезом по разделителям ≥2
  (`unavailable`/`merged`): читатели действуют разрезом (политика сбоя, `tu_rule = parse_<статус>`) — это НЕ
  запасной путь; `decided` — разбор решён без LLM (`single`);
- ответ, перевод на оператора, метка, план, услуга и телеметрия хода (`tu_parse_status`, `tu_rule`) — как в эталоне
  `tests/golden/price_turns_parse_fallback.json` (побайтно). Эталон держит поведение как есть, вместе с известными
  дефектами запасного пути (сегментатор режет фразу одной услуги, вопрос о сроке получает цену): намеренная правка
  меняет его только перезаписью с разбором диффа;
- два набора выключателей: `deploy_parse` (набор деплоя классовых тестов + `turn_parse = on`, `parse_fail_variants = 5`) и `prod` (прод 15.10:
  ещё `parse_fail_variants = 3`, `journal_phrase = mention`, `context_shown = on`, `strong_match = data`); ленивые
  правила (`parse_agree_telemetry = off`, T7 шаг 4) дают эталон `prod` без изменений;
- два хода: короткий вопрос после ответа запасного пути не получает цены, которой пациент не видел.

Проверка ценности (15.10, ветка вычеркнута в памяти, код на диске не менялся): без K1а «одно название покрывает»
падают `one_name_covers_*` (5 классов; у четырёх с разделителями ответ — корзина с чужими строками: приёмы уролога и
онколога, краснуха IgM); без K1б «сплошной список — только распознанный» — `run_on_extra_*`, `run_on_ten_four`
(подбор анализов становится корзиной); без гарда K3 — `qualifier_oms_run_on` (платные ферритин и глюкоза на вопрос
о полисе), `phrase_qualifier_throat` (корзина из обрывков фразы), `readiness_run_on_priced` (телеметрия).

Перезапись эталона после НАМЕРЕННОЙ правки: `GOLDEN_UPDATE=1 pytest tests/test_turn_parse_fallback_class.py`, затем
дифф `tests/golden/price_turns_parse_fallback.json` — в коммит вместе с правкой и с объяснением.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_parse
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_shadow import _deploy
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _assert_follow_up_stays_on_what_was_shown, _Oracle

GOLDEN = Path(__file__).resolve().parent / "golden" / "price_turns_parse_fallback.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"

# Наборы выключателей поверх набора деплоя (`_deploy`); `prod_lazy` сверяется с эталоном `prod`. У `deploy_parse` пять
# вариантов при сбое — явно: эталон снят при прежнем умолчании, `prod` держит три (STEP-2026-10-15-DEFAULTS-B).
_SETS: dict[str, dict[str, Any]] = {
    "deploy_parse": {"MR_TURN_PARSE": "on", "MR_PARSE_FAIL_VARIANTS": 5},
    "prod": {
        "MR_TURN_PARSE": "on",
        "MR_PARSE_FAIL_VARIANTS": 3,
        "MR_JOURNAL_PHRASE": "mention",
        "MR_CONTEXT_SHOWN": "on",
        "MR_STRONG_MATCH": "data",
    },
}
_SETS["prod_lazy"] = {**_SETS["prod"], "MR_PARSE_AGREE_TELEMETRY": "off"}
_GOLDEN_SETS = ("deploy_parse", "prod")

_FALLBACK_STATUSES = {"empty", "too_many", "unavailable"}
_POLICY_STATUSES = {"unavailable", "merged"}


@dataclass(frozen=True)
class Case:
    """Реплика, путь хода (`fallback` | `policy` | `decided`) и правило, на котором держится класс.

    `whole` — LLM разбора выбирает всю реплику одной фразой (оракул), остальные места — «недоступна»; иначе LLM
    недоступна вовсе (гейт)."""

    text: str
    path: str
    holds: str
    whole: str = ""


_TOO_MANY = (
    "сколько стоит ферритин, глюкоза, холестерол, ттг, оак, оам, витамин д, кальций, железо, магний, калий, натрий, "
    "хлор, мочевина, креатинин, билирубин"
)

CASES: dict[str, Case] = {
    # --- сплошной список без разделителей (S2, K2)
    "run_on_bare": Case("ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА", "fallback", "S2: list_segment — корзина правил"),
    "run_on_words": Case("сколько стоит общий анализ крови ферритин глюкоза билирубин", "fallback", "S2: list_segment"),
    "run_on_caps_priced": Case("сколько стоит ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ", "fallback", "S2 без списка — «не нашёл»"),
    "run_on_ten_four": Case("ферритин ттг т4 свободный", "fallback", "S2 в подборе анализов (lab_tests)"),
    "run_on_unknown_word": Case("сколько стоит ферритин глюкоза спирометрия", "fallback", "S2 без списка — «Уточните»"),
    # --- сплошной список с лишними словами (K1 «сплошной список — только распознанный»)
    "run_on_extra_words": Case("сколько стоит сдать общий анализ крови ферритин глюкоза натощак", "fallback", "K1б: segment_unrecognized"),
    "run_on_extra_service": Case("сколько стоит оак оам ферритин глюкоза узи", "fallback", "K1б: segment_unrecognized"),
    # --- одно название прайса покрывает весь текст (K1 «одно название покрывает»): не дословно, поэтому разбор не
    # сильный; разрез по разделителям одно название гасит (`fallback_split`) — запасной путь, а не политика сбоя
    "one_name_covers_run_on": Case("сколько стоит Билирубин общий прямой непрямой", "fallback", "K1а: one_catalog_name"),
    "one_name_covers_commas": Case("сколько стоит прием, осмотр и консультация терапевта", "fallback", "K1а: one_catalog_name"),
    "one_name_covers_and": Case("сколько стоит билирубин и фракции общий, прямой, непрямой", "fallback", "K1а: one_catalog_name"),
    "one_name_covers_plus": Case("сколько стоит глюкозотолерантный тест, глюкоза и C-пептид", "fallback", "K1а: one_catalog_name"),
    "one_name_covers_igg": Case("сколько стоит экспресс-тест на сифилис IgG и IgM", "fallback", "K1а: one_catalog_name"),
    # --- уточнение, которого в прайсе нет (П2), в сплошном тексте: K3 не даёт платных цен на вопрос о полисе
    "qualifier_oms_run_on": Case("сколько стоит ферритин глюкоза по омс", "fallback", "K3: «омс» в остатке гасит корзину"),
    # --- фраза одной услуги (S2 режет; K3 гасит)
    "phrase_qualifier_throat": Case(
        "сколько стоит мазок на микрофлору с поверхности задней стенки глотки", "fallback", "K3: уточнение внутри фразы"
    ),
    "phrase_smear_pharynx": Case("сколько стоит мазок на микрофлору из зева", "fallback", "S2 режет одну услугу (известный дефект)"),
    "phrase_scraping_worms": Case("сколько стоит соскоб на яйца глистов", "fallback", "S2 режет одну услугу (известный дефект)"),
    # --- вопрос о сроке и сплошной список
    "readiness_run_on": Case("сколько по времени делается ОАК ОАМ ферритин глюкоза", "fallback", "срок: list_segment — перевод"),
    "readiness_run_on_priced": Case(
        "когда будут готовы ОАК ОАМ ферритин", "fallback", "срок получает цену одной услуги (известный дефект)"
    ),
    # --- статусы разбора без LLM
    "status_empty": Case("сколько стоит?", "fallback", "empty"),
    "status_too_many": Case(_TOO_MANY, "fallback", "too_many: корзина правил по разделителям"),
    # --- уточнение П2 без услуги: разбор решён (`single`), правила списка не доходят до гарда
    "qualifier_oms": Case("сколько стоит приём терапевта по ОМС", "decided", "single: «Уточните», без платных приёмов"),
    # --- контроль: сбой LLM с разделителями — политика сбоя, не запасной путь
    "control_separated": Case("сколько стоит ферритин, глюкоза и ОАК", "policy", "unavailable + разрез ≥2"),
    "control_readiness_separated": Case("когда будет готов анализ на ферритин и глюкозу", "policy", "unavailable + разрез ≥2"),
    "control_merged": Case(
        "сколько стоит ферритин и анализ на сахар", "policy", "merged + разрез ≥2", whole="ферритин и анализ на сахар"
    ),
}


def _spy_acting_for(monkeypatch) -> list[tuple[str, bool]]:
    """Вызовы `acting_for` за диалог: вход читателя и получил ли он разбор (не None)."""

    calls: list[tuple[str, bool]] = []
    real = turn_parse.acting_for

    def spy(text: str):
        parse = real(text)
        calls.append((str(text or ""), parse is not None))
        return parse

    monkeypatch.setattr(turn_parse, "acting_for", spy)
    return calls


def _dialog(name: str, config: str, turns: list[str], monkeypatch, world_dir) -> list[dict[str, Any]]:
    """Ходы через эндпоинт под набором `config`; запись хода — запись эталона цены плюс путь разбора."""

    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    for key, value in _SETS[config].items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    case = CASES[name]
    if case.whole:
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Oracle((case.whole,)))
    calls = _spy_acting_for(monkeypatch)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        records = run_dialog(turns, f"parse-fallback-{name}-{config}")
        rows, attempts = list(world.telemetry), list(world.net.attempts)
    assert attempts == []
    assert len(rows) == len(records), rows
    # Путь разбора — у первого хода: разбор действует только на свою реплику (`for_text`), поэтому получить его мог
    # лишь читатель со входом — репликой первого хода.
    records[0].update(
        tu_parse_status=rows[0].get("tu_parse_status"),
        tu_rule=rows[0].get("tu_rule"),
        parse_acted=any(acted for text, acted in calls if text == turns[0]),
    )
    return records


def _load() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}


def _store(key: str, record: list[dict[str, Any]]) -> None:
    data = _load()
    data[key] = record
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _assert_path(case: Case, turn: dict[str, Any]) -> None:
    status, rule, acted = turn["tu_parse_status"], str(turn["tu_rule"] or ""), turn["parse_acted"]
    if case.path == "fallback":
        assert status in _FALLBACK_STATUSES and not acted and not rule.startswith("parse_"), (status, rule, acted)
    elif case.path == "policy":
        assert status in _POLICY_STATUSES and acted and rule == f"parse_{status}", (status, rule, acted)
    else:
        assert status == "single" and acted and rule == "parse_single", (status, rule, acted)


@pytest.mark.parametrize("config", [*_GOLDEN_SETS, "prod_lazy"])
@pytest.mark.parametrize("name", sorted(CASES))
def test_fallback_path_matches_golden(name, config, mini_prod_world, monkeypatch):  # noqa: F811
    case = CASES[name]
    [turn] = _dialog(name, config, [case.text], monkeypatch, mini_prod_world)
    assert turn["text"].strip(), turn
    _assert_path(case, turn)
    golden_config = "prod" if config == "prod_lazy" else config
    key = f"{name}-{golden_config}"
    if UPDATE and config in _GOLDEN_SETS:
        _store(key, [turn])
        return
    expected = _load().get(key)
    assert expected is not None, f"нет эталона для {key}: GOLDEN_UPDATE=1"
    assert [turn] == expected


def test_golden_has_no_orphans():
    """Эталон держит ровно случаи набора (переименование не теряет случай молча)."""

    assert set(_load()) == {f"{name}-{config}" for name in CASES for config in _GOLDEN_SETS}


def test_the_corpus_reaches_each_guard_on_the_fallback():
    """Встречная проверка набора: запасной путь здесь идёт через каждое правило, которое снимает T7 шаг 3 —
    защиты K1 (`one_catalog_name`, `segment_unrecognized`), S2 (`list_segment`) и оба статуса без LLM."""

    golden = _load()
    rules = {golden[f"{name}-prod"][0]["tu_rule"] for name, case in CASES.items() if case.path == "fallback"}
    statuses = {golden[f"{name}-prod"][0]["tu_parse_status"] for name, case in CASES.items() if case.path == "fallback"}
    assert {"one_catalog_name", "segment_unrecognized", "list_segment", "not_list", "list_split"} <= rules, rules
    assert _FALLBACK_STATUSES <= statuses, statuses


_TWO_TURNS = ("run_on_bare", "run_on_extra_words", "one_name_covers_igg", "phrase_smear_pharynx")


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
@pytest.mark.parametrize("name", _TWO_TURNS)
def test_follow_up_after_the_fallback_answer_stays_on_what_was_shown(name, follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    """Короткий вопрос после ответа запасного пути («а сколько?», «да», цифра): цены — только строк, которые
    пациент уже видел, услуга сессии — из ответа; первый ход — как в эталоне."""

    first, second = _dialog(name, "prod", [CASES[name].text, follow_up], monkeypatch, mini_prod_world)
    expected = _load()[f"{name}-prod"][0]
    assert first["text"] == expected["text"] and first["tu_rule"] == expected["tu_rule"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
