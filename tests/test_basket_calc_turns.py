"""Корзина-калькулятор — характеристики диалогов корзины через эндпоинт (STEP-2026-10-25-BASKET-CALC).

Ходы через обработчик эндпоинта `messenger_generate_once` с выключателями прода деплоя 17 и ценой этапа 2
(`price_by_parse = on` — набор деплоя 18), мини-прод (+ строка «Взятие мазка (соскоба)» прайса 08.10), день, сеть закрыта,
материалы МИС — выдержка `biomatNames` мини-прода. Каждый ход — в обоих положениях `basket_calc`: эталон off — ответ корзины
как есть, on — те же позиции, под ними итог и взятие биоматериала (`tests/golden/basket_calc_turns.json`; перезапись —
`GOLDEN_UPDATE=1 pytest tests/test_basket_calc_turns.py`, дифф разобрать в журнале).

LLM — оракул `test_price_stage2_turns` (разметка — ровно фразы случая, выбор строки — строки с заданным началом названия)
или «недоступна» (`down`: без LLM услугу не угадываем — «уточните», калькулятор не при чём).
Реплики — классов корзины, не вопросы эталона стенда (Гудхарт, 25.10).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _biomaterial as bio
from tests.test_basket_calc_class import _BIOMAT, _SWAB_ROW, _read
from tests.test_price_stage2_turns import PROD_D17, Case, _Down, _Oracle
from tests.test_price_turns_golden import FIXTURE, run_dialog

GOLDEN = Path(__file__).resolve().parent / "golden" / "basket_calc_turns.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
SWITCH = ("off", "on")

CASES: dict[str, Case] = {
    "oak_variants": Case(
        ("сколько стоит оак, ферритин, глюкоза",), ("оак", "ферритин", "глюкоза"),
        rows=("Общий анализ крови (Le", "Общий анализ крови (полный)"), why="развилка ОАК без cito и капиллярных — «от … до …»",
    ),
    "thyroid_by_synonyms": Case(
        ("т3, т4, ттг",), ("т3", "т4", "ттг"), rows=("ТТГ (TSH)",), why="т3 / т4 — свободные по синонимам МИС (ответ заказчика 16.09)",
    ),
    "urine_feces": Case(("общий анализ мочи, кал на яйца глист",), ("общий анализ мочи", "кал на яйца глист"), why="взятия нет"),
    "smear_and_blood": Case(
        ("мазок на степень чистоты и ферритин",), ("мазок на степень чистоты", "ферритин"), rows=("Мазок на степень чистоты влагалища",),
        why="мазок и вена",
    ),
    "capillary_asked": Case(
        ("общий анализ крови из пальца, глюкоза",), ("общий анализ крови из пальца", "глюкоза"),
        rows=("Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная",),
        why="капиллярная — по просьбе: палец и вена",
    ),
    "named_take": Case(("взятие крови из вены, ферритин",), ("взятие крови из вены", "ферритин"), why="взятие — позиция, не перечень"),
    "partial": Case(("ферритин, глюкоза, флороценоз",), ("ферритин", "глюкоза", "флороценоз"), rows=(), why="итог по найденным"),
    "llm_down": Case(("сколько стоит оак, ферритин, глюкоза",), ("оак", "ферритин", "глюкоза"), why="без LLM — «уточните»"),
}
_SYNONYMS = {"т3": ("сТ3 (FT3) Свободный трийодтиронин",), "т4": ("сТ4 (FT4) Свободный тироксин",)}  # как в справочнике МИС 03.10
_TELEMETRY = ("basket_total", "basket_takes", "basket_take_unknown", "tu_parse_status", "price_policy")


@pytest.fixture(scope="module")
def basket_world(tmp_path_factory) -> Path:
    source = tmp_path_factory.mktemp("basket_src")
    shutil.copytree(FIXTURE, source, dirs_exist_ok=True)
    price = source / "price_by_region" / "price_region_3_20261008.jsonl"
    with price.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_SWAB_ROW, ensure_ascii=False) + "\n")
    return ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("basket_calc") / "world")


def _turns(case_name: str, switch: str, world_dir: Path, monkeypatch) -> dict[str, Any]:
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D17, "MR_PRICE_BY_PARSE": "on", "MR_BASKET_CALC": switch}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    vocab = bio._build_vocabularies(_read(_BIOMAT))
    if case_name == "thyroid_by_synonyms":
        vocab.synonym_candidates = dict(_SYNONYMS)
        vocab.synonym_to_service = {key: names[0] for key, names in _SYNONYMS.items()}
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    case = CASES[case_name]
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Down() if case_name == "llm_down" else _Oracle(case))
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        records = run_dialog(list(case.turns), f"basket-calc-{case_name}-{switch}")
        telemetry = list(world.telemetry)
        attempts = list(world.net.attempts)
    assert attempts == []
    for record, row in zip(records, telemetry):
        record["telemetry"] = {key: row.get(key) for key in _TELEMETRY if row.get(key) not in (None, [], "")}
    return {"turns": records}


def _load() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case_name", sorted(CASES))
def test_turn_matches_golden(case_name, switch, basket_world, monkeypatch):
    record = json.loads(json.dumps(_turns(case_name, switch, basket_world, monkeypatch), ensure_ascii=False))
    key = f"{case_name}-{switch}"
    if UPDATE:
        data = _load()
        data[key] = record
        GOLDEN.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        return
    assert record == _load().get(key), f"эталон {key}: GOLDEN_UPDATE=1"


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_on_only_adds_the_calculation_under_the_positions(case_name):
    """Под on ответ корзины — тот же ответ off, в который вставлены только строки итога и взятия (позиции, «Не распознал»,
    подписи — как были); вне корзины ответ не меняется."""

    golden = _load()
    off = golden[f"{case_name}-off"]["turns"][-1]["text"].splitlines()
    on = golden[f"{case_name}-on"]["turns"][-1]["text"].splitlines()
    added: list[str] = []
    rest = iter(off)
    expected = next(rest, None)
    for line in on:  # строки off — подпоследовательность строк on, остальное — вставки
        if line == expected:
            expected = next(rest, None)
        else:
            added.append(line)
    assert expected is None, f"строка off потеряна: {expected!r}"
    calc = ("Итого по прайсу", "Взятие биоматериала", "— Взятие", "Скидки применяет")
    assert all(line.startswith(calc) or not line.strip() for line in added), added


def test_golden_has_no_orphans():
    assert set(_load()) == {f"{name}-{switch}" for name in CASES for switch in SWITCH}
