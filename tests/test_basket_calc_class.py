"""Корзина-калькулятор (ТЗ 17.08 §6, базовый объём; ответы заказчика 16.09) — инварианты класса под `basket_calc = on`.

STEP-2026-10-25-BASKET-CALC. Под позициями корзины — итог и взятие биоматериала; путь одной услуги не меняется.

Инварианты (на всём прайсе мини-прода, `tests/fixtures/mini_prod`, а не на трёх примерах):
- итог — сумма напечатанных позиций: у каждой одна цена — одна сумма; у позиции развилка вариантов — «от суммы меньших до
  суммы больших», одна сумма не выдумывается (решение владельца 22.07). Проверяет независимый детектор стенда
  (`evalbench.prices`): итог, который не сходится с позициями, он считает «ценой без источника»;
- взятие — по данным: «капиллярная» в названии строки → «Взятие крови из пальца»; материалы МИС, которые берут в клинике, —
  кровь из вены → «Взятие крови из вены», соскоб → «Взятие мазка (соскоба)» (материалы без взятия — моча, пунктат — его не
  отменяют); материалов со взятием нет (моча, кал) — взятия нет; у ДОПУСТИМЫХ материалов разное взятие — оговорка
  заказчика без догадки; «Взятие крови без последующего исследования» — никогда; цена взятия — цена его строки прайса;
- базовый вариант позиции — без cito и капиллярных, если пациент их не просил; вариантов после этого больше одного —
  развилка (итог «от … до …»), а не «самый дорогой».

Материалы МИС — выдержка `biomatNames` для строк мини-прода (`tests/fixtures/mini_prod/biomat_names_20261003.jsonl`, срез
справочника 03.10): гейт герметичен, справочника в облаке нет. Строки «Взятие мазка (соскоба)» в мини-проде нет — тест
дописывает её из прайса 08.10.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from evalbench.prices import PriceCatalog
from messengers_router import renderer
from messengers_router.services import _basket_select as B
from messengers_router.services import _biomaterial as bio
from messengers_router.services._prices_helpers import multi_price_payload_from_items
from messengers_router.text_templates import BASKET_DISCOUNT_NOTE, BASKET_TAKE_NOTE

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mini_prod"
_PRICE = _FIXTURES / "snapshots" / "price_by_region" / "price_region_3_20261008.jsonl"
_BIOMAT = _FIXTURES / "biomat_names_20261003.jsonl"
_SWAB_ROW = {"serviceId": 5698, "priceUnitId": 136, "serviceHomecode": "922", "serviceName": "Взятие мазка (соскоба)", "deadline": " ", "cost": 280.0}
_TAKE_NAMES = {B._VEIN, B._FINGER, B._SWAB}
_NO_TAKE = "Взятие крови без последующего исследования"


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return [*_read(_PRICE), dict(_SWAB_ROW)]


@pytest.fixture
def materials() -> dict[str, list[str]]:
    """Материалы МИС строки как их пишет клиника (по названию)."""

    out: dict[str, list[str]] = {}
    for row in _read(_BIOMAT):
        raw = row["biomatNames"]
        values = raw if isinstance(raw, list) else [raw]
        out[row["serviceName"]] = [" ".join(part.split()).lower() for value in values for part in str(value or "").split(",") if part.strip()]
    return out


@pytest.fixture(autouse=True)
def _world(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    vocab = bio._build_vocabularies(_read(_BIOMAT))
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    monkeypatch.setattr(cfg, "MR_BASKET_CALC", "on", raising=False)


def _position(label: str, prices: list[dict[str, Any]]) -> dict[str, Any]:
    return {"service_name": label, "prices": prices, "fragment": label}


def _answer(positions: list[dict[str, Any]], rows: list[dict[str, Any]], unrecognized: tuple[str, ...] = ()) -> tuple[str, dict[str, Any]]:
    calc = B.basket_calculation(positions, rows)
    payload = multi_price_payload_from_items(positions, list(unrecognized), allow_single=True, calculation=calc)
    assert payload is not None
    return renderer._format_price_family_variants(payload), payload


def _take_lines(text: str) -> list[str]:
    lines = text.splitlines()
    if f"{BASKET_TAKE_NOTE}:" not in lines:
        return []
    start = lines.index(f"{BASKET_TAKE_NOTE}:") + 1
    out = []
    for line in lines[start:]:
        if not line.startswith("— "):
            break
        out.append(line)
    return out


def _taken(text: str) -> set[str]:
    return {line[2:].rsplit(" — ", 1)[0] for line in _take_lines(text)}


def _cost(rows: list[dict[str, Any]], name: str) -> int:
    return next(int(round(float(row["cost"]))) for row in rows if row["serviceName"] == name)


def _priced(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["serviceName"] not in _TAKE_NAMES | {_NO_TAKE}]


# --- итог ---------------------------------------------------------------------------------------------------------------------


def test_total_is_the_sum_of_printed_positions_on_whole_price(rows):
    """Каждые 5 строк прайса — корзина из трёх позиций (одна строка, развилка из двух, одна строка): итог сходится с
    напечатанным по независимому детектору стенда; «от … до …» — тогда и только тогда, когда у развилки разные цены."""

    catalog = PriceCatalog(rows)
    priced = _priced(rows)
    checked = ranges = 0
    for i in range(0, len(priced) - 4, 5):
        fork = [priced[i + 1], priced[i + 2]]
        positions = [_position(f"а{i}", [priced[i]]), _position(f"б{i}", fork), _position(f"в{i}", [priced[i + 3]])]
        text, payload = _answer(positions, rows)
        totals = [line for line in text.splitlines() if line.startswith("Итого")]
        if any(float(row["cost"]) <= 0 for row in (priced[i], *fork, priced[i + 3])):
            assert totals == [], text  # цена не указана — итога нет (`test_no_total_when_a_shown_price_is_not_given`)
            continue
        assert len(totals) == 1, text
        statuses = [pair.status for pair in catalog.pairs(text)]
        assert "total" in statuses and not {"unsourced", "unattributed"} & set(statuses), text
        costs = {int(round(float(row["cost"]))) for row in fork}
        is_range = len(costs) > 1
        assert totals[0].startswith("Итого по прайсу: от ") == is_range, text
        assert BASKET_DISCOUNT_NOTE in text.splitlines()
        assert (f"«б{i}»" in totals[0]) == is_range  # развилку называет та позиция, что её даёт
        calc = payload["basket_calc"]
        low = sum(int(round(float(row["cost"]))) for row in (priced[i], priced[i + 3])) + min(costs)
        assert (calc["total_min"], calc["total_max"]) == (low, low - min(costs) + max(costs))
        checked += 1
        ranges += is_range
    assert checked >= 70 and ranges >= 30


def test_total_counts_only_found_prices_and_says_so(rows):
    """Нераспознанные позиции в итог не входят, и итог это говорит; позиция с одной ценой — без итога (нечего складывать)."""

    priced = _priced(rows)
    text, _ = _answer([_position("а", [priced[0]]), _position("б", [priced[1]])], rows, unrecognized=("флороценоз",))
    assert any(line.startswith("Итого по прайсу за найденные позиции: ") for line in text.splitlines()), text
    single, payload = _answer([_position("а", [priced[0]])], rows, unrecognized=("флороценоз",))
    assert "Итого" not in single and payload["basket_calc"]["positions"] == 1


def test_no_total_when_a_shown_price_is_not_given(rows):
    """0 в прайсе — цена не указана: с ней сумма неизвестна, итог не печатается ни для одной строки, ни для развилки
    (g0345 стенда кнопок: «от 2 800 руб.» из варианта «аборт — 0 руб.»); перечень взятия остаётся."""

    priced = _priced(rows)
    ferritin = next(row for row in rows if row["serviceName"] == "Ферритин")
    for i in range(0, len(priced) - 1, 7):
        free = dict(priced[i], cost=0.0)
        for positions in ([_position("а", [free]), _position("б", [ferritin])], [_position("а", [free, priced[i + 1]]), _position("б", [ferritin])]):
            text, payload = _answer(positions, rows)
            assert "Итого" not in text and BASKET_DISCOUNT_NOTE not in text, text
            assert payload["basket_calc"]["total_unknown"] is True
            assert _taken(text) >= {B._VEIN} or f"{BASKET_TAKE_NOTE}." in text.splitlines()  # ферритин — из вены


def test_capillary_asked_but_venous_rows_shown_is_not_a_vein_take(rows):
    """Пациент просил «из пальца», а LLM показала строки без «капиллярная»: «Взятие крови из вены» спорило бы с его словами
    (g0363) — взятие из данных не выводится, только оговорка; капиллярная строка по просьбе — из пальца."""

    oak = next(row for row in rows if row["serviceName"] == "Общий анализ крови (Le, Er, Hb, СОЭ)")
    oak_cap = next(row for row in rows if row["serviceName"] == "Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная кровь")
    urine = next(row for row in rows if row["serviceName"] == "Общий анализ мочи")
    asked = {"service_name": "ОАК из пальца", "fragment": "ОАК ребенку из пальца", "prices": [oak]}
    text, _ = _answer([asked, _position("ОАМ", [urine])], rows)
    assert _taken(text) == set() and f"{BASKET_TAKE_NOTE}." in text.splitlines(), text
    text, _ = _answer([{**asked, "prices": [oak_cap]}, _position("ОАМ", [urine])], rows)
    assert _taken(text) == {B._FINGER}


def test_positions_with_one_heading_are_summed_as_printed(rows):
    """Две позиции с одним названием рендер печатает одной группой — итог по ней же (детектор сходится)."""

    priced = _priced(rows)
    a, b = priced[10], priced[11]
    text, payload = _answer([_position("X", [a]), _position("X", [b]), _position("Y", [priced[12]])], rows)
    statuses = [pair.status for pair in PriceCatalog(rows).pairs(text)]
    assert "total" in statuses and "unsourced" not in statuses, text
    assert payload["basket_calc"]["positions"] == 2


# --- взятие биоматериала ----------------------------------------------------------------------------------------------


def _neutral(rows: list[dict[str, Any]], materials: dict[str, list[str]]) -> dict[str, Any]:
    """Строка без материалов МИС и без «капиллярная» — вторая позиция корзины, взятия не добавляет."""

    return next(row for row in _priced(rows) if row["serviceName"] not in materials and "капилляр" not in row["serviceName"].lower())


def test_takes_follow_mis_materials_on_whole_price(rows, materials):
    """Каждая строка прайса — корзина «строка + позиция без материалов»: взятие выводится из данных строки."""

    neutral = _neutral(rows, materials)
    seen = {"finger": 0, "vein": 0, "none": 0, "swab": 0, "note": 0}
    for row in _priced(rows):
        if row is neutral:
            continue
        text, _ = _answer([_position("а", [row]), _position("б", [neutral])], rows)
        assert _NO_TAKE not in text
        taken = _taken(text)
        for line in _take_lines(text):  # цена взятия — цена его строки прайса
            name, amount = line[2:].rsplit(" — ", 1)
            assert amount == f"{_cost(rows, name):,}".replace(",", " ") + " руб."
        # Материалы, которые берут в клинике (кровь, соскоб, мазок); прочие (моча, кал, пунктат) взятия не требуют.
        taken_here = [m for m in materials.get(row["serviceName"], []) if re.search(r"кров|соскоб|мазок", m)]
        if "капиллярная" in row["serviceName"].lower():
            assert taken == {B._FINGER}, row["serviceName"]
            seen["finger"] += 1
        elif not taken_here:
            assert taken == set() and BASKET_TAKE_NOTE not in text, row["serviceName"]  # моча, кал, без справочника
            seen["none"] += 1
        elif all(m == "кровь из вены" for m in taken_here):
            assert taken == {B._VEIN}, row["serviceName"]
            seen["vein"] += 1
        elif all(m.startswith("соскоб") for m in taken_here):
            assert taken == {B._SWAB}, row["serviceName"]
            seen["swab"] += 1
        else:
            # Взятие у допустимых материалов разное («кровь из вены, соскоб») — не молчим и не гадаем: перечень или оговорка.
            assert taken or f"{BASKET_TAKE_NOTE}." in text.splitlines(), row["serviceName"]
            seen["note"] += 1
    assert seen["finger"] >= 3 and seen["vein"] >= 150 and seen["swab"] >= 30 and seen["none"] >= 100 and seen["note"] >= 2, seen


def test_oak_with_both_materials_takes_from_vein(rows):
    """У ОАК в МИС оба допустимых материала («кровь из вены, кровь капиллярная»): строка без «капиллярная» — из вены
    (ответ заказчика 16.09), капиллярный вариант — из пальца."""

    oak = next(row for row in rows if row["serviceName"] == "Общий анализ крови (Le, Er, Hb, СОЭ)")
    oak_cap = next(row for row in rows if row["serviceName"] == "Общий анализ крови (Le, Er, Hb, СОЭ) капиллярная кровь")
    glucose = next(row for row in rows if row["serviceName"] == "Глюкоза")
    assert _taken(_answer([_position("оак", [oak]), _position("глюкоза", [glucose])], rows)[0]) == {B._VEIN}
    both = _answer([_position("оак из пальца", [oak_cap]), _position("глюкоза", [glucose])], rows)[0]
    assert [line[2:].rsplit(" — ", 1)[0] for line in _take_lines(both)] == [B._VEIN, B._FINGER]  # порядок перечня


def test_take_prices_are_read_from_the_price(rows):
    """Цена взятия — из прайса: поменялась строка — поменялся ответ; строки взятия нет — перечень без неё, оговорка остаётся."""

    glucose = next(row for row in rows if row["serviceName"] == "Глюкоза")
    ferritin = next(row for row in rows if row["serviceName"] == "Ферритин")
    changed = [dict(row, cost=999.0) if row["serviceName"] == B._VEIN else row for row in rows]
    text, _ = _answer([_position("глюкоза", [glucose]), _position("ферритин", [ferritin])], changed)
    assert _take_lines(text) == [f"— {B._VEIN} — 999 руб."]
    missing = [row for row in rows if row["serviceName"] != B._VEIN]
    text, payload = _answer([_position("глюкоза", [glucose]), _position("ферритин", [ferritin])], missing)
    assert _take_lines(text) == [] and f"{BASKET_TAKE_NOTE}." in text.splitlines()
    assert payload["basket_calc"]["take_unknown"] is True


def test_named_take_is_a_position_not_a_disclaimer_line(rows):
    """Взятие, которое пациент назвал сам, — позиция корзины (в итоге), второй раз под итогом не печатается."""

    vein = next(row for row in rows if row["serviceName"] == B._VEIN)
    ferritin = next(row for row in rows if row["serviceName"] == "Ферритин")
    text, payload = _answer([_position(B._VEIN, [vein]), _position("Ферритин", [ferritin])], rows)
    assert _take_lines(text) == [] and BASKET_TAKE_NOTE not in text
    assert f"1. {B._VEIN} — {_cost(rows, B._VEIN)} руб." in text.splitlines()
    assert payload["basket_calc"]["total_min"] == _cost(rows, B._VEIN) + _cost(rows, "Ферритин")


def test_fork_takes_only_when_all_variants_agree(rows, materials):
    """Развилка: взятие — только если всем вариантам нужно одно и то же; иначе — оговорка, не догадка."""

    vein_rows = [row for row in _priced(rows) if materials.get(row["serviceName"]) == ["кровь из вены"]][:2]
    urine = next(row for row in _priced(rows) if materials.get(row["serviceName"]) == ["моча"])
    neutral = _neutral(rows, materials)
    agree = _answer([_position("а", vein_rows), _position("б", [neutral])], rows)[0]
    assert _taken(agree) == {B._VEIN}
    differ = _answer([_position("а", [vein_rows[0], urine]), _position("б", [neutral])], rows)[0]
    assert _taken(differ) == set() and f"{BASKET_TAKE_NOTE}." in differ.splitlines()


# --- базовый вариант ------------------------------------------------------------------------------------------------------


def _families(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Семейства мини-прода «базовая строка + её cito / капиллярный вариант» — по названию строки (как их пишет клиника)."""

    by_name = {row["serviceName"]: row for row in rows}
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row["serviceName"]
        base = re.sub(r"^Cito\s+", "", re.sub(r"\s+капиллярная кровь$", "", name))
        if base != name and base in by_name:
            out.setdefault(base, [by_name[base]]).append(row)
    return out


def test_base_variant_drops_cito_and_capillary_unless_asked(rows):
    families = _families(rows)
    assert len(families) >= 4  # ОАК (полный), ОАК, ОАК без формулы, и т. д.
    for base, family in families.items():
        [position], _ = B._positions([{"service_name": "анализ", "fragment": "анализ"}], {0: list(family)})
        assert [row["serviceName"] for row in position["prices"]] == [base]
        for asked, word in (("cito", "Cito"), ("из пальца", "капиллярная")):
            variants = [row for row in family if word in row["serviceName"]]
            if variants:
                [kept], _ = B._positions([{"service_name": f"анализ {asked}", "fragment": f"анализ {asked}"}], {0: list(family)})
                assert {row["serviceName"] for row in variants} <= {row["serviceName"] for row in kept["prices"]}


def test_oak_family_is_a_fork_not_the_most_expensive(rows):
    """Пять строк ОАК (ответ заказчика 16.09): cito и капиллярные — не базовый вариант; остаётся больше одного — развилка
    «от … до …», а не «самый дорогой» механически (ТЗ: это систематически завышало бы цены)."""

    oak = [row for row in rows if re.search(r"общий анализ крови|^оак ", row["serviceName"], re.I) and "гельминт" not in row["serviceName"]]
    [position], _ = B._positions([{"service_name": "оак", "fragment": "оак"}], {0: oak})
    names = [row["serviceName"] for row in position["prices"]]
    assert names and not any("Cito" in name or "капиллярная" in name for name in names)
    glucose = next(row for row in rows if row["serviceName"] == "Глюкоза")
    text, payload = _answer([position, _position("Глюкоза", [glucose])], rows)
    costs = [int(row["cost"]) for row in position["prices"]]
    assert len(set(costs)) > 1
    calc = payload["basket_calc"]
    assert (calc["total_min"], calc["total_max"]) == (min(costs) + 190, max(costs) + 190) and calc["forks"] == ["оак"]
    assert "Итого по прайсу: от " in text


# --- выключатель ---------------------------------------------------------------------------------------------------------


def test_off_leaves_the_basket_answer_as_it_was(rows, monkeypatch):
    from messengers_router.runtime_config import config as cfg

    priced = _priced(rows)
    positions = [_position("а", [priced[0]]), _position("б", [priced[1], priced[2]])]
    monkeypatch.setattr(cfg, "MR_BASKET_CALC", "off", raising=False)
    assert B.basket_calculation(positions, rows) is None
    off = renderer._format_price_family_variants(multi_price_payload_from_items(positions, [], allow_single=True))
    assert "Итого" not in off and BASKET_TAKE_NOTE not in off and BASKET_DISCOUNT_NOTE not in off
    monkeypatch.setattr(cfg, "MR_BASKET_CALC", "on", raising=False)
    on = _answer(positions, rows)[0]
    assert on.startswith(off.split("\nЕсли нужно")[0])  # позиции — те же, итог и взятие — ниже
