"""BUG-2026-09-09-CODE-FAMILY (витаминная часть): «витамин б6» отдавал цену B12.

Корень: `_VITAMIN_CODE_MAP` собран из ВИЗУАЛЬНЫХ гомоглифов (а/a, в/b, с/c,
д/d, е/e, к/k) — букв, которые в кириллице и латинице выглядят одинаково.
Но «б» латинскую «B» не напоминает: пациент пишет её ФОНЕТИЧЕСКИ, по звуку.
Карта покрывала не ту ось, поэтому «витамин в6» находился, а «витамин б6» —
нет, и запрос садился на первую строку семейства.

Цена ошибки: B6 стоит 2240 руб., B12 — 760 руб. Занижение втрое, причём
пациент узнаёт настоящую сумму на кассе.

Инвариант (класс): обе записи одного витаминного кода — латиницей, визуальным
гомоглифом и фонетической кириллицей — обязаны давать ОДНУ услугу. Проверяется
свипом по всем витаминным строкам живого каталога, а не списком витаминов.
"""

from __future__ import annotations

import re

import pytest

from agent_logic_2.nayka_api import api_price
from messengers_router.services._prices_helpers import (
    SAMARA_PRICE_REGION_ID,
    _extract_vitamin_designator,
    _select_patient_price_rows,
    resolve_price_service_name_from_catalog,
)

ROWS = [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def _vitamin_rows() -> list[str]:
    """Названия строк каталога вида «Витамин <код>» — источник кейсов."""
    out = []
    for row in ROWS:
        name = str(row.get("serviceName") or "")
        if re.match(r"^\s*витамин\w*\s+[a-zа-я]\d{0,2}\b", name, re.I):
            out.append(name)
    return out


def test_catalog_has_vitamin_family_to_test_against():
    """Гард окружения: свип имеет смысл только на непустом семействе."""
    assert len(_vitamin_rows()) >= 3, _vitamin_rows()


@pytest.mark.parametrize("cyr,lat", [("б", "b"), ("в", "b"), ("а", "a"), ("д", "d")])
def test_designator_folds_cyrillic_spelling_to_latin(cyr: str, lat: str):
    """Кириллическая запись кода витамина сворачивается к латинской."""
    assert _extract_vitamin_designator(f"витамин {cyr}6") == f"{lat}6"


def _rows_for_code(code: str) -> list[str]:
    """Названия каталога, попадающие под один витаминный код."""
    return [
        str(r.get("serviceName") or "")
        for r in ROWS
        if _extract_vitamin_designator(str(r.get("serviceName") or "")) == code
    ]


def test_phonetic_and_homoglyph_spellings_resolve_identically():
    """Класс-инвариант: «витамин б6» и «витамин в6» — одна и та же услуга.

    Свип по витаминным строкам каталога: для каждого ОДНОЗНАЧНОГО кода (одна
    услуга в прайсе) спрашиваем обеими кириллическими записями. Ответы обязаны
    совпасть. Коды, за которыми в каталоге стоит несколько РАЗНЫХ услуг, здесь
    не проверяются — там вопрос не в написании, см. xfail ниже.
    """
    mismatches = []
    for name in _vitamin_rows():
        code = _extract_vitamin_designator(name)
        if not code or not code.startswith("b"):
            continue
        if len({n.lower() for n in _rows_for_code(code)}) > 1:
            continue
        digits = code[1:]
        phonetic = resolve_price_service_name_from_catalog(f"витамин б{digits}", rows=ROWS)
        homoglyph = resolve_price_service_name_from_catalog(f"витамин в{digits}", rows=ROWS)
        if phonetic != homoglyph:
            mismatches.append((f"б{digits}", phonetic, f"в{digits}", homoglyph))
    assert not mismatches, mismatches


def test_unambiguous_vitamin_codes_are_actually_covered():
    """Гард: свип выше не должен выродиться в пустой прогон."""
    codes = {
        _extract_vitamin_designator(n)
        for n in _vitamin_rows()
        if _extract_vitamin_designator(n).startswith("b")
    }
    unambiguous = [c for c in codes if len({n.lower() for n in _rows_for_code(c)}) == 1]
    assert len(unambiguous) >= 5, sorted(codes)


def test_no_spelling_leaves_the_patient_without_an_answer():
    """Ни одно написание кода не должно приводить к ПУСТОМУ ответу.

    До фикса «витамин б12» резолвился в None: токен «б12» в каталоге не
    встречается, и гард неудовлетворимого уточнения гасил матч — ветка
    `price_info` обрывалась, пациент не получал ничего.
    """
    for spelling in ("витамин б12", "витамин в12", "витамин b12", "витамин б6", "витамин в6"):
        assert resolve_price_service_name_from_catalog(spelling, rows=ROWS) is not None, spelling
        assert _select_patient_price_rows(ROWS, spelling, limit=5), spelling


@pytest.mark.xfail(
    strict=True,
    reason="family-mode отключён для витаминов — неоднозначный код не разводится развилкой",
)
def test_ambiguous_vitamin_code_shows_both_variants_on_every_spelling():
    """ОТКРЫТО. Под кодом b12 в каталоге ДВЕ разные услуги:

        Витамин B12                                 760 руб.
        Витамин В12, активный холотранскобаламин   1350 руб.

    Это разные исследования, и выбирать должен пациент. Сейчас набор вариантов
    зависит от написания: «в12» и «b12» показывают оба, «б12» — только один,
    потому что счёт строк расходится (у «в12» кириллический токен совпадает с
    кириллическим названием буквально).

    Чинится не орфографией, а включением family-mode для витаминов: сейчас
    `_build_price_family_payload` для них выключен намеренно
    («их пропускаем, чтобы не задвоить логику»), и настоящей развилки с
    `family_variants` у витаминов нет — то, что видно на «в12», это побочный
    tie-case `_select_patient_price_rows`, а не спроектированный форк.
    """
    expected = {n.lower() for n in _rows_for_code("b12")}
    assert len(expected) > 1, "каталог изменился — кейс потерял смысл"
    for spelling in ("витамин б12", "витамин в12", "витамин b12"):
        shown = {
            str(r.get("serviceName") or "").lower()
            for r in _select_patient_price_rows(ROWS, spelling, limit=5)
        }
        assert expected <= shown, (spelling, sorted(shown))
