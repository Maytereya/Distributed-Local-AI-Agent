"""Словарь токенов цены сводит формы ОДНОГО слова, а не два анализа (BUG-2026-10-01-VLDL-AS-LDL).

## Что произошло

«сколько стоит лпонп» → «Холестерол - ЛПНП». В прайсе есть своя строка «Липопротеиды очень
низкой плотности ЛПОНП». ЛПОНП и ЛПНП — разные анализы (липопротеиды очень низкой и низкой
плотности), но `_PRICE_QUERY_CANONICAL_TOKENS` с апреля 2026 сводил «лпонп» к «лпнп», и
строка ЛПНП набирала больше очков.

## Инвариант

Синонимы услуг придумывает клиника (CLAUDE.md). Словарь в коде — только формы одного слова
(«общего» → «общий», «алт» → «алат»). Если клиника называет две РАЗНЫЕ строки прайса двумя
аббревиатурами, словарь не вправе свести одну к другой. Проверка — по живому прайсу.
"""

from __future__ import annotations

import re
from collections import defaultdict

from agent_logic_2.nayka_api import api_price
from messengers_router.russian_nlu import normalize_ru
from messengers_router.services._prices_helpers import (
    _PRICE_QUERY_CANONICAL_TOKENS,
    SAMARA_PRICE_REGION_ID,
    resolve_price_service_name_from_catalog,
)


def _rows():
    return [r for r in api_price.load_price_by_region(SAMARA_PRICE_REGION_ID) if isinstance(r, dict)]


def test_token_map_does_not_merge_acronyms_of_different_catalog_rows():
    rows = _rows()
    assert len(rows) > 1000, "нужен живой прайс региона"
    rows_by_acronym: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        name = str(row.get("serviceName") or "")
        for acronym in re.findall(r"\b[А-ЯЁA-Z]{3,6}\b", name):
            rows_by_acronym[normalize_ru(acronym)].add(name)

    merged = []
    for source, target in _PRICE_QUERY_CANONICAL_TOKENS.items():
        if source == target:
            continue
        source_rows, target_rows = rows_by_acronym.get(source), rows_by_acronym.get(target)
        if source_rows and target_rows and not source_rows & target_rows:
            merged.append((source, target, sorted(source_rows)[:1], sorted(target_rows)[:1]))
    assert not merged, f"словарь сводит разные строки прайса: {merged}"


def test_vldl_and_ldl_are_different_rows():
    rows = _rows()
    vldl = resolve_price_service_name_from_catalog("лпонп", rows=rows)
    ldl = resolve_price_service_name_from_catalog("лпнп", rows=rows)
    assert vldl and "лпонп" in normalize_ru(vldl), vldl
    assert ldl and "лпнп" in normalize_ru(ldl) and "лпонп" not in normalize_ru(ldl), ldl
