"""`strong_rows` по индексу среза и память хода генератора фраз (L-15b, шаг 6: O1, O6, O2).

Профиль шага 6: 99% времени генератора фраз — `strong_rows`, 3 622 вызова на 314 ходов, каждый проходил
весь прайс с нормализацией (16–22 мс) и заново строил индекс 854 синонимов МИС; половина вызовов — повтор
в том же ходе. Ускорение не должно менять ни одного ответа: инвариант — новый `strong_rows` возвращает
ровно то же (те же строки, тот же порядок), что прежний проход по всему срезу, на всех названиях и кодах
мини-прода, на фразах с пунктуацией и обвязкой, с синонимами МИС и без.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from messengers_router import turn_parse
from messengers_router.services import _biomaterial as bio
from messengers_router.services import _strong_match as sm
from messengers_router.services import _turn_phrases as tp
from messengers_router.services._common import _normalise_catalog_text
from messengers_router.services._prices_helpers import _extract_price_service_from_query
from tests.test_price_turns_golden import FIXTURE
from tests.test_strong_match import _MIS_SYNONYMS


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in (FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl").open(encoding="utf-8")]


@pytest.fixture
def mis_synonyms(monkeypatch):
    vocab = bio._MisVocabularies()
    vocab.synonym_candidates = dict(_MIS_SYNONYMS)
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    return vocab


def _reference(phrase: str, catalog, *, pool=None, verbatim: bool = False) -> list[dict[str, Any]]:
    """Прежний `strong_rows` (до шага 6) — проход по всему срезу; эталон эквивалентности."""

    key = _normalise_catalog_text(_extract_price_service_from_query(phrase) or phrase)
    names = {key, _normalise_catalog_text(phrase)} if verbatim else {key}
    names.discard("")
    if not names:
        return []
    synonyms = set(bio.mis_synonym_candidates(phrase))
    by_code = [row for row in catalog if key.isdigit() and str(row.get("serviceHomecode") or "").strip() == key]
    out: list[dict[str, Any]] = []
    for row in [*(catalog if pool is None else pool), *by_code]:
        name = sm._name(row)
        if (_normalise_catalog_text(name) in names or name in synonyms or row in by_code) and row not in out:
            out.append(row)
    return out


def _phrases(rows) -> list[str]:
    names = sorted({sm._name(row) for row in rows if sm._name(row)})
    codes = sorted({str(row.get("serviceHomecode") or "").strip() for row in rows} - {""})
    wrapped = [f"сколько стоит {name.lower()}?" for name in names[::7]]
    bare = [name.replace("(", " ").replace(")", " ").replace(",", " ") for name in names[::5]]
    return [*names, *codes, *wrapped, *bare, *_MIS_SYNONYMS, "сколько стоит оак", "холестерин?", "", "анализ"]


@pytest.mark.parametrize("verbatim", [False, True])
@pytest.mark.parametrize("with_synonyms", [False, True])
def test_index_returns_exactly_what_the_full_scan_returned(rows, verbatim, with_synonyms, request):
    if with_synonyms:
        request.getfixturevalue("mis_synonyms")
    diffs = [
        phrase
        for phrase in _phrases(rows)
        if sm.strong_rows(phrase, rows, verbatim=verbatim) != _reference(phrase, rows, verbatim=verbatim)
    ]
    assert diffs == []


def test_pool_path_is_unchanged(rows, mis_synonyms):
    pool = rows[::3]
    for phrase in _phrases(rows)[::4]:
        assert sm.strong_rows(phrase, rows, pool=pool) == _reference(phrase, rows, pool=pool)


def test_code_rows_keep_their_catalog_position(rows):
    coded = [row for row in rows if str(row.get("serviceHomecode") or "").strip().isdigit()]
    assert coded, "в мини-проде нужны строки с кодом"
    row = coded[len(coded) // 2]
    code = str(row["serviceHomecode"]).strip()
    # Строка с названием-числом впереди среза: прежний проход отдавал строки в порядке среза.
    catalog = [*rows, {"serviceName": code, "serviceHomecode": "", "price": 1}]
    catalog.insert(0, catalog.pop())
    assert sm.strong_rows(code, catalog) == _reference(code, catalog)


def test_second_call_does_not_normalise_the_catalog_again(rows, monkeypatch):
    sm.strong_rows("Глюкоза", rows)  # индекс построен
    calls = []
    real = sm._normalise_catalog_text

    def counting(text):
        calls.append(text)
        return real(text)

    monkeypatch.setattr(sm, "_normalise_catalog_text", counting)
    sm.strong_rows("Глюкоза", rows)
    assert len(calls) <= 3, len(calls)


def test_another_catalog_with_the_same_fingerprint_gets_its_own_index(rows):
    first = [dict(row) for row in rows]
    target = next(i for i in range(1, len(first) - 1) if sm.strong_rows(sm._name(first[i]), first, verbatim=True))
    name = sm._name(first[target])
    second = [dict(row) for row in rows]
    second[target] = {**second[target], "serviceName": "Пробная услуга Х"}
    # Тот же отпечаток (число строк, первая и последняя), другое содержимое внутри.
    assert sm.strong_rows(name, second, verbatim=True) == _reference(name, second, verbatim=True)
    assert sm.strong_rows("Пробная услуга Х", second, verbatim=True) == [second[target]]


def test_catalog_appended_in_place_is_reindexed(rows):
    catalog = [dict(row) for row in rows]
    assert sm.strong_rows("Пробная услуга Х", catalog, verbatim=True) == []
    catalog.append({"serviceName": "Пробная услуга Х", "price": 10})
    assert sm.strong_rows("Пробная услуга Х", catalog, verbatim=True) == [catalog[-1]]


def test_synonym_index_is_built_once_per_vocabulary(monkeypatch):
    vocab = bio._MisVocabularies()
    vocab.synonym_candidates = dict(_MIS_SYNONYMS)
    built = []
    real = bio._significant_tokens

    def counting(text):
        built.append(text)
        return real(text)

    monkeypatch.setattr(bio, "_significant_tokens", counting)
    first = bio._synonym_index(vocab)
    n = len(built)
    assert bio._synonym_index(vocab) is first
    assert len(built) == n
    # Подменённый словарь (так делают тесты и пересборка среза) — индекс заново.
    vocab.synonym_candidates = {"ттг": ("ТТГ (TSH) тиреотропный гормон",)}
    assert list(bio._synonym_index(vocab)) == ["ттг"]
    assert bio.mis_synonym_candidates("сколько стоит ттг", vocab=vocab) == ("ТТГ (TSH) тиреотропный гормон",)


def test_phrase_rows_is_remembered_for_the_turn(rows, monkeypatch):
    calls = []
    real = tp.strong_rows

    def counting(raw, catalog, **kw):
        calls.append(raw)
        return real(raw, catalog, **kw)

    monkeypatch.setattr(tp, "strong_rows", counting)
    outside = tp.phrase_rows("Глюкоза", rows)

    async def turn():
        turn_parse.begin()
        first = tp.phrase_rows("Глюкоза", rows)
        first.clear()  # вызывающий портит свой результат — память хода не портится
        return tp.phrase_rows("Глюкоза", rows)

    again = asyncio.run(turn())
    assert again == outside
    assert calls == ["Глюкоза", "Глюкоза"]  # вне хода — без памяти; в ходе — один расчёт на два вызова


def test_strong_basket_position_does_not_rank_the_catalog(rows, monkeypatch):
    """O5: у сильной позиции разбора кандидаты не считаются — `_pick` их не читает."""

    from messengers_router.services import _basket_select as bs

    ranked = []
    # Кандидаты позиции считает `_score_price_rows` (путь `_rank_price_rows` снят вместе с `basket_scored_candidates`, 22.10).
    monkeypatch.setattr(bs, "_score_price_rows", lambda *a, **k: ranked.append(a[1]) or [])
    glucose = sm.strong_rows("Глюкоза", rows, verbatim=True)
    assert glucose
    items = [{"service_name": "Глюкоза", "fragment": "Глюкоза", "strong": glucose}]
    chosen, weak, failed = asyncio.run(bs._pick("сколько стоит глюкоза", items, rows))
    assert chosen == {0: glucose} and weak == [] and not failed
    assert ranked == []


def test_resolver_and_row_selection_are_remembered_for_the_turn(rows, monkeypatch):
    """O4: корзина из фраз и подбор анализов берут у правил понимания уже разрешённые позиции."""

    from messengers_router.services import _prices_helpers as ph

    resolved, selected = [], []
    real_resolve, real_select = ph.resolve_price_service_name_from_catalog, ph._select_patient_price_rows_uncached
    monkeypatch.setattr(ph, "resolve_price_service_name_from_catalog", lambda q, **k: resolved.append(q) or real_resolve(q, **k))
    monkeypatch.setattr(ph, "_select_patient_price_rows_uncached", lambda r, q, **k: selected.append(q) or real_select(r, q, **k))
    outside = (ph._resolve_fragment_service("глюкоза", rows), ph._select_patient_price_rows(rows, "Глюкоза", limit=2))

    async def turn():
        turn_parse.begin()
        first = (ph._resolve_fragment_service("глюкоза", rows), ph._select_patient_price_rows(rows, "Глюкоза", limit=2))
        first[1].clear()
        return ph._resolve_fragment_service("глюкоза", rows), ph._select_patient_price_rows(rows, "Глюкоза", limit=2)

    again = asyncio.run(turn())
    assert again == outside and again[1]
    assert (resolved, selected) == (["глюкоза"] * 2, ["Глюкоза"] * 2)  # вне хода раз и в ходе раз на два вызова
    assert ph._select_patient_price_rows(rows, "Глюкоза", limit=3) == real_select(rows, "Глюкоза", limit=3)
