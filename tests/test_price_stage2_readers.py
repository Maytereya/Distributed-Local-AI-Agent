"""Цена, этап 2 «без стоп-слов» — характеристики всех читателей вырезателя правил (DESIGN-2026-10-25-PRICE-STAGE2).

Этап 2 (решение владельца 22.10, `docs/DESIGN_2026-10-23_no_stopwords.md` §3): потребители читают фразы разбора хода вместо
вырезателя `_extract_price_service_from_query` и одиннадцати списков цены. Вырезатель — не одна функция ответа о цене:
его вывод и списки играют четыре роли — фраза услуги (что ищем, что печатаем пациенту, что уходит клинике в журнал
синонимов), «названа ли услуга», ключ сильного совпадения по данным МИС (`strong_rows` — у цены, корзины, срока,
подготовки, разбора хода) и слова поиска (`_price_query_tokens`). Консилиум 25.10 (МЕГА, п. 3): характеристики — на всех
читателей, не только на ответ цены.

Первый слой (этот файл): каждый прямой читатель вырезателя, списков и производных от них множеств (`_WRAPPER`, `_FILLER`,
`_MENTION_EDGE`) на одном корпусе синтетических реплик, прайс «мини-прод» (404 строки настоящего прайса), день, сеть
закрыта, синонимы МИС герметичны. Эталон `tests/golden/price_stage2_readers.json` держит поведение как есть — вместе с
известными дефектами «букв»: намеренная правка меняет его только перезаписью с разбором диффа. Второй слой — ходы через
эндпоинт с разбором LLM (`tests/test_price_stage2_turns.py`).

Инварианты класса:
- у каждого прямого читателя (AST по `messengers_router`) — объявление: какой характеристикой он закреплён или почему
  его нет на пути прода деплоя 17; читатель без объявления и объявление без читателя — красный тест (новый читатель
  списков не появится молча, снятый — не оставит пустой характеристики);
- корпус — реплики классов, а не вопросы эталона (Гудхарт, 25.10): ни одной реплики из наборов стенда.

Перезапись эталона после НАМЕРЕННОЙ правки: `GOLDEN_UPDATE=1 pytest tests/test_price_stage2_readers.py`, затем дифф
`tests/golden/price_stage2_readers.json` — в коммит вместе с правкой и с объяснением по каждому читателю.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from evalbench import world as ev_world
from tests.test_price_turns_golden import FIXTURE, mini_prod_world  # noqa: F401 — фикстура

GOLDEN = Path(__file__).resolve().parent / "golden" / "price_stage2_readers.json"
UPDATE = os.environ.get("GOLDEN_UPDATE") == "1"
ROOT = Path(__file__).resolve().parent.parent

# Реплики классов (не вопросы эталона). Ключ — класс, значение — реплика как написал бы пациент.
CORPUS: dict[str, str] = {
    # одна услуга — название, словоформы, обвязка вопроса
    "bare_name": "ферритин",
    "bare_name_caps": "ФЕРРИТИН",
    "q_name": "сколько стоит ферритин",
    "q_analysis_on": "сколько стоит анализ на ферритин",
    "q_analyses_on": "сколько стоят анализы на ферритин",
    "q_price_of_analyses": "цена анализов на ферритин",
    "q_polite_wrap": "подскажите пожалуйста сколько стоит ттг",
    "q_pochem": "почем экг",
    "q_price_after": "экг цена",
    "q_want_to_take": "хочу сдать кровь на сахар",
    "q_can_do": "можно ли сделать кардиограмму",
    "q_unknown_word": "сколько стоит спирометрия",
    "q_long_no_hint": "добрый день хотела бы узнать сколько будет стоить сдать кровь на гормоны щитовидки",
    # сокращения, коды, название целиком
    "abbr_oak": "сколько стоит оак",
    "abbr_oak_cito": "оак срочно",
    "abbr_two_letter_caps": "сколько стоит ЛГ",
    "abbr_two_letter_lower": "лг сколько",
    "code_homecode": "240",
    "full_name_brackets": "Общий анализ крови (Le, Er, Hb, СОЭ)",
    "full_name_hyphen_plus": "сколько стоит Глюкозотолерантный тест (глюкоза+ C-пептид)",
    "homoglyph_code": "сколько стоит Са-125",
    "vitamin_letter": "сколько стоит витамин D",
    "family_word": "сколько стоит мазок",
    # общее слово, не услуга
    "generic_analysis": "сколько стоит анализ",
    "generic_analyses": "сколько стоят анализы",
    "generic_services": "какие у вас цены на услуги",
    # приём врача
    "consult_specialty": "сколько стоит прием уролога",
    "consult_no_specialty": "сколько стоит прием врача",
    "consult_word": "сколько стоит консультация",
    "qualifier_oms": "сколько стоит прием терапевта по ОМС",
    # УЗИ
    "uzi_generic": "сколько стоит узи",
    "uzi_compound": "сколько стоит узи почек и щитовидной железы",
    # другие свойства услуги
    "readiness_named": "когда будет готов анализ на ферритин",
    "readiness_generic": "сколько готовятся анализы",
    "prepare_named": "как подготовиться к анализу на глюкозу",
    "walkin_named": "а на ферритин нужна запись?",
    # ответы без услуги: дата, время, контекст
    "date_digits": "6.09.2026",
    "time_digits": "10:30",
    "date_words": "20 ноября в 10 утра",
    "context_question": "а сколько стоит?",
    "context_price": "а цена?",
    "yes": "да",
    "digit": "1",
    # список
    "list_commas": "ферритин, глюкоза, холестерин",
    "list_and": "ферритин и глюкоза",
    "list_run_on": "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА",
}

# Пары для читателей с контекстом: (реплика корпуса, строка/услуга контекста).
_STALE_CONTEXT = "ТТГ (TSH) тиреотропный гормон"
# Сводка записи оператору (`policies.extract_price_rub`): строка прайса в ответе и услуга записи.
OPERATOR_PAIRS: dict[str, tuple[str, str]] = {
    "consult_same": ("Прием (осмотр, консультация) врача-уролога", "прием уролога"),
    "consult_wordform": ("Прием (осмотр, консультация) врача-уролога", "уролог"),
    "consult_other": ("Прием (осмотр, консультация) врача-терапевта", "прием уролога"),
    "foreign_by_code": ("Индейка (F284)", "прием ортопеда"),
    "lab_same": ("Ферритин", "анализ на ферритин"),
    "lab_generic_only": ("Ферритин", "анализ"),
    "uzi_other_organ": ("УЗИ органов мошонки + ЦДК сосудов", "узи щитовидной железы"),
}


def _names(rows: Any, limit: int = 6) -> list[str]:
    return [str(row.get("serviceName") or row.get("name") or "") for row in list(rows or [])[:limit] if isinstance(row, dict)]


def _rows() -> list[dict[str, Any]]:
    from messengers_router.services._prices_helpers import _catalog_rows

    return [row for row in _catalog_rows(None) if isinstance(row, dict)]


def _two_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(row.get("serviceName")): row for row in rows}
    return [by_name["Ферритин"], by_name["Глюкоза"]]


def _readers() -> dict[str, Callable[[str, list[dict[str, Any]]], Any]]:
    """Читатель → функция (реплика, прайс) → значение для эталона. Имена — как в объявлениях `DECLARED`."""

    from messengers_router import turn_understanding as TU
    from messengers_router.services import _doctors_helpers as D
    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _price_select_llm as PS
    from messengers_router.services import _turn_phrases as TP
    from messengers_router.services import prepare as PR
    from messengers_router.services import prices as P
    from messengers_router.services._readiness_select import readiness_from_catalog
    from messengers_router.services._strong_match import strong_rows

    def single_mention(q: str, _rows: list) -> Any:
        return [[m.phrase, m.strength, list(m.candidates)[:3]] for m in TU._single_mention(q)]

    def understanding(q: str, _rows: list) -> Any:
        tu = TU.understand_price_turn(q)
        return {"services": [m.phrase for m in tu.services], "rule": tu.services_rule, "facet": tu.facet}

    def multi_items(q: str, rows: list) -> Any:
        items, unrecognized = H._multi_price_items(q, rows)
        return {"items": [[i.get("fragment"), i.get("service_name")] for i in items], "unrecognized": list(unrecognized)}

    def family_payload(q: str, rows: list) -> Any:
        payload = H._build_price_family_payload(q, rows)
        return _names((payload or {}).get("prices")) if isinstance(payload, dict) else payload

    def readiness(q: str, rows: list) -> Any:
        found = readiness_from_catalog(q, rows=rows)
        return None if found is None else [dict(item) for item in found]

    def strong_window(q: str, rows: list) -> Any:
        found = TP.strong_window(q, rows)
        return None if found is None else [found[0], _names(found[1]), found[2]]

    def retail_candidate(q: str, rows: list) -> Any:
        resolved = H.resolve_price_service_name_from_catalog(q, rows=rows) or ""
        return D._should_prefer_retail_query_candidate(H._extract_price_service_from_query(q) or "", resolved)

    def effective_service(q: str, rows: list) -> Any:
        query = H.resolve_price_service_name_from_catalog(q, rows=rows) or H._extract_price_service_from_query(q) or ""
        return D._select_effective_price_service_name(_STALE_CONTEXT, query)

    return {
        # вырезатель и роль «фраза услуги»
        "extractor": lambda q, _r: H._extract_price_service_from_query(q),
        "shown_phrase": lambda q, rows: P._llm_selection_service_name(q, _two_rows(rows)),
        "single_mention": single_mention,
        "bare_phrase": lambda q, _r: TU._bare_phrase(q),
        "understanding_rules": understanding,
        "catalog_queries": lambda q, _r: H._build_price_catalog_queries(q),
        "resolver": lambda q, rows: H.resolve_price_service_name_from_catalog(q, rows=rows),
        "alias_candidates": lambda q, _r: H._price_alias_candidates(q),
        "family_candidate": lambda q, rows: H._is_family_query_candidate(q, rows),
        "family_root_tokens": lambda q, _r: H._family_query_root_tokens(q),
        "family_rows": lambda q, rows: _names(H._build_family_candidate_rows(q, rows)),
        "family_payload": family_payload,
        "generic_uzi": lambda q, _r: H._is_generic_uzi_price_request(q),
        "prefer_current_over_context": lambda q, _r: H._should_prefer_current_price_query_over_context(q, _STALE_CONTEXT),
        "retail_candidate": retail_candidate,
        "effective_service": effective_service,
        # роль «названа ли услуга»
        "names_service": lambda q, _r: PS.names_service(q),
        "names_own_service": lambda q, _r: H.price_reply_names_own_service(q),
        "names_a_service": lambda q, _r: P._names_a_service([q]),
        # роль «ключ сильного совпадения»
        "strong_rows": lambda q, rows: _names(strong_rows(q, rows)),
        "phrase_rows": lambda q, rows: _names(TP.phrase_rows(q, rows)),
        "strong_window": strong_window,
        "readiness_from_catalog": readiness,
        "prepare_own_subject": lambda q, _r: PR._names_own_subject(q),
        # роль «слова поиска» и прочие списки
        "query_tokens": lambda q, _r: H._price_query_tokens(q),
        "short_name_words": lambda q, _r: PS._short_name_words(q),
        "specific_tokens": lambda q, _r: H._has_specific_price_tokens(q),
        "distinctive_tokens": lambda q, _r: sorted(H._distinctive_service_tokens(q)),
        "meaningful_tokens": lambda q, _r: D._meaningful_price_service_tokens(q),
        "noise_tokens": lambda q, _r: [t for t in H._price_query_tokens(q) if D._is_price_service_noise_token(t)],
        "trailing_noise": lambda q, _r: H._strip_trailing_price_noise(q),
        "content_tokens": lambda q, _r: H._price_query_content_tokens(q),
        "readiness_strip": lambda q, _r: list(H.strip_readiness_phrasing(q)),
        "datetime_only": lambda q, _r: H._is_datetime_only_query(q),
        # корзина правил
        "split_items": lambda q, _r: H._split_price_query_items(q),
        "multi_items": multi_items,
        "fragment_service": lambda q, rows: H._resolve_fragment_service(q, rows),
    }


def _snapshot(reader: str) -> dict[str, Any]:
    fn = _readers()[reader]
    rows = _rows()
    return {case: fn(text, rows) for case, text in CORPUS.items()}


def _operator_snapshot() -> dict[str, Any]:
    from messengers_router.policies import _price_row_matches_service

    return {case: _price_row_matches_service(row, service) for case, (row, service) in OPERATOR_PAIRS.items()}


def _load() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}


def _store(section: str, value: dict[str, Any]) -> None:
    data = _load()
    data[section] = value
    GOLDEN.write_text(json.dumps(dict(sorted(data.items())), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _check(section: str, record: dict[str, Any]) -> None:
    record = json.loads(json.dumps(record, ensure_ascii=False))  # кортежи → списки, как в эталоне
    if UPDATE:
        _store(section, record)
        return
    base = _load().get(section)
    assert base is not None, f"нет эталона для {section}: GOLDEN_UPDATE=1"
    changed = {case: {"было": base.get(case), "стало": value} for case, value in record.items() if base.get(case) != value}
    assert not changed and set(base) == set(record), json.dumps(changed, ensure_ascii=False, indent=1)


@pytest.mark.parametrize("reader", sorted(_readers()))
def test_reader_matches_golden(reader, mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = _snapshot(reader)
        attempts = list(world.net.attempts)
    assert attempts == []
    _check(reader, record)


def test_operator_summary_price_matches_golden(mini_prod_world):  # noqa: F811
    """Цена в сводке записи оператору: строка прайса из ответа должна относиться к услуге записи (`extract_price_rub`)."""

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = _operator_snapshot()
    _check("operator_summary_price", record)


def test_golden_has_no_orphans():
    assert set(_load()) == {*_readers(), "operator_summary_price"}


# --------------------------------------------------------------------------- объявления читателей

# Что читают: вырезатель, ключ сильного совпадения, одиннадцать списков цены (дизайн 23.10, §1) и производные множества.
_SOURCES = frozenset(
    {
        "_extract_price_service_from_query",
        "strong_rows",
        "_PRICE_QUERY_STOPWORDS",
        "_PRICE_SPEECH_ACT_STOPWORDS",
        "_MULTI_PRICE_NOISE_WORDS",
        "_MULTI_PRICE_NOISE_RE",
        "_PRICE_GENERIC_SERVICE_TOKENS",
        "_PRICE_QUERY_SERVICE_NOISE_TOKENS",
        "_GENERIC_SERVICE_PREFIXES",
        "_PRICE_GENERIC_FAMILY_ROOT_TOKENS",
        "_READINESS_LEADING_FILLER",
        "_DATETIME_FILLER_RE",
        "_PRICE_MATCH_GENERIC_TOKENS",
        "_WRAPPER",
        "_FILLER",
        "_MENTION_EDGE",
    }
)

_OLD_PARSE = "вне пути прода: генератор окон разбора (`turn_spans_llm = on` с деплоя 17), уходит со снятием выключателя этапа 1"

# Прямой читатель → чем закреплён: `reader:<имя>` (этот файл), `turns:<класс>` (ходы через эндпоинт), `off:<почему>`.
DECLARED: dict[str, str] = {
    "messengers_router.policies._price_row_matches_service": "reader:operator_summary_price",
    "messengers_router.services._basket_select._word_candidates": "turns:basket",
    "messengers_router.services._basket_select._context_candidates": "turns:basket",
    "messengers_router.services._basket_select._context_candidates.words": "turns:basket",
    "messengers_router.services._basket_select._pick": "turns:basket",
    "messengers_router.services._doctors_helpers._should_prefer_retail_query_candidate": "reader:retail_candidate",
    "messengers_router.services._doctors_helpers._meaningful_price_service_tokens": "reader:meaningful_tokens",
    "messengers_router.services._doctors_helpers._is_price_service_noise_token": "reader:noise_tokens",
    "messengers_router.services._price_select_llm._short_name_words": "reader:short_name_words",
    "messengers_router.services._price_select_llm.select_price_rows": "turns:price_select",
    "messengers_router.services._prices_helpers.strip_readiness_phrasing": "reader:readiness_strip",
    "messengers_router.services._prices_helpers.price_reply_names_own_service": "reader:names_own_service",
    "messengers_router.services._prices_helpers._price_query_tokens": "reader:query_tokens",
    "messengers_router.services._prices_helpers._price_alias_candidates": "reader:alias_candidates",
    "messengers_router.services._prices_helpers._is_datetime_only_query": "reader:datetime_only",
    "messengers_router.services._prices_helpers._is_generic_uzi_price_request": "reader:generic_uzi",
    "messengers_router.services._prices_helpers._should_prefer_current_price_query_over_context": "reader:prefer_current_over_context",
    "messengers_router.services._prices_helpers._has_specific_price_tokens": "reader:specific_tokens",
    "messengers_router.services._prices_helpers._build_price_catalog_queries": "reader:catalog_queries",
    "messengers_router.services._prices_helpers._distinctive_service_tokens": "reader:distinctive_tokens",
    "messengers_router.services._prices_helpers._strip_trailing_price_noise": "reader:trailing_noise",
    "messengers_router.services._prices_helpers._family_query_root_tokens": "reader:family_root_tokens",
    "messengers_router.services._prices_helpers._build_family_candidate_rows": "reader:family_rows",
    "messengers_router.services._prices_helpers._is_family_query_candidate": "reader:family_candidate",
    "messengers_router.services._prices_helpers._build_price_family_payload": "reader:family_payload",
    "messengers_router.services._prices_helpers._split_price_query_items": "reader:split_items",
    "messengers_router.services._prices_helpers._is_multi_price_noise": "reader:content_tokens",
    "messengers_router.services._prices_helpers._resolve_fragment_service": "reader:fragment_service",
    "messengers_router.services._readiness_select.readiness_from_catalog": "reader:readiness_from_catalog",
    "messengers_router.services._turn_phrases.phrase_rows": "reader:phrase_rows",
    "messengers_router.services._turn_phrases.strong_window": "reader:strong_window",
    "messengers_router.services._turn_phrases.bare": _OLD_PARSE,
    "messengers_router.services._turn_phrases._phrase": _OLD_PARSE,
    "messengers_router.services._turn_phrases._whole": _OLD_PARSE,
    "messengers_router.services._turn_phrases._strong_window_pieces": _OLD_PARSE,
    "messengers_router.services._turn_phrases._strong_window_pieces.flush": _OLD_PARSE,
    "messengers_router.services._turn_phrases._variant": _OLD_PARSE,
    "messengers_router.services._turn_phrases._with_beside": _OLD_PARSE,
    "messengers_router.services._turn_phrases._only_wrapper_outside": _OLD_PARSE,
    "messengers_router.services.prepare._names_own_subject": "reader:prepare_own_subject",
    "messengers_router.services.prices._match_policy_payload": "reader:shown_phrase",
    "messengers_router.services.prices._names_a_service": "reader:names_a_service",
    "messengers_router.services.prices._llm_selection_service_name": "reader:shown_phrase",
    "messengers_router.services.prices.price_info": "turns:price_one",
    "messengers_router.services.prices.service_bundle_info": "turns:price_one",
    "messengers_router.turn_understanding._bare_phrase": "reader:bare_phrase",
    "messengers_router.turn_understanding._single_mention": "reader:single_mention",
}


def _direct_readers() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted((ROOT / "messengers_router").rglob("*.py")):
        module = path.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")

        def visit(node: ast.AST, prefix: str) -> None:
            for child in ast.iter_child_nodes(node):
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                name = f"{prefix}{child.name}"
                if not isinstance(child, ast.ClassDef) and child.name not in _SOURCES:
                    own = {n.id for n in ast.walk(child) if isinstance(n, ast.Name)} | {
                        n.attr for n in ast.walk(child) if isinstance(n, ast.Attribute)
                    }
                    if own & _SOURCES:
                        out[f"{module}.{name}"] = sorted(own & _SOURCES)
                visit(child, f"{name}.")

        visit(ast.parse(path.read_text(encoding="utf-8")), "")
    return out


def test_every_direct_reader_is_declared():
    found = _direct_readers()
    assert not set(found) - set(DECLARED), {name: found[name] for name in sorted(set(found) - set(DECLARED))}
    assert not set(DECLARED) - set(found), sorted(set(DECLARED) - set(found))


def test_declarations_point_at_existing_characterizations():
    from tests import test_price_stage2_turns as turns

    readers = {*_readers(), "operator_summary_price"}
    for name, where in DECLARED.items():
        kind, _, target = where.partition(":")
        if kind == "reader":
            assert target in readers, (name, where)
        elif kind == "turns":
            assert target in turns.CLASSES, (name, where)
        else:
            assert where.startswith("вне пути прода"), (name, where)
