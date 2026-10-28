"""Сильное совпадение по данным МИС — одно определение для цены, корзины, срока и L-15b (шаг 3).

Решение владельца 13.10 (вариант А): сильное — только данные клиники: дословное название строки прайса,
синоним справочника МИС (все его услуги), код услуги. «Название до скобок» и курируемый словарь кода
`_PRICE_SERVICE_ALIASES` — слабое: строку выбирает LLM (`choose()`), а словарь — клинике как предложения
в справочник синонимов.

Инварианты класса (принципы 1–2):
- сильное = данные МИС: дословное название (без учёта пунктуации), синоним МИС — все его услуги
  (у неоднозначного — несколько строк), код услуги; аббревиатура словаря кода и название до скобок —
  не сильное;
- цена и корзина считают сильное тем же определением, что и до шага: их эталоны
  (`test_price_turns_golden.py`, `test_price_match_policy.py`, `test_basket_select.py`) не правились;
- срок без выбора LLM печатается только по сильному совпадению по данным МИС (одна услуга), иначе — услугу
  выбирает LLM и срок печатает код, а LLM недоступна — честный перевод. Прежнее определение срока
  (`strong_match = legacy`: название до скобок, словарь кода) снято 16.10 (T7 шаг 1, решение владельца).
"""

from __future__ import annotations

import json

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import FIXTURE, run_dialog
from tests.test_readiness_select import _ReadinessLLM

# Синонимы как в живом справочнике клиники (срез 20261003); гейт держит их пустыми
# (`_hermetic_mis_synonyms`), поэтому словарь задан явно.
_MIS_SYNONYMS = {
    "холестерин": ("Холестерол - ЛПНП", "Холестерол", "Холестерол - ЛПВП", "Триглицериды"),
    "оак": (
        "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты)",
        "Общий анализ крови (Le, Er, Hb, СОЭ)",
        "Общий анализ крови (полный)(СОЭ,Le,Er,Hb,L-формула, тромбоциты, эритроциты) капиллярная кровь",
    ),
    "фгдс": ("Эзофагогастродуоденоскопия (ФГДС)",),
    "сахар": ("Глюкоза",),
}
_TSH = "ТТГ (TSH) тиреотропный гормон"
_OAK_LE = "Общий анализ крови (Le, Er, Hb, СОЭ)"
_DEPLOY = {
    "MR_TURN_UNDERSTANDING_PRICE": "on",
}


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in (FIXTURE / "price_by_region" / "price_region_3_20261008.jsonl").open(encoding="utf-8")]


@pytest.fixture
def mis_synonyms(monkeypatch):
    from messengers_router.services import _biomaterial as _bio

    hermetic = _bio._vocabularies

    def _with_synonyms():
        vocab = hermetic()
        vocab.synonym_candidates = dict(_MIS_SYNONYMS)
        vocab.synonym_to_service = {}
        return vocab

    monkeypatch.setattr(_bio, "_vocabularies", _with_synonyms)


def _names(found) -> list[str]:
    return sorted({str(row["serviceName"]) for row in found})


# --- (а) одно определение: только данные МИС


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("ферритин", ["Ферритин"]),  # дословное название
        ("ферритин?", ["Ферритин"]),  # пунктуация не мешает
        # Слова вопроса — не услуга: под `price_by_parse = on` (прод с 26.10, Ш1) ключ — фраза как есть, без вырезателя правил,
        # и реплика сильна, только если она сама — название (решение 22.10 (а)); фразу услуги из вопроса выделяет разбор хода.
        # До 27.10 тут было ["Ферритин"] (путь off, вырезатель). Пациент без LLM видит «Уточните название» — нейтрально (политика 27.10).
        ("сколько стоит ферритин?", []),
        ("общий анализ крови le er hb соэ", [_OAK_LE]),  # название со скобками и запятыми, набранное без них
        ("липидограмма", ["Липидограмма"]),  # дословное название сильное, хотя слово есть и в словаре кода
        ("фгдс", ["Эзофагогастродуоденоскопия (ФГДС)"]),  # синоним МИС
        ("холестерин", ["Триглицериды", "Холестерол", "Холестерол - ЛПНП"]),  # синоним МИС — все его услуги в прайсе
        ("оак", sorted(_MIS_SYNONYMS["оак"])),  # синоним МИС — три варианта ОАК
        ("640", ["Глюкоза"]),  # код услуги
        ("ттг", []),  # название до скобок «ТТГ (TSH) …» — не сильное
        ("общий анализ крови", []),  # словарь кода — не сильное
        ("витамин д", []),  # синонима в словаре теста нет — только LLM
    ],
)
def test_strong_is_mis_data_only(phrase, expected, rows, mis_synonyms):
    from messengers_router.services._strong_match import strong_rows

    found = strong_rows(phrase, rows)
    assert _names(found) == expected
    assert "Липидограмма" not in _names(found) or phrase == "липидограмма"
    assert "ОАК (без лейкоцитарной формулы и СОЭ)" not in _names(found)


def test_code_alias_and_name_before_brackets_are_weak_without_mis_synonyms(rows):
    """Словарь кода («холестерин» → «Липидограмма», «оак» → ОАК) сильным не делает: синонимов нет — пусто."""

    from messengers_router.services._strong_match import strong_rows

    assert [strong_rows(phrase, rows) for phrase in ("холестерин", "оак", "оам", "ттг", "лпнп")] == [[]] * 5


def test_pool_limits_name_and_synonym_but_not_the_code(rows, mis_synonyms):
    from messengers_router.services._strong_match import strong_rows

    glucose = [row for row in rows if row["serviceName"] == "Глюкоза"]
    ferritin = [row for row in rows if row["serviceName"] == "Ферритин"]
    assert strong_rows("ферритин", rows, pool=glucose) == []
    assert strong_rows("ферритин", rows, pool=[*glucose, *ferritin]) == ferritin
    assert _names(strong_rows("сахар", rows, pool=glucose)) == ["Глюкоза"]
    assert _names(strong_rows("640", rows, pool=[])) == ["Глюкоза"]  # код ищется по всему срезу


def test_price_and_basket_use_the_one_definition():
    """Цена (L-13) и корзина (L-15a) зовут одну функцию; прежних определений в модулях нет."""

    from messengers_router.services import _basket_select, _price_select_llm, _prices_helpers, _strong_match

    assert _basket_select.strong_rows is _strong_match.strong_rows
    assert _price_select_llm.strong_rows is _strong_match.strong_rows
    assert not hasattr(_price_select_llm, "_strong_rows") and not hasattr(_prices_helpers, "_strong_service_match")


def test_one_definition_without_a_switch():
    """T7 шаг 1 (16.10): выключателя `strong_match` и прежнего определения срока нет — одно определение."""

    from agent_logic_2 import config
    from messengers_router.services import _readiness_select, _strong_match

    assert not hasattr(_strong_match, "strong_match_mode") and not hasattr(_readiness_select, "_legacy_strong_service_match")
    assert not hasattr(config, "MR_STRONG_MATCH") and not hasattr(config.settings.messenger_router, "strong_match")


# --- (в) срок: только данные МИС


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("сколько готовится ферритин", "Ферритин"),  # дословное название
        ("сколько готовится сахар", "Глюкоза"),  # синоним МИС на одну услугу
        ("сколько готовится ттг", None),  # название до скобок — не сильное
        ("сколько готовится общий анализ крови", None),  # словарь кода — не сильное
        ("сколько готовится оак", None),  # синоним МИС — три услуги, не одна
        ("сколько готовится холестерин", None),  # синоним на четыре анализа — не одна услуга
    ],
)
def test_readiness_strong_by_definition(question, expected, rows, mis_synonyms):
    from messengers_router.services._readiness_select import readiness_from_catalog

    found = readiness_from_catalog(question, rows=rows)
    assert (found[0]["name"] if found else None) == expected
    assert found is None or found[0]["deadline"] == "1–2 дня"


def _ask(question: str, session: str, world_root) -> dict:
    with ev_world.FrozenWorld(world_root, "day"):
        [turn] = run_dialog([question], session)
    return turn


def _deploy(monkeypatch, llm: _ReadinessLLM, phrases: tuple[str, ...]) -> None:
    """Набор деплоя; разметка хода — оракул с фразой услуги `phrases` (LLM разметки работает), выбор срока — `llm`.

    Разбор реплики — LLM разметки (`turn_spans_llm = on`, прод с 24.10): без неё вопрос о сроке — перевод до выбора, и тест
    не дошёл бы до своего предмета (сильное по данным / выбор модели)."""

    from messengers_router.runtime_config import config as cfg

    for key, value in _DEPLOY.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", SpansOracle(phrases, inner=llm))
    keep_llm_breaker_closed(monkeypatch)


@pytest.mark.parametrize("question", ["сколько готовится ттг", "сколько готовится оак", "сколько готовится общий анализ крови"])
def test_data_prints_no_deadline_by_letters_without_the_model(question, mini_prod_world, monkeypatch):
    """Слабое совпадение без LLM выбора — честный перевод, а не срок строки, которую угадали буквы (снятое определение
    `legacy` печатало срок без LLM: «до скобок», словарь кода)."""

    llm = _ReadinessLLM(fail=True)
    _deploy(monkeypatch, llm, (question.removeprefix("сколько готовится "),))
    turn = _ask(question, f"strong-data-{question}", mini_prod_world)
    assert (turn["text"], turn["handoff"]) == (RESULT_TIMING_OPERATOR_TEXT, True)
    assert len(llm.prompts) == 1  # спросили модель; она недоступна


def test_data_weak_match_prints_the_deadline_of_the_row_the_model_picked(mini_prod_world, monkeypatch):
    llm = _ReadinessLLM(wanted=(_TSH,))
    _deploy(monkeypatch, llm, ("ттг",))
    turn = _ask("сколько готовится ттг", "strong-data-pick", mini_prod_world)
    assert turn["text"] == f"Срок готовности «{_TSH}» — 1–2 дня."
    assert len(llm.prompts) == 1 and "1-2" not in llm.prompts[0]  # срок модели не показывается


@pytest.mark.parametrize(
    ("question", "phrase", "name"),
    [
        ("сколько готовится ферритин", "ферритин", "Ферритин"),  # дословное название
        ("сколько готовится анализ на сахар", "анализ на сахар", "Глюкоза"),  # синоним МИС на одну услугу
        ("сколько готовится анализ 640", "640", "Глюкоза"),  # код услуги
    ],
)
def test_data_strong_match_prints_the_deadline_without_the_model(question, phrase, name, mini_prod_world, mis_synonyms, monkeypatch):
    llm = _ReadinessLLM(fail=True)
    _deploy(monkeypatch, llm, (phrase,))
    turn = _ask(question, f"strong-data-strong-{question}", mini_prod_world)
    assert turn["text"] == f"Срок готовности «{name}» — 1–2 дня."
    assert llm.prompts == []


def test_data_ambiguous_mis_synonym_is_not_one_deadline(mini_prod_world, mis_synonyms, monkeypatch):
    """Синоним МИС на четыре анализа — не одна услуга: срок одного из них без выбора не печатается."""

    llm = _ReadinessLLM(fail=True)
    _deploy(monkeypatch, llm, ("анализ на холестерин",))
    turn = _ask("сколько готовится анализ на холестерин", "strong-data-cholesterol", mini_prod_world)
    assert (turn["text"], turn["handoff"]) == (RESULT_TIMING_OPERATOR_TEXT, True)
    assert len(llm.prompts) == 1  # не одна услуга — выбор модели, а она недоступна
