"""Класс: каждая услуга списка получает свою цену; вопрос о сроке готовности цену не получает.

BUG-2026-10-05-PRICE-LIST-COLLAPSES-TO-ONE: список в вопросе цены сводится к одной услуге раньше
разбора корзины — подстановка каталога и грундер приземляют реплику целиком на одну строку
(«холестерин» → «Липидограмма»), а планировщик с известной услугой зовёт `service_bundle_info`,
где корзины нет. Сборщик корзины верен (`test_multi_price_survives_patient_synonyms`), но
живой путь до него не доходит — поэтому тест идёт через обработчик эндпоинта.

BUG-2026-10-05-READINESS-AS-PRICE: «когда будет готов анализ на ферритин» — подбор анализов
приземляется на услугу каталога и переклеивается в цену; переклейка смотрит только флаг
доставки результата, а не то, о каком свойстве спросили.

Инварианты класса:
- список из названий, которые каталог узнаёт, — цена каждой позиции, текстом и ответом на
  кнопку цены, со словом «сколько стоит» и без него, в новом диалоге и поверх старой услуги;
- вопрос о сроке готовности не получает цену и срока свободным текстом LLM: срок печатает код из
  справочника МИС, а нет срока или услуга не названа — честный перевод на оператора.

Ход (б), шаг 1 (L-14): `услуги[]` + `свойство` на метке «цена», под выключателем
`turn_understanding_price`. Тесты включают его явно (`on`). Разбор реплики на фразы (`turn_parse`) — единственный
режим (T7 шаг 3б, 16.10).
До шага 14 из 16 мест списка и 2 из 5 формулировок срока были красными.

Путь прода с 27.10 в гейте (`turn_spans_llm`, `price_by_parse` — on; STEP-2026-10-15-DEFAULTS-B): реплику на услуги размечает
LLM, без неё список и вопрос о сроке до своего предмета не доходят («Уточните…» / перевод). Поэтому разметку отвечает оракул
(`tests/spans_oracle.py`) — фразы, которые пациент назвал услугами; прочие места LLM — «недоступна», как в мире гейта.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_one_or_many_class import Case, _OracleLLM
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

# Список → названия услуг каталога, которые обязаны прозвучать в ответе.
LISTS = {
    "ферритин, глюкоза, холестерин": ("ферритин", "глюкоза", "липидограмма"),
    "оак, оам": ("общий анализ крови", "общий анализ мочи"),
    "ферритин и глюкоза": ("ферритин", "глюкоза"),
    "ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА": ("общий анализ крови", "общий анализ мочи", "ферритин", "глюкоза"),
}

# Как пациент задаёт список: (ходы до списка, шаблон реплики).
PATHS = {
    "text": ((), "сколько стоит {}"),
    "button": ((("menu.price.test", "Цена анализа"),), "{}"),
    "bare": ((), "{}"),
    "over_context": (("сколько стоит ТТГ",), "а {}?"),
}

# Вопрос о сроке → срок из МИС кодом (в мини-прод у ферритина и глюкозы срок «1-2»).
READINESS_KNOWN = {
    "когда будет готов анализ на ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "через сколько будет готов ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "сколько делается анализ на ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
    "сколько готовится ферритин": "Срок готовности «Ферритин» — 1–2 дня.",
}
# Услуга не названа, срока в МИС нет (у липидограммы поле пустое) или это список (строку каждой
# позиции выберет choose(), ход (в)) — честный перевод.
READINESS_UNKNOWN = ["сколько готовятся анализы", "когда будет готов анализ на липидограмму", "когда будет готов анализ на ферритин и глюкозу"]

# Что пациент назвал услугами (ответ оракула разметки; один перечень на диалог — длинная фраза раньше короткой).
# «анализы» само по себе — не услуга (правило 5 промпта `turn_spans`). «анализ на ферритин» — не дословное название: строку
# срока выбирает LLM (`choose_readiness`), оракул отмечает строку услуги, которую назвал пациент.
_READINESS_PHRASES = ("анализ на липидограмму", "анализ на ферритин", "ферритин", "глюкозу")
_READINESS_ROWS = {"ферритин": ("Ферритин",), "глюкозу": ("Глюкоза",), "анализ на липидограмму": ("Липидограмма",)}


@pytest.fixture
def understanding_on(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "on", raising=False)


def _spans(monkeypatch, phrases) -> SpansOracle:
    """Разметка слов отвечает `phrases`, прочие места LLM — «недоступна»; предохранитель LLM не размыкается от этих сбоев
    (в двухходовом пути разметку второго хода иначе не спросят — `keep_llm_breaker_closed`)."""

    oracle = SpansOracle(phrases)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    keep_llm_breaker_closed(monkeypatch)
    return oracle


def _readiness_oracle(monkeypatch) -> _OracleLLM:
    """Разметка — `_READINESS_PHRASES`, выбор строки срока — строки услуги из `_READINESS_ROWS`; прочие места — «недоступна»."""

    oracle = _OracleLLM(Case("", "readiness", _READINESS_ROWS), _READINESS_PHRASES)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    keep_llm_breaker_closed(monkeypatch)
    return oracle


def _list_phrases(items: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in re.split(r",| и |\s+", items) if part.strip())


# Сплошной текст «ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА» разрезает разметка слов (путь прода); отметка xfail «LLM недоступна — реплика
# целиком» (путь окон по данным МИС) снята 27.10 вместе с переводом теста на оракул разметки.
@pytest.mark.parametrize("items", sorted(LISTS))
@pytest.mark.parametrize("path", sorted(PATHS))
def test_every_listed_service_gets_its_price(path, items, mini_prod_world, understanding_on, monkeypatch):  # noqa: F811
    before, template = PATHS[path]
    oracle = _spans(monkeypatch, ("ТТГ", *_list_phrases(items)))
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, template.format(items)], f"list-{path}-{abs(hash(items))}")
    answer = record[-1]["text"].lower()
    missing = [name for name in LISTS[items] if name not in answer]
    assert missing == [], record[-1]["text"]
    assert "руб" in answer, record[-1]["text"]
    assert "spans" in oracle.asked


_BEFORE = [(), (("menu.price.test", "Цена анализа"),)]


@pytest.mark.parametrize("before", _BEFORE, ids=["text", "after_price_button"])
@pytest.mark.parametrize("question", sorted(READINESS_KNOWN))
def test_readiness_question_gets_the_deadline_from_mis(question, before, mini_prod_world, understanding_on, monkeypatch):  # noqa: F811
    """Срок печатает код из поля `deadline` строки прайса: без единицы — дни (решение 11.10)."""

    oracle = _readiness_oracle(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, question], f"ready-{len(before)}-{abs(hash(question))}")
    turn = record[-1]
    assert (turn["text"], turn["handoff"]) == (READINESS_KNOWN[question], False)
    assert "phrases" in oracle.calls


@pytest.mark.parametrize("before", _BEFORE, ids=["text", "after_price_button"])
@pytest.mark.parametrize("question", READINESS_UNKNOWN)
def test_readiness_without_deadline_is_honest_operator_not_price(question, before, mini_prod_world, understanding_on, monkeypatch):  # noqa: F811
    """Срока нет или услуга не названа: не цена и не срок свободным текстом LLM, а честный перевод —
    тот же текст, что у справочного вопроса о сроках в пути результатов (прод #675). Разметка отвечает (путь прода):
    перевод — потому что срока нет, а не потому что услугу не разобрали."""

    from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT

    _readiness_oracle(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        record = run_dialog([*before, question], f"ready-none-{len(before)}-{abs(hash(question))}")
    turn = record[-1]
    assert not re.search(r"\d\s*руб", turn["text"]), turn["text"]
    assert (turn["text"], turn["handoff"]) == (RESULT_TIMING_OPERATOR_TEXT, True)
