"""Кандидаты строки цены по смыслу (`price_candidates_by_meaning`, п. 1 плана 19.10; ход (в), L-31).

Диагностика 19.10 (test-срез, 8 настоящих промахов «Уточните название» на названную услугу): решали буквы, LLM до выбора
не доходила —
1) словарный гард П2 считал незнакомое прайсу слово «уточнением, которого нет» («бак», «ВМС», «карбамазепина»,
   «доплером») и не пускал вопрос к LLM;
2) пул из 24 строк заполняли частые слова («анализ», «крови»), строка по редкому слову в него не попадала;
3) сокращения в две буквы («ЛГ», «аТ ТГ») разбор цены отбрасывал как шум — «услуга не названа».

Инварианты класса (под on; под off — как было):
- незнакомое прайсу слово не останавливает вопрос: его судит LLM (правило 3 промпта), «условия нет» — `none_fit` без
  второго круга (П2 не ослаблен: чужой цены нет);
- строка по редкому слову реплики — в списке LLM, даже когда частые слова дают полный пул;
- сокращение с заглавной — слова об услуге; предлоги строчными — нет;
- термины второго круга ищутся и по редким словам;
- выбирает по-прежнему LLM: строки показываются только выбранные.
"""

from __future__ import annotations

import asyncio

import pytest

from messengers_router.choice import Choice, Status
from messengers_router.services import _price_select_llm as sel

_FIRST = sel._RARE_WORDS * sel._PER_RARE_WORD

# Частые слова реплики «анализ крови на концентрацию …» — по десятку строк на слово: поиск по отдельным словам
# заполняет ими пул (24) раньше, чем доходит до редкого слова (как в живом прайсе, диагностика 19.10).
COMMON = [
    {"serviceId": 100 + i, "serviceName": f"{stem} {i}", "cost": 300 + i, "priceUnitId": 1}
    for stem in ("Анализ мочи вариант", "Анализ кала вариант", "Концентрация антител вариант")
    for i in range(12)
]
ROWS = [
    *COMMON,
    {"serviceId": 13440, "serviceName": "Карбамазепин (Carbamazepine)", "cost": 1200, "priceUnitId": 1},
    {"serviceId": 1270, "serviceName": "Эхокардиография", "cost": 2500, "priceUnitId": 1},
    {"serviceId": 1271, "serviceName": "Эхокардиография плода", "cost": 2900, "priceUnitId": 1},
    {"serviceId": 352, "serviceName": "ЛГ (LH) Лютеинизирующий гормон", "cost": 450, "priceUnitId": 1},
    {"serviceId": 343, "serviceName": "АТ - ТГ (Anti-Tg) антитела к тиреоглобулину", "cost": 600, "priceUnitId": 1},
    {"serviceId": 5039, "serviceName": "Введение внутриматочного контрацептива", "cost": 3000, "priceUnitId": 1},
    {"serviceId": 7001, "serviceName": "Приём терапевта первичный", "cost": 1800, "priceUnitId": 1},
]


class _Oracle:
    """Подделка `choose()`: запоминает списки позиций; отвечает по очереди заданными выборами."""

    def __init__(self, *answers: Choice):
        self.answers = list(answers)
        self.calls: list[list[str]] = []

    async def __call__(self, task, question, options, **_kw):
        self.calls.append([o.text for o in options])
        return self.answers.pop(0) if self.answers else Choice(Status.NONE_FIT, reason="other_service")


def _run(monkeypatch, mode: str, question: str, oracle: _Oracle, **kw):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", mode, raising=False)
    monkeypatch.setattr(sel, "choose", oracle)
    monkeypatch.setattr(sel, "_record_synonym_suggestion", lambda *a, **k: None)
    return asyncio.run(sel.select_price_rows(question, ROWS, **kw))


def _offered(oracle: _Oracle, call: int, name: str) -> bool:
    return any(name in text for text in oracle.calls[call]) if len(oracle.calls) > call else False


def test_rare_word_row_is_offered_even_when_common_words_fill_the_pool(monkeypatch):
    q = "Сколько стоит анализ крови на концентрацию карбамазепина"
    on = _Oracle(Choice(Status.PICKED, ids=("0",)))
    picked = _run(monkeypatch, "on", q, on)
    # Строка по редкому слову — в списке, хотя частые слова заполняют его целиком (на живом прайсе 20261008 под off её в
    # списке из 24 строк нет вовсе — замер 19.10), и в его хвосте, а не первой (карточка dev 20.10: первое место LLM
    # принимала за подсказку).
    assert on.calls and len(on.calls[0]) > 3 * _FIRST // 2, on.calls  # частые слова дают полный по сути список
    assert any("Карбамазепин" in text for text in on.calls[0][-_FIRST:]), on.calls
    assert not any("Карбамазепин" in text for text in on.calls[0][:3]), on.calls
    assert picked is not None


@pytest.mark.parametrize("question", ["ЛГ стоимость", "Цена аТ ТГ"])
def test_capitalised_short_abbreviation_reaches_the_llm(question, monkeypatch):
    on = _Oracle(Choice(Status.NONE_FIT, reason="other_service"))
    _run(monkeypatch, "on", question, on)
    assert on.calls, "под on сокращение — слова об услуге, LLM зовётся"
    off = _Oracle()
    assert _run(monkeypatch, "off", question, off) is None and not off.calls  # как было: «услуга не названа»


def test_lowercase_prepositions_are_not_abbreviations():
    assert sel._short_name_words("платить за взятие крови из вены") == []
    assert sel._short_name_words("Цена аТ ТГ") == ["ат", "тг"]


def _dictionary_guard_fires(monkeypatch) -> None:
    """Словарный гард П2 считает слово реплики «уточнением, которого нет в прайсе» (на живом прайсе — «вмс», «бак»,
    «карбамазепина»; на синтетике словарь гарда мал, поэтому его вердикт задаёт тест)."""

    monkeypatch.setattr(sel, "query_has_unsatisfiable_qualifier", lambda *_a, **_k: True)
    monkeypatch.setattr(sel, "query_names_service_in_other_words", lambda *_a, **_k: False)


def test_word_unknown_to_the_price_is_judged_by_the_llm_not_the_dictionary(monkeypatch):
    _dictionary_guard_fires(monkeypatch)
    q = "Введение ВМС — цена"
    off = _Oracle()
    assert _run(monkeypatch, "off", q, off) is None and not off.calls  # как было: ответ правил до LLM
    on = _Oracle(
        Choice(Status.NONE_FIT, reason="other_service", terms=("Введение внутриматочного контрацептива",)),
        Choice(Status.PICKED, ids=("0",)),
    )
    picked = _run(monkeypatch, "on", q, on)
    assert on.calls, "словарный гард П2 не остановил вопрос — его судит LLM"
    assert picked is not None and [r["serviceId"] for r in picked.rows] == [5039]


def test_condition_absent_from_the_price_is_still_no_price(monkeypatch):
    """П2 не ослаблен: «по ОМС» судит LLM (правило 3) — «условия нет», без второго круга и без строк."""

    _dictionary_guard_fires(monkeypatch)
    on = _Oracle(Choice(Status.NONE_FIT, reason="qualifier_absent", terms=("Приём терапевта",)))
    picked = _run(monkeypatch, "on", "сколько стоит приём терапевта по ОМС", on)
    assert len(on.calls) == 1
    assert picked is not None and picked.status == "none_fit" and not picked.rows


@pytest.mark.parametrize("failure", [Choice(Status.UNAVAILABLE), Choice(Status.INVALID)], ids=["unavailable", "invalid"])
def test_llm_failure_after_the_bypass_is_the_rules_answer(failure, monkeypatch):
    """Гард пропустил вопрос только ради суда LLM: без её решения — ответ правил, как под off (встречный свип 19.10:
    иначе «Уролог цена приема» при сбое LLM уходил к оператору вместо цены правил)."""

    _dictionary_guard_fires(monkeypatch)
    candidates = [ROWS[-1]]
    assert _run(monkeypatch, "on", "Уролог цена приема", _Oracle(failure), candidates=candidates) is None


def test_round_two_terms_are_searched_by_rare_words(monkeypatch):
    on = _Oracle(
        Choice(Status.NONE_FIT, reason="other_service", terms=("Эхокардиография с цветовым доплером",)),
        Choice(Status.NONE_FIT, reason="other_service"),
    )
    _run(monkeypatch, "on", "Сколько стоит ЭХО КГ с цветовым доплером", on)
    assert len(on.calls) == 2 and _offered(on, 1, "Эхокардиография"), on.calls


def test_off_does_not_search_rare_words(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("под off поиск по редким словам не зовётся")

    monkeypatch.setattr(sel, "_rare_rows", boom)
    _run(monkeypatch, "off", "Сколько стоит анализ крови на концентрацию карбамазепина", _Oracle())


# --- Карточка dev 20.10 (BUG-2026-10-19-PRICE-NAMED-SERVICE-NOT-SHOWN-TO-LLM, доработка): два класса под on ---
# 1) Обход гарда П2 потерял настоящее условие: «ФКС с наркозом» → 8 100 руб. без наркоза, «экспресс 17-ОН день в день» →
#    обычная цена. Инвариант: вопрос, который судит LLM (гард сработал или суд условия от разбора), идёт с промптом, где
#    вариант выполнения, от которого зависит цена (срочно, наркоз, на дому), — тоже условие; под off и без суда — прежний.
# 2) Редкие слова всей реплики тянули в начало списка чужие строки («готов» → «Консультация готовых гистологических
#    стёкол», «сердца» → «Здоровье сердца и сосудов!»), во втором круге — и после «слишком общее». Инвариант: строки
#    по редким словам — только из названной услуги, в хвосте списка, во втором круге — только если LLM признала услугу.

MORE_ROWS = [
    *ROWS,
    {"serviceId": 1516, "serviceName": "Тотальная фиброколоноскопия (ФКС)", "cost": 8100, "priceUnitId": 1},
    {"serviceId": 501, "serviceName": "Посев на грибы дерматофиты", "cost": 750, "priceUnitId": 1},
    {"serviceId": 9001, "serviceName": "Консультация готовых гистологических стекол (1 стекло)", "cost": 850, "priceUnitId": 1},
    {"serviceId": 9002, "serviceName": "«Вакцинация COVID-19: готов ли ваш организм?»", "cost": 2000, "priceUnitId": 1},
    {"serviceId": 9003, "serviceName": "Здоровье сердца и сосудов!", "cost": 1470, "priceUnitId": 1},
    {"serviceId": 9004, "serviceName": "Биохимия крови", "cost": 2835, "priceUnitId": 1},
]


class _TaskOracle(_Oracle):
    """Как `_Oracle`, плюс имя задачи (промпта) каждого вызова."""

    def __init__(self, *answers: Choice):
        super().__init__(*answers)
        self.tasks: list[str] = []

    async def __call__(self, task, question, options, **kw):
        self.tasks.append(task.name)
        return await super().__call__(task, question, options, **kw)


def _run_rows(monkeypatch, mode: str, question: str, oracle: _Oracle, *, mention=None, **kw):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", mode, raising=False)
    monkeypatch.setattr(sel, "choose", oracle)
    monkeypatch.setattr(sel, "_record_synonym_suggestion", lambda *a, **k: None)
    monkeypatch.setattr(sel, "_mention_phrase", lambda _q: mention)
    return asyncio.run(sel.select_price_rows(question, MORE_ROWS, **kw))


def test_bypassed_guard_is_judged_with_the_price_changing_variant_prompt(monkeypatch):
    _dictionary_guard_fires(monkeypatch)
    on = _TaskOracle(
        Choice(Status.NONE_FIT, reason="other_service", terms=("Фиброколоноскопия под наркозом",)),
        Choice(Status.NONE_FIT, reason="qualifier_absent"),
    )
    _run_rows(monkeypatch, "on", "Стоимость фиброколоноскопии с наркозом", on)
    assert on.tasks == ["price_service_qualifier", "price_service_qualifier"]  # оба круга судят условие


@pytest.mark.parametrize(
    ("mode", "judge", "task"),
    [
        ("on", True, "price_service_qualifier"),  # суд условия от разбора хода — тоже строгий промпт
        ("off", True, "price_service"),  # под off прод не меняется
        ("on", False, "price_service"),  # без суда условия — прежний промпт для всех вопросов о цене
    ],
)
def test_qualifier_prompt_only_for_judged_questions_under_on(mode, judge, task, monkeypatch):
    oracle = _TaskOracle(Choice(Status.NONE_FIT, reason="other_service"))
    _run_rows(monkeypatch, mode, "сколько стоит анализ на концентрацию карбамазепина срочно", oracle, judge_qualifier=judge)
    assert oracle.tasks and set(oracle.tasks) == {task}


def test_qualifier_prompt_differs_only_by_rule_three_and_lives_in_both_places():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    repo = (root / "messengers_router/prompts/choose_price_service_qualifier.txt").read_text(encoding="utf-8")
    prod = (root / "app_data/prompts/mr_choose_price_service_qualifier.txt").read_text(encoding="utf-8")
    base = (root / "messengers_router/prompts/choose_price_service_conditions.txt").read_text(encoding="utf-8")
    assert repo == prod  # прод берёт app_data/prompts — две локации одинаковы
    changed = [(a, b) for a, b in zip(base.splitlines(), repo.splitlines()) if a != b]
    assert len(base.splitlines()) == len(repo.splitlines()) and len(changed) == 1
    rule = changed[0][1]
    assert rule.startswith(changed[0][0].split(" — подходящих позиций нет.")[0])  # прежнее начало правила 3 — как было
    for word in ("наркоз", "экспресс", "на дому"):
        assert word in rule
    assert "условия сбора" in rule  # вопрос о сдаче — по-прежнему не условие (`price_judge_conditions`)


def _rare_sources(monkeypatch) -> list[list[str]]:
    """Тексты, из которых бралась каждая порция строк по редким словам (первый круг, затем термины второго)."""

    calls: list[list[str]] = []
    real = sel._rare_rows

    def spy(texts, rows, **kw):
        calls.append([str(t) for t in texts])
        return real(texts, rows, **kw)

    monkeypatch.setattr(sel, "_rare_rows", spy)
    return calls


@pytest.mark.parametrize(
    ("question", "mention", "junk"),
    [
        ("Сдала посев на грибы, анализ готов, перешла по коду — что дальше?", "посев на грибы", ("готов", "коду")),
        ("Сколько стоит ЭхоКГ узи сердца?", "ЭхоКГ", ("сердца",)),
        ("цены на анализы можешь озвучить?", "", ("озвучить", "анализы")),  # понимание хода: услуги не названо
    ],
)
def test_rare_words_come_only_from_the_named_service(question, mention, junk, monkeypatch):
    sources = _rare_sources(monkeypatch)
    _run_rows(monkeypatch, "on", question, _Oracle(), mention=mention, judge_qualifier=True)
    assert sources, "под on поиск по редким словам зовётся"
    words = " ".join(sources[0]).lower()
    assert not any(j in words for j in junk), sources[0]


def test_rare_rows_without_understanding_use_the_rules_phrase(monkeypatch):
    sources = _rare_sources(monkeypatch)
    _run_rows(monkeypatch, "on", "Сколько стоит анализ крови на концентрацию карбамазепина", _Oracle(), mention=None)
    assert sources and "сколько" not in " ".join(sources[0]).lower() and "карбамазепин" in " ".join(sources[0]).lower()


def test_round_two_after_too_general_adds_no_rare_rows(monkeypatch):
    sources = _rare_sources(monkeypatch)
    oracle = _Oracle(
        Choice(Status.NONE_FIT, reason="too_general", terms=("Биохимический анализ крови",)),
        Choice(Status.NONE_FIT, reason="too_general"),
    )
    _run_rows(monkeypatch, "on", "цены на анализы можешь озвучить?", oracle, mention="", judge_qualifier=True)
    assert len(oracle.calls) == 2
    assert not any("Биохимический анализ крови" in texts for texts in sources), sources


def test_round_two_after_other_service_still_searches_rare_words(monkeypatch):
    on = _Oracle(
        Choice(Status.NONE_FIT, reason="other_service", terms=("Карбамазепин в крови",)),
        Choice(Status.NONE_FIT, reason="other_service"),
    )
    _run_rows(monkeypatch, "on", "Сколько стоит анализ на концентрацию лекарства", on)
    # Термины назвала LLM — строки по их редким словам первыми (проба 20.10: «Rh- принадлежность», «Эхокардиография»
    # находились во втором круге только так; хвост — для слов пациента в первом круге).
    assert len(on.calls) == 2 and any("Карбамазепин" in text for text in on.calls[1][:_FIRST]), on.calls


def test_short_abbreviation_is_taken_from_the_message_as_written(monkeypatch):
    """Фраза правил приходит строчными («лг»): сокращение берётся из реплики, как её написал пациент."""

    on = _Oracle(Choice(Status.NONE_FIT, reason="other_service"))
    _run_rows(monkeypatch, "on", "ЛГ стоимость", on)
    assert _offered(on, 0, "ЛГ (LH)"), on.calls
