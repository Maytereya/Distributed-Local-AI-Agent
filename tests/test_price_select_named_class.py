"""Выбор строки прайса: «та услуга, записанная иначе» и «другой предмет — не та» (п. 3 топ-5 разбора 95 цен, классы F + P1;
выключатель `price_select_named`, off).

Разбор 95 (деплой 18): P1 — верная строка в списке LLM была, ответ «ни одна»: «Cito …» на «срочный» анализ, «… D2 и D3,
общий» на «D3», копрограмма на «общий анализ кала» (g0048, g0587, g0452); F — нужной строки в первом списке нет, LLM брала
ближайшую чужую: ЭхоКГ плода на «эхо сердца» (g0419). Второй круг поиска по терминам LLM уже есть, но до него F не доходил
(выбор не пуст), а P1 упирался в «ни одна» на верной строке.

Инварианты:
- путь ответа о цене под on — промпты `*_general_named`: прежний промпт пути + правила 6 и 7 (правило 4 отсылает к 7), в
  двух локациях одинаково;
  суд записи (`price_answer=False`) и off — прежние промпты;
- первый круг пуст и LLM сказала «слишком общее», не видя ни одной строки, — под on второй круг ищет и по редким словам её
  терминов; off — как было.
Живое поведение правил — карточка (test / dev, варианты деплоя 17, встречный свип), не этот тест.
"""

from __future__ import annotations

import asyncio
import pathlib
import pytest

from messengers_router.choice import Choice, Status
from messengers_router.services import _price_select_llm as sel

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASES = ["choose_price_service_general", "choose_price_service_qualifier_general"]


@pytest.fixture
def named(monkeypatch):
    from messengers_router.runtime_config import config as cfg

    def set_switch(value: str) -> None:
        monkeypatch.setattr(cfg, "MR_PRICE_SELECT_NAMED", value, raising=False)

    return set_switch


@pytest.mark.parametrize(("judge", "task"), [(False, "price_service_general_named"), (True, "price_service_qualifier_general_named")])
def test_price_answer_reads_the_named_prompt_under_on(judge, task, named) -> None:
    named("on")
    assert sel.price_task(judge_qualifier=judge).name == task


@pytest.mark.parametrize("value", ["off", "", "мусор"])
@pytest.mark.parametrize(("judge", "task"), [(False, "price_service_general"), (True, "price_service_qualifier_general")])
def test_off_keeps_the_prompts_of_prod(judge, task, value, named) -> None:
    named(value)
    assert sel.price_task(judge_qualifier=judge).name == task


@pytest.mark.parametrize(("judge", "task"), [(False, "price_service"), (True, "price_service_qualifier")])
def test_booking_judge_keeps_the_old_prompt_under_on(judge, task, named) -> None:
    named("on")
    assert sel.price_task(judge_qualifier=judge, price_answer=False).name == task


def _text(path: str) -> str:
    return (_ROOT / path).read_text(encoding="utf-8")


@pytest.mark.parametrize("base", _BASES)
def test_named_prompt_is_the_prod_prompt_plus_rules_6_and_7(base) -> None:
    old, new = _text(f"messengers_router/prompts/{base}.txt").splitlines(), _text(f"messengers_router/prompts/{base}_named.txt").splitlines()
    added = [line for line in new if line not in old]
    removed = [line for line in old if line not in new]
    # Прежние строки не тронуты, кроме правила 4: оно отсылает к правилу 7 (позиции без особого условия в списке нет — не та).
    assert [line[:2] for line in removed] == ["4."], removed
    assert [line[:2] for line in added] == ["4.", "6.", "7."]
    rule4, rule6, rule7 = added
    assert rule4 == removed[0].replace("когда есть позиция без него.", "когда есть позиция без него; если позиции без него нет — смотри правило 7.")
    assert "«Cito» в названии позиции — это и есть «срочно»" in rule6
    for word in ("другом предмете", "другом пациенте", "которого пациент не называл"):
        assert word in rule7, word
    # Карточка 1 (26.10): «более узкая специальность» в правиле 7 и «пояснение / близкие показатели» в правиле 6 делали выбор
    # строже нужного (дерматолог, липидный спектр → «уточните») и подменяли марку (Immulite → RIDA-2) — сняты.
    assert "специальност" not in rule7 and "близкими показателями" not in rule6
    assert "названия для поиска" in rule7  # «не та» ведёт во второй круг, а не к ближайшей строке
    assert new.index(rule7) == new.index(rule6) + 1 < new.index("Ответ:")


@pytest.mark.parametrize("base", _BASES)
def test_named_prompt_lives_in_both_places(base) -> None:
    assert _text(f"messengers_router/prompts/{base}_named.txt") == _text(f"app_data/prompts/mr_{base}_named.txt")


# --------------------------------------------------------------------------- второй круг при пустом первом

_ROWS = [
    {"serviceId": 1, "serviceName": "Комплекс ЗППП основной", "cost": 3000, "serviceHomecode": "1"},
    {"serviceId": 2, "serviceName": "Общий анализ крови", "cost": 390, "serviceHomecode": "2"},
]


def _select(monkeypatch, named, switch: str) -> list[list[str]]:
    """Первый круг: список пуст, LLM — «слишком общее» и термин; второй круг — какие строки увидела LLM."""

    named(switch)
    seen: list[list[str]] = []

    async def fake_choose(task, question, options, **_kw) -> Choice:
        seen.append([o.text for o in options])
        if len(seen) == 1:
            return Choice(status=Status.NONE_FIT, reason="too_general", terms=("ЗППП",))
        return Choice(status=Status.NONE_FIT, reason="other_service")

    monkeypatch.setattr(sel, "choose", fake_choose)
    monkeypatch.setattr(sel, "_build_pool", lambda texts, rows, **_kw: [] if not seen else list(_kw.get("first") or ()))
    monkeypatch.setattr(sel, "_rare_rows", lambda texts, rows, **_kw: [_ROWS[0]] if list(texts) == ["ЗППП"] else [])
    monkeypatch.setattr(sel, "_synonym_rows", lambda texts, rows: [])
    monkeypatch.setattr(sel, "strong_rows", lambda *a, **k: [])
    monkeypatch.setattr(sel, "names_service", lambda q: True)
    asyncio.run(sel.select_price_rows("сколько стоит сдать на ипп", _ROWS, candidates=[_ROWS[1]], price_answer=True, general_as_none_fit=True))
    return seen


def test_empty_first_round_searches_rare_words_of_the_terms_under_on(monkeypatch, named) -> None:
    seen = _select(monkeypatch, named, "on")

    assert len(seen) == 2 and seen[0] == [], seen
    assert any(text.startswith("Комплекс ЗППП основной") for text in seen[1]), seen


def test_empty_first_round_as_before_under_off(monkeypatch, named) -> None:
    seen = _select(monkeypatch, named, "off")

    assert all(not text.startswith("Комплекс ЗППП основной") for texts in seen[1:] for text in texts), seen
