"""Этап 2 без стоп-слов: биоматериал, заземление, классы walk-in (STEP-2026-10-24-S2-DOCS-PHRASE).

Решения владельца 22.10: 28 списков стоп-слов удалить; без LLM услугу не угадывать. Инварианты класса:
- классы «без записи» («анализы», «ЭКГ», «анализы и ЭКГ») — одно определение, варианты выбора LLM `walkin`; копии списка нет;
- все словоформы слова дают один ответ: путь `on` не делит слова на «знакомые списку» и прочие;
- путь `on` не читает удалённый список (AST) и не зависит от его содержимого (подмена списка ответов не меняет).
Тесты герметичны: словари МИС и прайс — синтетические.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from messengers_router import walkin
from messengers_router.choice import Choice, Status
from messengers_router.policies import nonbookable_service_hint

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"


# --------------------------------------------------------------------------- классы walk-in — одно определение


def test_walkin_classes_cover_what_the_bot_puts_into_service_name(monkeypatch) -> None:
    """Классы — ровно то, что ставит бот: ответы детектора при сбое LLM и любой ответ выбора LLM `walkin`."""

    detector = {nonbookable_service_hint(t) for t in ("где сдать кровь", "где сделать экг", "где сдать анализы и сделать экг")}
    assert detector == {"анализы", "ЭКГ", "анализы и ЭКГ"}

    async def picked_both(_task, _text, options):
        return Choice(status=Status.PICKED, ids=tuple(o.id for o in options))

    monkeypatch.setattr(walkin, "choose", picked_both)
    by_llm = asyncio.run(walkin.service("где сдать кровь и сделать экг", site="test"))
    for value in detector | {by_llm}:
        assert walkin.is_class(value) and walkin.is_class(value.upper()) and walkin.is_class(value.lower())


@pytest.mark.parametrize("value", ["Ферритин", "анализ", "анализ крови", "ЭКГ с нагрузкой", "УЗИ брюшной полости", "", None])
def test_catalog_names_and_other_words_are_not_walkin_classes(value) -> None:
    assert not walkin.is_class(value)


def test_no_module_keeps_its_own_copy_of_walkin_classes() -> None:
    """Список классов в заземлении (`_NONBOOKABLE_GENERIC_HINTS`) снят: читатели спрашивают `walkin.is_class`."""

    for path in _CORE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
        }
        assert "_NONBOOKABLE_GENERIC_HINTS" not in names, path
