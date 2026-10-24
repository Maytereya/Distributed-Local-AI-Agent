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
import re
from pathlib import Path

import pytest

from agent_logic_2 import config as cfg
from messengers_router import entity_grounder as G
from messengers_router import walkin
from messengers_router.choice import Choice, Status
from messengers_router.mess_types import RouteDecision, SessionState
from messengers_router.policies import nonbookable_service_hint
from messengers_router.service_phrase import extract_service_phrase
from messengers_router.services import Services

_CORE = Path(__file__).resolve().parents[1] / "messengers_router"


def _readers(path: Path, name: str) -> set[str]:
    """Функции модуля, в теле которых встречается имя `name`."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(fn))
    }


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


# --------------------------------------------------------------------------- заземление: значение LLM на метке без каталога

# Значения `service_name`, которые LLM-классификатор кладёт на метке без каталога, а фразы услуги в реплике нет: общие слова
# в разных формах (регулярка пути off знала их по основам и точным словам) и обычное название — эталон ответа.
_REPLY = "подскажите, кто ведёт"
_CONTROL = "общий анализ крови"
_GENERIC_FORMS = ("приём врача-кардиолога", "консультация доктора", "запись к специалисту", "консультацию специалиста")


def _ground(label: str, value: str, monkeypatch, *, mode: str) -> tuple[dict, set]:
    monkeypatch.setattr(cfg, "MR_GROUNDER_GENERIC_BY_LLM", mode, raising=False)
    decision = RouteDecision(label=label, confidence=0.9, entities={"service_name": value}, flags=set())
    out = asyncio.run(
        G.ground_decision_entities(decision=decision, user_text=_REPLY, state=SessionState(session_id=f"g-{label}"), services=Services())
    )
    return out.entities, out.flags


def test_values_reach_the_llm_value_branch() -> None:
    """Предусловие класса: ни реплика, ни значения фразы услуги не дают — решает ветка значения LLM."""

    assert extract_service_phrase(_REPLY) is None
    assert all(extract_service_phrase(v) is None for v in (_CONTROL, *_GENERIC_FORMS))


@pytest.mark.parametrize("label", ["DOCTOR_INFO", "PREPARE"])
def test_on_every_form_of_a_generic_word_gets_the_answer_of_a_plain_name(label, monkeypatch) -> None:
    control, _ = _ground(label, _CONTROL, monkeypatch, mode="on")
    assert control == {"service_name": _CONTROL}
    for value in _GENERIC_FORMS:
        entities, flags = _ground(label, value, monkeypatch, mode="on")
        assert entities == {"service_name": value} and "entity_kept_llm_service_name" in flags, value


@pytest.mark.parametrize("label", ["DOCTOR_INFO", "PREPARE"])
def test_off_drops_generic_words_by_letters_as_before(label, monkeypatch) -> None:
    assert _ground(label, _CONTROL, monkeypatch, mode="off")[0] == {"service_name": _CONTROL}
    for value in _GENERIC_FORMS:
        entities, flags = _ground(label, value, monkeypatch, mode="off")
        assert entities == {} and "entity_dropped_unverified_service_name" in flags, value


def test_on_does_not_depend_on_the_letter_regex(monkeypatch) -> None:
    """Мутация: регулярка «общее слово» совпадает со всем — под on ответ тот же, под off значение отброшено."""

    monkeypatch.setattr(G, "_GENERIC_SERVICE_FALLBACK_RE", re.compile("."))
    assert _ground("PREPARE", _CONTROL, monkeypatch, mode="on")[0] == {"service_name": _CONTROL}
    assert _ground("PREPARE", _CONTROL, monkeypatch, mode="off")[0] == {}


def test_only_the_off_helper_reads_the_letter_regex() -> None:
    assert _readers(Path(G.__file__), "_GENERIC_SERVICE_FALLBACK_RE") == {"_llm_service_name_is_generic"}
