"""Гард П2 у названия из сущности NLU / контекста судит LLM (BUG-2026-10-15-P2-ENTITY-BYPASS, `p2_llm_judge`).

П2 (решение владельца): уточнение, которого клиника не оказывает («по ОМС»), — честное «не нашёл», а не платная цена
«почти той» услуги. Резолвер по реплике «узи щитовидной железы по омс» отказывает (гард П2), но `price_info` /
`service_bundle_info` брали название из сущности NLU / контекста — и пациент получал платную цену УЗИ. Пятый стык
по буквам (ветка `exp/p2-entity-guard`) встречный свип отверг: 18 ложных заглушений из 28 — разговорное слово вне
словаря прайса («первый приём невролога», «для иностранного гражданина») гард считает уточнением (инвариант 3 класса
`guard_over_blocks_silently`). Путь — LLM-first, как класс A (`parse_price_guard`): «та ли это услуга с этим
свойством» решает выбор строки (`select_price_rows(..., judge_qualifier=True)`, правило 3 промпта отсекает «по ОМС»).

Инварианты класса (через обработчик эндпоинта, мини-прод, `turn_parse` off и on, оракул LLM):
- `off` — как было: цена по сущности, LLM о строке не спрашивается;
- `on` + LLM «не та» — без цены («не нашёл»); `on` + LLM выбрала строку — цена;
- `on` + LLM недоступна — ответ тот же, что под `off` (не делаем хуже при сбое);
- разговорные формулировки свипа под `on` судит LLM, не словарь: выбрала строку — цена;
- реплики вне класса (без уточнения П2, без сущности) под `on` не меняются;
- два хода: «а сколько стоит по омс?» после цены — услуга из контекста — тоже судит LLM; короткий вопрос после
  «не нашёл» не получает цену того, чего пациент не видел.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_shadow import _deploy
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _Oracle, _assert_follow_up_stays_on_what_was_shown

_P2 = "сколько стоит узи щитовидной железы по омс"
_USG = "Ультразвуковое исследование щитовидной железы и паращитовидных желез — 2 100"
# Строки, которые оракул «выбирает»: префикс строки варианта «название — цена руб.» (только базовые приёмы).
_PICK = (
    "Ультразвуковое исследование щитовидной железы и паращитовидных желез — 2100",
    "Прием (осмотр, консультация) врача- невролога — 2700",
    "Прием (осмотр, консультация) врача-терапевта — 2700",
)
_CONVERSATIONAL = (  # встречный свип 15.10: g0766, g0300 — сущность была единственным путём к верной цене
    "Сколько стоит первый приём невролога?",
    "Сколько стоит первый приём терапевта?",
    "Цена приёма терапевта для иностранного гражданина такая же?",
)
_OUTSIDE_THE_CLASS = (
    "сколько стоит узи щитовидной железы",
    "сколько стоит узи щитовидки",
    "сколько стоит ферритин",
    "сколько стоит анализ на ферритин",
    "сколько стоит приём терапевта",
    "сколько стоит ультразвуковое исследование щитовидной железы и паращитовидных желез",
    "сколько стоит ферритин по омс",  # уточнение П2 без сущности — «Уточните» и до правки
    "сколько стоит приём терапевта по омс",
)


class _Down:
    """LLM недоступна во всех местах (как промах кэша стенда)."""

    async def generate(self, **_kwargs):
        raise ConnectionError("тест: LLM недоступна")


class _PickOnP2(_Oracle):
    """Выбирает строки `_PICK` только на реплике с «омс»; на прочих — «подходящих нет» (первый ход двух — список
    правил, как под `not_it`)."""

    def __init__(self) -> None:
        super().__init__((), _PICK)

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        question = re.search(r"пациента: «(.*?)»", prompt, re.S)  # не правило 3 промпта, где тоже «по ОМС»
        self.rows = _PICK if question and "омс" in question.group(1).lower() else ("—",)
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _llm(kind: str):
    """`not_it` — на выбор строки LLM отвечает «подходящих нет»; `pick` — выбирает строки `_PICK`; `pick_p2` — их же,
    но только на реплике с «омс»; `down` — сбой. Выбор фраз разбора у оракулов — пустой; прочие места — «недоступна»."""

    return {
        "not_it": lambda: _Oracle((), ("—",)),
        "pick": lambda: _Oracle((), _PICK),
        "pick_p2": _PickOnP2,
        "down": _Down,
    }[kind]()


def _run(turns, world, monkeypatch, *, judge: str, parse: str, llm: str, session: str) -> tuple[list[dict], list[dict], object]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    monkeypatch.setattr(cfg, "MR_TURN_PARSE", parse, raising=False)
    monkeypatch.setattr(cfg, "MR_P2_LLM_JUDGE", judge, raising=False)
    client = _llm(llm)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", client)
    with ev_world.FrozenWorld(world, "day") as frozen:
        record = run_dialog(list(turns), session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return record, rows, client


def _priced(text: str) -> bool:
    return bool(re.search(r"\d\s*руб", text))


def test_precondition_the_resolver_refuses_and_the_entity_names_the_service(mini_prod_world):  # noqa: F811
    """Предусловие класса на мини-прайсе: резолвер по реплике отказывает (П2), а фраза грундера без «по омс»
    находит услугу — сущность есть; разговорные слова свипа гард по буквам тоже считает уточнением."""

    from agent_logic_2.nayka_api import api_price

    from messengers_router.entity_grounder import extract_service_phrase
    from messengers_router.services import _prices_helpers as H

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        assert H.query_has_unsatisfiable_qualifier(_P2, rows)
        assert H.resolve_price_service_name_from_catalog(_P2) is None
        assert "омс" not in extract_service_phrase(_P2).lower()
        for reply in _CONVERSATIONAL:
            assert H.query_has_unsatisfiable_qualifier(reply, rows), reply


@pytest.mark.parametrize("llm", ["not_it", "pick", "down"])
@pytest.mark.parametrize("parse", ["off", "on"])
def test_off_keeps_the_old_answer(parse, llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Как было: цена УЗИ по сущности, выбор строки LLM не спрашивается (гард отсекает реплику до LLM)."""

    [turn], [row], client = _run([_P2], mini_prod_world, monkeypatch, judge="off", parse=parse, llm=llm, session=f"p2llm-off-{parse}-{llm}")
    assert _USG in turn["text"], turn["text"]
    assert "rows" not in getattr(client, "asked", [])
    assert "p2_llm_judge" not in row


@pytest.mark.parametrize("parse", ["off", "on"])
def test_on_the_llm_says_not_it_and_there_is_no_price(parse, mini_prod_world, monkeypatch):  # noqa: F811
    [turn], [row], client = _run([_P2], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="not_it", session=f"p2llm-not-{parse}")
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert "rows" in client.asked and row["p2_llm_judge"] == "not_found"


@pytest.mark.parametrize("parse", ["off", "on"])
def test_on_the_llm_picks_the_row_and_it_is_priced(parse, mini_prod_world, monkeypatch):  # noqa: F811
    [turn], [row], _client = _run([_P2], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="pick", session=f"p2llm-pick-{parse}")
    assert _USG in turn["text"], turn["text"]
    assert row["p2_llm_judge"] == "rows"


@pytest.mark.parametrize("parse", ["off", "on"])
@pytest.mark.parametrize("reply", (_P2, *_CONVERSATIONAL))
def test_on_with_the_llm_down_answers_as_off(reply, parse, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM — ответ как без выключателя (цена по сущности): не делаем хуже при сбое."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge="off", parse=parse, llm="down", session=f"p2llm-d-off-{parse}-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="down", session=f"p2llm-d-on-{parse}-{reply[-9:]}")
    assert on["text"] == off["text"] and _priced(on["text"]), on["text"]
    assert row["p2_llm_judge"] == "fallback"


@pytest.mark.parametrize("parse", ["off", "on"])
@pytest.mark.parametrize("reply", _CONVERSATIONAL)
def test_a_conversational_word_is_judged_by_the_llm_not_by_the_dictionary(reply, parse, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный инвариант (класс `guard_over_blocks_silently`): «первый», «для иностранного гражданина» гард по
    буквам считал уточнением П2. Под `on` их судит LLM: выбрала строку приёма — цена, как под `off`; решение —
    за LLM (тот же ход с ответом «не та» — без цены), а не за словарём каталога."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge="off", parse=parse, llm="pick", session=f"p2llm-c-off-{parse}-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="pick", session=f"p2llm-c-on-{parse}-{reply[-9:]}")
    assert _priced(off["text"]) and _priced(on["text"]), on["text"]
    assert row["p2_llm_judge"] == "rows"
    [not_it], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="not_it", session=f"p2llm-c-not-{parse}-{reply[-9:]}")
    assert not _priced(not_it["text"]) and row["p2_llm_judge"] == "not_found", not_it["text"]


@pytest.mark.parametrize("parse", ["off", "on"])
@pytest.mark.parametrize("reply", _OUTSIDE_THE_CLASS)
def test_on_does_not_touch_replies_outside_the_class(reply, parse, mini_prod_world, monkeypatch):  # noqa: F811
    """Без уточнения П2 или без названия из сущности — ответ под `on` тот же, что под `off`, судья не зовётся."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge="off", parse=parse, llm="not_it", session=f"p2llm-o-off-{parse}-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="not_it", session=f"p2llm-o-on-{parse}-{reply[-9:]}")
    assert on["text"] == off["text"], on["text"]
    assert "p2_llm_judge" not in row


_TWO_TURNS = ("сколько стоит узи щитовидной железы", "а сколько стоит по омс?")


@pytest.mark.parametrize("llm", ["not_it", "pick_p2", "down"])
@pytest.mark.parametrize("judge", ["off", "on"])
@pytest.mark.parametrize("parse", ["off", "on"])
def test_two_turns_the_p2_follow_up_on_the_context_service(parse, judge, llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Второй ход с уточнением П2 без названия — услуга из контекста диалога. off — как было (цена услуги
    контекста); on — решает LLM: «не та» — без цены, выбрала строку или сбой — цена. Первый ответ не меняется."""

    (first, second), _rows, _ = _run(
        _TWO_TURNS, mini_prod_world, monkeypatch, judge=judge, parse=parse, llm=llm, session=f"p2llm-two-{parse}-{judge}-{llm}"
    )
    assert _priced(first["text"]), first["text"]
    assert _priced(second["text"]) == (judge == "off" or llm != "not_it"), second["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
@pytest.mark.parametrize("parse", ["off", "on"])
def test_follow_up_after_not_found_stays_on_what_was_shown(parse, follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    """После «не нашёл» короткий вопрос («а сколько?», «да», цифра) не получает цену услуги из сущности."""

    (first, second), _rows, _ = _run(
        [_P2, follow_up], mini_prod_world, monkeypatch, judge="on", parse=parse, llm="not_it", session=f"p2llm-fu-{parse}-{follow_up}"
    )
    assert not _priced(first["text"]), first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
