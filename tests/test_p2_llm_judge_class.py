"""Гард П2 у названия из сущности NLU / контекста судит LLM (BUG-2026-10-15-P2-ENTITY-BYPASS). Единственный путь с
17.10: выключатель `p2_llm_judge` снят (STEP-2026-10-17-ONE-PATH-P2-LLM-JUDGE); «без судьи» в тестах — подмена
`prices._p2_name_from_entity` (прежний ответ для сравнений), прежнее значение ключа в конфиге ничего не меняет.

П2 (решение владельца): уточнение, которого клиника не оказывает («по ОМС»), — честное «не нашёл», а не платная цена
«почти той» услуги. Резолвер по реплике «узи щитовидной железы по омс» отказывает (гард П2), но `price_info` /
`service_bundle_info` брали название из сущности NLU / контекста — и пациент получал платную цену УЗИ. Пятый стык
по буквам (ветка `exp/p2-entity-guard`) встречный свип отверг: 18 ложных заглушений из 28 — разговорное слово вне
словаря прайса («первый приём невролога», «для иностранного гражданина») гард считает уточнением (инвариант 3 класса
`guard_over_blocks_silently`). Путь — LLM-first, как класс A: «та ли это услуга с этим
свойством» решает выбор строки (`select_price_rows(..., judge_qualifier=True)`, правило 3 промпта отсекает «по ОМС»).

Инварианты класса (через обработчик эндпоинта, мини-прод, разбор хода — единственный режим, оракул LLM):
- LLM «не та» — без цены («не нашёл»), в том числе при `p2_llm_judge = off` в конфиге; LLM выбрала строку — цена;
- LLM недоступна — ответ тот же, что без судьи (не делаем хуже при сбое);
- разговорные формулировки свипа судит LLM, не словарь: выбрала строку — цена;
- реплики вне класса (без уточнения П2, без сущности) не меняются — судья не зовётся;
- два хода: «а сколько стоит по омс?» после цены — услуга из контекста — тоже судит LLM; короткий вопрос после
  «не нашёл» не получает цену того, чего пациент не видел.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import prices
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
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


class _NotItOnP2(_Oracle):
    """Выбирает строки `_PICK` на репликах без «омс»; на реплике с «омс» — «подходящих нет» (судья П2 отвечает «не та»).
    Под политикой L-13 (единственная с 16.10) первый ход двух — выбор LLM, а не ответ правил."""

    def __init__(self) -> None:
        super().__init__((), _PICK)

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        question = re.search(r"пациента: «(.*?)»", prompt, re.S)
        self.rows = ("—",) if question and "омс" in question.group(1).lower() else _PICK
        return await super().generate(prompt=prompt, system=system, **kwargs)


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
        "not_it_p2": _NotItOnP2,
        "down": _Down,
    }[kind]()


_WITH_JUDGE = prices._p2_name_from_entity


def _run(turns, world, monkeypatch, *, judge: bool, llm: str, session: str) -> tuple[list[dict], list[dict], object]:
    """`judge=False` — ход без судьи (прежний ответ): название из сущности судье не отдаётся."""

    _deploy(monkeypatch)
    monkeypatch.setattr(prices, "_p2_name_from_entity", _WITH_JUDGE if judge else (lambda *_args, **_kwargs: False))
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


def test_old_off_key_in_the_config_changes_nothing(mini_prod_world, monkeypatch):  # noqa: F811
    """Прод держит ключ до деплоя 6: `p2_llm_judge = off` в конфиге не читается — судья судит, «не та» — без цены."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_P2_LLM_JUDGE", "off", raising=False)
    [turn], [row], client = _run([_P2], mini_prod_world, monkeypatch, judge=True, llm="not_it", session="p2llm-old-off")
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert "rows" in client.asked and row["p2_llm_judge"] == "not_found"


@pytest.mark.parametrize("llm", ["down"])
def test_without_the_judge_the_entity_prices(llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Эталон сравнений — ход без судьи (было до 15.10): цена УЗИ по сущности, выбор строки LLM не спрашивается.

    Только при недоступной LLM: с 20.10 (деплой 11; выключатель `price_candidates_by_meaning` снят 22.10) словарный гард П2
    не останавливает вопрос до LLM — «по омс» судит выбор строки и без судьи сущности, так что «без судьи» с работающей LLM
    на проде не бывает."""

    [turn], [row], client = _run([_P2], mini_prod_world, monkeypatch, judge=False, llm=llm, session=f"p2llm-off-{llm}")
    assert _USG in turn["text"], turn["text"]
    assert "rows" not in getattr(client, "asked", [])
    assert "p2_llm_judge" not in row


def test_on_the_llm_says_not_it_and_there_is_no_price(mini_prod_world, monkeypatch):  # noqa: F811
    [turn], [row], client = _run([_P2], mini_prod_world, monkeypatch, judge=True, llm="not_it", session="p2llm-not")
    assert not _priced(turn["text"]) and "Не нашёл" in turn["text"], turn["text"]
    assert "rows" in client.asked and row["p2_llm_judge"] == "not_found"


def test_on_the_llm_picks_the_row_and_it_is_priced(mini_prod_world, monkeypatch):  # noqa: F811
    [turn], [row], _client = _run([_P2], mini_prod_world, monkeypatch, judge=True, llm="pick", session="p2llm-pick")
    assert _USG in turn["text"], turn["text"]
    assert row["p2_llm_judge"] == "rows"


@pytest.mark.parametrize("reply", (_P2, *_CONVERSATIONAL))
def test_on_with_the_llm_down_answers_as_off(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM — ответ как без судьи (цена по сущности): не делаем хуже при сбое."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge=False, llm="down", session=f"p2llm-d-off-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge=True, llm="down", session=f"p2llm-d-on-{reply[-9:]}")
    assert on["text"] == off["text"] and _priced(on["text"]), on["text"]
    assert row["p2_llm_judge"] == "fallback"


@pytest.mark.parametrize("reply", _CONVERSATIONAL)
def test_a_conversational_word_is_judged_by_the_llm_not_by_the_dictionary(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный инвариант (класс `guard_over_blocks_silently`): «первый», «для иностранного гражданина» гард по
    буквам считал уточнением П2. Их судит LLM: выбрала строку приёма — цена, как без судьи; решение —
    за LLM (тот же ход с ответом «не та» — без цены), а не за словарём каталога."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge=False, llm="pick", session=f"p2llm-c-off-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge=True, llm="pick", session=f"p2llm-c-on-{reply[-9:]}")
    assert _priced(off["text"]) and _priced(on["text"]), on["text"]
    assert row["p2_llm_judge"] == "rows"
    [not_it], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge=True, llm="not_it", session=f"p2llm-c-not-{reply[-9:]}")
    assert not _priced(not_it["text"]) and row["p2_llm_judge"] == "not_found", not_it["text"]


@pytest.mark.parametrize("reply", _OUTSIDE_THE_CLASS)
def test_on_does_not_touch_replies_outside_the_class(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Без уточнения П2 или без названия из сущности — ответ тот же, что без судьи, судья не зовётся."""

    [off], _rows, _ = _run([reply], mini_prod_world, monkeypatch, judge=False, llm="not_it", session=f"p2llm-o-off-{reply[-9:]}")
    [on], [row], _ = _run([reply], mini_prod_world, monkeypatch, judge=True, llm="not_it", session=f"p2llm-o-on-{reply[-9:]}")
    assert on["text"] == off["text"], on["text"]
    assert "p2_llm_judge" not in row


_TWO_TURNS = ("сколько стоит узи щитовидной железы", "а сколько стоит по омс?")


@pytest.mark.parametrize("llm", ["not_it_p2", "pick", "down"])
@pytest.mark.parametrize("judge", [False, True], ids=["no_judge", "judge"])
def test_two_turns_the_p2_follow_up_on_the_context_service(judge, llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Второй ход с уточнением П2 без названия после цены услуги — «а сколько стоит по омс?» не получает цену ни при
    каком положении судьи (П2). С 16.10 (L-13 — единственная политика): цена, выбранная LLM, услугой контекста не
    становится — второй ход «Уточните» (П2 соблюдён раньше судьи); LLM недоступна — первый ответ — список правил, а
    второй — как без судьи (сбой судьи не обходит политику L-13)."""

    (first, second), _rows, _ = _run(
        _TWO_TURNS, mini_prod_world, monkeypatch, judge=judge, llm=llm, session=f"p2llm-two-{judge}-{llm}"
    )
    assert _priced(first["text"]), first["text"]
    assert not _priced(second["text"]), second["text"]
    if llm == "down" and judge:
        (_first_off, second_off), _rows, _ = _run(
            _TWO_TURNS, mini_prod_world, monkeypatch, judge=False, llm=llm, session=f"p2llm-two-cmp-{llm}"
        )
        assert second["text"] == second_off["text"], (second["text"], second_off["text"])


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_not_found_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    """После «не нашёл» короткий вопрос («а сколько?», «да», цифра) не получает цену услуги из сущности."""

    (first, second), _rows, _ = _run(
        [_P2, follow_up], mini_prod_world, monkeypatch, judge=True, llm="not_it", session=f"p2llm-fu-{follow_up}"
    )
    assert not _priced(first["text"]), first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
