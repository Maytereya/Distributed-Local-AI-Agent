"""Читатели разбора реплики на фразы под `turn_parse = on` (L-15b, шаг 5) — через обработчик эндпоинта, мини-прод.

Под `on` одна услуга или несколько — это разбор хода (данные МИС или выбор LLM из фраз кода, шаг 4), и
его читают:
- понимание хода: услуги — фразы разбора; его читатели (`acting_list()` — планировщик, подстановка
  каталога, грундер, дозаполнение) идут за разбором, а не за разрезом правил;
- корзина (`prices._multi_price_payload`): позиции — фразы разбора; сильная — её строки по данным МИС без
  LLM; слабые — один выбор строк корзины (`choose()`); фрагменты вне выбора — «Не распознал»; разбор
  решил одну услугу — путь одной услуги;
- путь одной услуги: одна сильная фраза — её строки без LLM, а не ответ правил по буквам;
- разбор читают, только если вход читателя — реплика хода (`for_text`), и только решённый; иначе —
  прежний путь правил.

Под `off` и `shadow` поведение прежнее: эталоны `test_price_turns_golden.py`, `test_turn_parse_shadow.py`.
Под `on` при недоступной LLM эталон пути цены — прежний, кроме диалогов `PARSE_ON_PLANNED` (их ответы — в
`golden/price_turns_parse_on.json`; перезапись — `GOLDEN_UPDATE=1`, дифф — в коммит с объяснением).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_parse
from tests.test_price_one_or_many_class import CASES
from tests.test_price_turns_golden import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    DIALOGS,
    UPDATE,
    _golden,
    _load,
    _store,
    mini_prod_world,
    run_dialog,
    synonym_fork,
)
from tests.test_turn_parse_shadow import _deploy, _PhrasesOracle

GOLDEN_PARSE_ON = Path(__file__).resolve().parent / "golden" / "price_turns_parse_on.json"

# Диалоги эталона, ответ которых под `turn_parse = on` (LLM недоступна, понимание хода в тени) другой — по
# решению владельца (STRONG — только данные МИС; одна услуга или несколько — разбор).
PARSE_ON_PLANNED: dict[str, str] = {
    "family_glucose": "«глюкоза» — дословное название: только строка «Глюкоза», без семьи («Глюкоза в моче», в синовиальной жидкости)",
    "list_text_and": "«ферритин и глюкоза» — обе позиции сильные по данным: у глюкозы только её строка, без «Глюкозы в моче»",
    "after_list_prepare": "первый ход — «ферритин, глюкоза» по данным, без «Глюкозы в моче»; второй — как раньше",
    "after_list_where": "первый ход — «ферритин, глюкоза» по данным, без «Глюкозы в моче»; второй — как раньше",
    "name_with_commas": "дословное название с запятыми в скобках — одна строка, а не семья ОАК с ценой второй услуги",
    "name_with_plus": "дословное название с «+» и дефисом — одна строка, а не 10 экспресс-тестов и ГТТ с инсулином",
    "synonym_fork_oak": "синоним МИС «оак» на три услуги — ровно эти три строки, а не семья ОАК из восьми",
}

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"


def _parse_on(monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", "on", raising=False)


# --------------------------------------------------------------------------- эталон под on, LLM недоступна


@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_parse_on_with_llm_down_changes_only_planned_dialogs(case, mini_prod_world, request, monkeypatch):  # noqa: F811
    _parse_on(monkeypatch)
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = run_dialog(DIALOGS[case], f"parse-on-down-{case}")
        rows, attempts = list(world.telemetry), list(world.net.attempts)
    assert attempts == []
    assert "late" not in {row.get("tu_parse_status") for row in rows}
    base = _golden()[case]
    if UPDATE:
        _store(GOLDEN_PARSE_ON, case, None if record == base else record)
        return
    assert record == _load(GOLDEN_PARSE_ON).get(case, base)


def test_parse_on_golden_holds_only_planned_dialogs():
    assert set(_load(GOLDEN_PARSE_ON)) == set(PARSE_ON_PLANNED)


# --------------------------------------------------------------------------- понимание и корзина читают разбор


class _Recorder(_PhrasesOracle):
    """Оракул класса (фразы — разрез случая, строки — разметка) и запись того, что спрашивали у LLM."""

    def __init__(self, name: str, cut: tuple[str, ...] | None = None) -> None:
        super().__init__(name)
        if cut is not None:
            from messengers_router.services._common import _normalise_catalog_text

            self.cut = {_normalise_catalog_text(text) for text in cut}
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        self.prompts.append(f"{system}\n\n{prompt}" if system else prompt)
        return await super().generate(prompt=prompt, system=system, **kwargs)

    def asked(self, mark: str) -> list[str]:
        return [prompt for prompt in self.prompts if mark in prompt]


def _turn(text: str, llm, world_dir, monkeypatch, session: str):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([text], session)
        row = world.take_telemetry()
    return turn, row


def test_run_on_list_the_rules_missed_is_a_basket_by_the_parse(mini_prod_world, monkeypatch):  # noqa: F811
    """«ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ» правила списком не видят (слов меньше четырёх), разбор LLM — видит: понимание
    — список фраз, планировщик отдаёт его корзине, позиции сильные — строки по данным без выбора строк."""

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("list_caps_run_on_priced")
    turn, row = _turn(CASES["list_caps_run_on_priced"].text, llm, mini_prod_world, monkeypatch, "readers-run-on")
    assert (row["tu_rule"], row["tu_list"], row["tu_parse_status"]) == ("parse_picked", 3, "picked")
    assert "planner_list_to_cart" in row["tu_acted"]
    for name, cost in (("Ферритин", "530"), ("Глюкоза", "190"), ("Холестерол", "225")):
        assert f"Формат: По услуге «{name}»:\n1. {name} — {cost} руб." in turn["text"]
    assert llm.asked(_BASKET_MARK) == [] and [entry["task"] for entry in row["choose"]] == ["turn_phrases"]


def test_one_service_phrase_the_rules_cut_is_not_a_basket(mini_prod_world, monkeypatch):  # noqa: F811
    """«мазок на микрофлору из зева» сегментация правил режет на три позиции; разбор — одна услуга: не
    корзина, строку выбирает выбор цены (второе решение хода)."""

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("phrase_smear_pharynx")
    turn, row = _turn(CASES["phrase_smear_pharynx"].text, llm, mini_prod_world, monkeypatch, "readers-smear")
    assert (row["tu_rule"], row["tu_list"]) == ("parse_picked", 0)
    assert "Формат: По услуге" not in turn["text"] and llm.asked(_BASKET_MARK) == []
    assert "Посев на микрофлору (идентификация агента и чувствительность к а\\б) — 1 120" in turn["text"]


def test_basket_asks_rows_only_for_weak_phrases(mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("list_true_three")
    turn, row = _turn(CASES["list_true_three"].text, llm, mini_prod_world, monkeypatch, "readers-weak")
    [basket] = llm.asked(_BASKET_MARK)
    assert "«ттг» →" in basket and "«витамин д» →" in basket and "«ферритин» →" not in basket
    assert [entry["task"] for entry in row["choose"]] == ["turn_phrases", "price_basket"]
    assert "Ферритин — 530" in turn["text"] and "ТТГ (TSH) тиреотропный гормон — 400" in turn["text"]


def test_fragment_outside_the_picked_phrases_is_named_unrecognized(mini_prod_world, monkeypatch):  # noqa: F811
    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("list_one_unrecognized", cut=("ферритин", "глюкоза"))  # «спирометрия» LLM услугой не сочла
    turn, row = _turn(CASES["list_one_unrecognized"].text, llm, mini_prod_world, monkeypatch, "readers-leftover")
    assert row["tu_parse_n"] == 2 and llm.asked(_BASKET_MARK) == []
    assert "Ферритин — 530" in turn["text"] and "Глюкоза — 190" in turn["text"] and "в моче" not in turn["text"]
    assert "Не распознал: спирометрия" in turn["text"]


def test_one_strong_phrase_is_priced_by_data_without_llm(mini_prod_world, monkeypatch):  # noqa: F811
    """Дословное название — строки по данным МИС, без выбора строки и без ответа правил с соседями (у ОАК —
    вторая услуга семьи с ценой)."""

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("name_commas_in_parens")
    turn, row = _turn(CASES["name_commas_in_parens"].text, llm, mini_prod_world, monkeypatch, "readers-strong")
    assert row["tu_parse_status"] == "strong_whole" and llm.prompts == []
    assert "Общий анализ крови (Le, Er, Hb, СОЭ) — 390" in turn["text"] and "(полный)" not in turn["text"]


# --------------------------------------------------------------------------- вход читателя и решённость


def _slot(monkeypatch, mode: str, parse: turn_parse.PhraseParse, probe: str):
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", mode, raising=False)

    async def run():
        turn_parse.begin()

        async def factory() -> turn_parse.PhraseParse:
            return parse

        turn_parse.start_phrases(factory, reply=parse.reply)
        await turn_parse.phrases()
        return turn_parse.acting_for(probe)

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("mode", "status", "probe", "acts"),
    [
        ("on", "picked", "сколько стоит ферритин и глюкоза", True),
        ("on", "picked", "ферритин и глюкоза", False),  # вход читателя — не реплика хода
        ("on", "unavailable", "сколько стоит ферритин и глюкоза", False),  # не решён — прежний путь
        ("shadow", "picked", "сколько стоит ферритин и глюкоза", False),
        ("off", "picked", "сколько стоит ферритин и глюкоза", False),
    ],
)
def test_parse_is_read_only_for_its_reply_decided_and_under_on(mode, status, probe, acts, monkeypatch):
    parse = turn_parse.PhraseParse(reply="сколько стоит ферритин и глюкоза", status=status, source="llm")
    assert (_slot(monkeypatch, mode, parse, probe) is parse) is acts

