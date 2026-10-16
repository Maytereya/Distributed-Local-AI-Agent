"""Читатели разбора реплики на фразы под `turn_parse = on` (L-15b, шаг 5) — через обработчик эндпоинта, мини-прод.

Под `on` одна услуга или несколько — это разбор хода (данные МИС или выбор LLM из фраз кода, шаг 4), и
его читают:
- понимание хода: услуги — фразы разбора; его читатели (`acting_list()` — планировщик, подстановка
  каталога, грундер, дозаполнение) идут за разбором, а не за разрезом правил;
- корзина (`prices._multi_price_payload`): позиции — фразы разбора; сильная — её строки по данным МИС без
  LLM; слабые — один выбор строк корзины (`choose()`); фрагменты вне выбора — «Не распознал»; разбор
  решил одну услугу — путь одной услуги;
- путь одной услуги: одна сильная фраза — её строки без LLM, а не ответ правил по буквам;
- срок готовности (`_readiness_select`): одна фраза, сильная по данным, — срок её строк кодом без LLM; одна
  слабая — выбор LLM по кандидатам этой фразы; список или ни одной услуги — честный перевод;
- подбор анализов (`lab_tests.test_assist`): список — фразы разбора; одна услуга — путь одной услуги;
- разбор читают, только если вход читателя — реплика хода (`for_text`), и только решённый; иначе —
  прежний путь правил;
- политика сбоя (поправка владельца 13.10): LLM не разобрала реплику — разрез по разделителям (б), если он
  есть и одно название прайса не покрывает реплику; сплошной текст без LLM не режется (Q2) — прежний путь
  правил. LLM не выбрала строки корзины — сильные позиции с ценой по данным, слабая — несколько вариантов
  кода (до 5) с полными названиями и ценами; одна найденная строка — её название без цены (вариант B L-13);
- журнал синонимов для клиники: фраза — выбранная LLM разбора, а не вырезатель правил; разбор «ни одна фраза
  не услуга» — ничего; слабые позиции корзины и срока — с выбором LLM или «не нашла», поле `source`
  (`price | basket | readiness`) — только под `on`.

Разбор реплики — единственный режим (T7 шаг 3б, 16.10): при недоступной LLM путь цены отвечает по базовому
эталону `test_price_turns_golden.py` (прежние отличия разбора, `price_turns_parse_on.json`, вошли в него).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_parse
from tests.test_price_one_or_many_class import CASES, _is_basket_section
from tests.test_price_turns_golden import (  # noqa: F401 — mini_prod_world, synonym_fork — фикстуры
    DIALOGS,
    _golden,
    mini_prod_world,
    run_dialog,
    synonym_fork,
)
from tests.test_turn_parse_shadow import _deploy, _PhrasesOracle

_PHRASES_MARK = "понять, какие услуги назвал пациент"
_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"


def _parse_on(monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", "on", raising=False)


# --------------------------------------------------------------------------- эталон под on, LLM недоступна


@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_parse_with_llm_down_is_the_golden(case, mini_prod_world, request, monkeypatch):  # noqa: F811
    """Разбор реплики — единственный режим (T7 шаг 3б): при недоступной LLM и понимании в тени — базовый эталон пути
    цены (прежние отличия разбора, `price_turns_parse_on.json`, вошли в него), разбор не опаздывает (`late`)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_UNDERSTANDING_PRICE", "shadow", raising=False)
    if case.startswith("synonym_fork"):
        request.getfixturevalue("synonym_fork")
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        record = run_dialog(DIALOGS[case], f"parse-on-down-{case}")
        rows, attempts = list(world.telemetry), list(world.net.attempts)
    assert attempts == []
    assert "late" not in {row.get("tu_parse_status") for row in rows}
    assert record == _golden()[case]


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
        assert f"По услуге «{name}»:\n1. {name} — {cost} руб." in turn["text"]
    assert llm.asked(_BASKET_MARK) == [] and [entry["task"] for entry in row["choose"]] == ["turn_phrases"]


def test_one_service_phrase_the_rules_cut_is_not_a_basket(mini_prod_world, monkeypatch):  # noqa: F811
    """«мазок на микрофлору из зева» сегментация правил режет на три позиции; разбор — одна услуга: не
    корзина, строку выбирает выбор цены (второе решение хода)."""

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("phrase_smear_pharynx")
    turn, row = _turn(CASES["phrase_smear_pharynx"].text, llm, mini_prod_world, monkeypatch, "readers-smear")
    assert (row["tu_rule"], row["tu_list"]) == ("parse_picked", 0)
    assert not any(_is_basket_section(line) for line in turn["text"].splitlines()) and llm.asked(_BASKET_MARK) == []
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


def _run_with(monkeypatch, parse: turn_parse.PhraseParse, call, *, mode: str = "on"):
    """`await call()` в ходе, где слот разбора уже держит `parse` (выключатель — `mode`)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", mode, raising=False)

    async def run():
        turn_parse.begin()

        async def factory() -> turn_parse.PhraseParse:
            return parse

        turn_parse.start_phrases(factory, reply=parse.reply)
        await turn_parse.phrases()
        return await call()

    return asyncio.run(run())


def _slot(monkeypatch, mode: str, parse: turn_parse.PhraseParse, probe: str):
    async def read():
        return turn_parse.acting_for(probe)

    return _run_with(monkeypatch, parse, read, mode=mode)


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


# --------------------------------------------------------------------------- срок и подбор анализов читают разбор


@pytest.fixture(scope="module")
def catalog(mini_prod_world):  # noqa: F811
    from messengers_router.services._prices_helpers import _catalog_rows

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        return _catalog_rows(None)


def _phrase(text: str, reply: str, catalog: list[dict] | None = None, name: str = "") -> turn_parse.ParsedPhrase:
    from messengers_router.choice import Strength

    rows = tuple(row for row in catalog or () if row["serviceName"] == name)
    start = reply.index(text)
    return turn_parse.ParsedPhrase(text=text, start=start, end=start + len(text), strength=Strength.STRONG if rows else Strength.WEAK, rows=rows)


def _parsed(reply: str, *phrases: turn_parse.ParsedPhrase, status: str = "picked") -> turn_parse.PhraseParse:
    return turn_parse.PhraseParse(reply=reply, status=status, source="llm", items=phrases)


_REPLY_DAYS = "когда будет готов анализ на ферритин"


def test_readiness_takes_the_deadline_of_the_one_strong_phrase(catalog, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.services._readiness_select import readiness_from_catalog

    one = _parsed(_REPLY_DAYS, _phrase("ферритин", _REPLY_DAYS, catalog, "Ферритин"))
    two = _parsed(_REPLY_DAYS, _phrase("анализ", _REPLY_DAYS), _phrase("ферритин", _REPLY_DAYS, catalog, "Ферритин"))
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        by_rules = readiness_from_catalog(_REPLY_DAYS, rows=catalog)
        by_one = _run_with(monkeypatch, one, lambda: asyncio.to_thread(readiness_from_catalog, _REPLY_DAYS, rows=catalog))
        by_list = _run_with(monkeypatch, two, lambda: asyncio.to_thread(readiness_from_catalog, _REPLY_DAYS, rows=catalog))
    assert by_one == by_rules and [item["name"] for item in by_one] == ["Ферритин"]
    assert by_list is None  # разбор — список: срок не одной услуги, перевод


def test_readiness_choice_reads_the_phrase_and_skips_what_data_decided(catalog, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.choice import Choice, Status
    from messengers_router.services import _readiness_select as RS

    asked: list[list[str]] = []

    async def fake_choose(_task, _question, options):
        asked.append([option.id for option in options])
        return Choice(Status.PICKED, ids=("Ферритин",))

    monkeypatch.setattr(RS, "choose", fake_choose)
    weak = _parsed(_REPLY_DAYS, _phrase("анализ на ферритин", _REPLY_DAYS))
    glucose = "сколько готовится глюкоза"
    by_data = _parsed(glucose, _phrase("глюкоза", glucose, catalog, "Глюкоза"))  # одна услуга по данным — срок решает код
    listed = _parsed(_REPLY_DAYS, _phrase("анализ", _REPLY_DAYS), _phrase("ферритин", _REPLY_DAYS, catalog, "Ферритин"))
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        found = _run_with(monkeypatch, weak, lambda: RS.readiness_by_choice(_REPLY_DAYS, rows=catalog))
        assert [item["name"] for item in found] == ["Ферритин"] and "Ферритин" in asked[-1]
        calls = len(asked)
        assert _run_with(monkeypatch, by_data, lambda: RS.readiness_by_choice(glucose, rows=catalog)) is None
        assert _run_with(monkeypatch, listed, lambda: RS.readiness_by_choice(_REPLY_DAYS, rows=catalog)) is None
    assert len(asked) == calls  # одна услуга по данным и список — без выбора LLM (соседняя строка срока не даст)


def test_readiness_of_a_run_on_list_is_a_handoff_not_one_deadline(mini_prod_world, monkeypatch):  # noqa: F811
    """«сколько дней делается ФЕРРИТИН ГЛЮКОЗА» правила списком не видят; разбор — две фразы: срок не одной
    услуги — честный перевод, без выбора срока (один срок на две услуги был бы сроком одной из них)."""

    from messengers_router.services.lab_tests import RESULT_TIMING_OPERATOR_TEXT

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    llm = _Recorder("list_caps_run_on_priced", cut=("ФЕРРИТИН", "ГЛЮКОЗА"))
    turn, row = _turn("сколько дней делается ФЕРРИТИН ГЛЮКОЗА", llm, mini_prod_world, monkeypatch, "readers-days-list")
    assert (row["tu_parse_status"], row["tu_parse_n"]) == ("picked", 2)
    assert turn["text"] == RESULT_TIMING_OPERATOR_TEXT
    assert [entry["task"] for entry in row["choose"]] == ["turn_phrases"]


def test_test_assist_list_is_the_parse(catalog, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.services import lab_tests

    run_on = "ферритин глюкоза"  # правилам не список: слов меньше четырёх и нет разделителей
    split = "сколько стоит ферритин, глюкоза"  # правилам — список; вопрос о цене
    bare = "ферритин, глюкоза"  # правилам — список; вопроса о цене нет
    as_list = _parsed(run_on, _phrase("ферритин", run_on, catalog, "Ферритин"), _phrase("глюкоза", run_on, catalog, "Глюкоза"))
    as_one = _parsed(split, _phrase("ферритин, глюкоза", split), status="single")
    bare_one = _parsed(bare, _phrase(bare, bare), status="single")
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        listed = _run_with(monkeypatch, as_list, lambda: lab_tests.test_assist(None, run_on, {}))
        one = _run_with(monkeypatch, as_one, lambda: lab_tests.test_assist(None, split, {}))
        not_price = _run_with(monkeypatch, bare_one, lambda: lab_tests.test_assist(None, bare, {}))
        rules = asyncio.run(lab_tests.test_assist(None, split, {}))  # вне хода — разрез правил
    assert "multi-item" in listed["note"] and [row["serviceName"] for row in listed["tests"]] == ["Ферритин", "Глюкоза"]
    assert "multi-item" not in one["note"] and "multi-item" in rules["note"]
    # Разбор «одна услуга» без вопроса о цене — не мнение о списке: сегментация правил, как под off (T3, п. 2).
    assert "multi-item" in not_price["note"]


# --------------------------------------------------------------------------- политика сбоя под on


def test_failed_parse_keeps_only_the_separator_cut(catalog, mini_prod_world):  # noqa: F811
    """LLM не ответила: у разбора нет фраз, а для читателей — разрез по разделителям (б) со силой по данным. Без
    разделителей (решение владельца 16.10, Q2 отменено) — разрез данными МИС (сильные окна), если в промежутках нет слова
    вне прайса; иначе — реплика целиком одной фразой; одно название прайса на всю реплику — тоже целиком."""

    from messengers_router.services import _turn_phrases as TP

    async def parse(reply: str) -> turn_parse.PhraseParse:
        return await TP.parse_turn_phrases(reply, facet="price", catalog=catalog, shadow=False)

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        listed = asyncio.run(parse("сколько стоит ферритин, ттг и витамин д"))
        run_on = asyncio.run(parse("сколько стоит ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА"))
        caps = asyncio.run(parse("сколько стоит ФЕРРИТИН ГЛЮКОЗА ХОЛЕСТЕРОЛ"))
        one_name = asyncio.run(parse("сколько стоит билирубин и фракции"))
    assert (listed.status, listed.items) == ("unavailable", ())
    assert [(p.text, p.strength.value, bool(p.rows)) for p in listed.split] == [
        ("ферритин", "strong", True),
        ("ттг", "weak", False),
        ("витамин д", "weak", False),
    ]
    # «ОАК ОАМ» без синонимов МИС в мини-проде — слова вне прайса в промежутке: без LLM не отличить от условия П2.
    assert run_on.status == "unavailable" and [p.text for p in run_on.split] == ["ОАК ОАМ ФЕРРИТИН ГЛЮКОЗА"]
    assert caps.status == "unavailable" and [(p.text, bool(p.rows)) for p in caps.split] == [
        ("ФЕРРИТИН", True),
        ("ГЛЮКОЗА", True),
        ("ХОЛЕСТЕРОЛ", True),
    ]
    assert one_name.status == "unavailable" and [p.text for p in one_name.split] == ["билирубин и фракции"]


def test_failed_parse_is_read_as_its_separator_cut(monkeypatch):
    reply = "сколько стоит ферритин и глюкоза"
    split = (_phrase("ферритин", reply), _phrase("глюкоза", reply))
    failed = turn_parse.PhraseParse(reply=reply, status="unavailable", source="llm", leftover=("ферритин", "глюкоза"), split=split)
    view = _slot(monkeypatch, "on", failed, reply)
    assert (view.status, view.items, view.leftover, view.is_list) == ("unavailable", split, (), True)
    assert not view.decided  # статус сбоя виден читателю
    # Одна фраза сбоя (реплика целиком, Q2 отменено 16.10) — тоже действует: путь одной услуги, а не правила.
    single = _slot(monkeypatch, "on", turn_parse.PhraseParse(reply=reply, status="invalid", source="llm", split=split[:1]), reply)
    assert single is not None and single.items == split[:1] and not single.is_list
    assert _slot(monkeypatch, "shadow", failed, reply) is None


def test_weak_positions_without_llm_are_variants_not_one_price(mini_prod_world, monkeypatch):  # noqa: F811
    """LLM недоступна: сильная позиция — её строка по данным; слабые («ттг», «витамин д») — несколько вариантов
    кода с полными названиями и ценами, а не одна строка правил как «та самая» (поправка владельца 13.10)."""

    from tests.test_price_one_or_many_class import _priced_sections, _violations

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog([CASES["list_true_three"].text], "readers-down-variants")
        row = world.take_telemetry()
    assert row["tu_parse_status"] == "unavailable" and row["tu_rule"] == "parse_unavailable"
    assert _violations(CASES["list_true_three"], turn["text"], "down") == []
    sections = _priced_sections(turn["text"])
    assert sections[0] == ["Ферритин"] and all(2 <= len(section) <= 5 for section in sections[1:]), sections


def test_one_found_row_without_llm_is_named_without_its_price():
    """Без выбора LLM у слабой позиции нашлась одна строка — название без цены (вариант B L-13), ни одной
    позиции с ценой — только названия."""

    from messengers_router.renderer import format_price_for_patient
    from messengers_router.services._basket_select import basket_from_phrases
    from messengers_router.services._prices_helpers import multi_price_payload_from_items
    from messengers_router.text_templates import price_names_without_prices_text

    catalog = [
        {"serviceName": "Ферритин", "serviceHomecode": "1", "cost": 530},
        {"serviceName": "Спирография компьютерная", "serviceHomecode": "2", "cost": 900},
        {"serviceName": "Глюкоза", "serviceHomecode": "3", "cost": 190},
    ]
    note = price_names_without_prices_text("спирография", ["Спирография компьютерная"])

    def answer(reply: str, *phrases: turn_parse.ParsedPhrase) -> dict:
        positions, rest = asyncio.run(basket_from_phrases(reply, phrases, (), catalog, llm_allowed=False))
        return multi_price_payload_from_items(positions, rest, allow_single=True)

    reply = "сколько стоит ферритин и спирография"
    payload = answer(reply, _phrase("ферритин", reply, catalog, "Ферритин"), _phrase("спирография", reply))
    text = format_price_for_patient(payload, {})
    assert "Ферритин — 530" in text and note in text and "900" not in text
    reply = "сколько стоит спирография и сахар"
    payload = answer(reply, _phrase("спирография", reply), _phrase("сахар", reply))
    assert (payload["prices"], payload["clarify_text"]) == ([], f"{note}\n\nНе распознал: сахар. Уточните эти названия — помогу по ним отдельно.")


# --------------------------------------------------------------------------- журнал синонимов под on


def _journal_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _JournalLLM:
    """Выбор фраз — ровно эти фразы; строки корзины — те, чьё название начинается с `rows_prefix`; прочие
    места — «недоступна»."""

    def __init__(self, phrases: tuple[str, ...], rows_prefix: str) -> None:
        from messengers_router.services._common import _normalise_catalog_text

        self.norm = _normalise_catalog_text
        self.phrases = {_normalise_catalog_text(text) for text in phrases}
        self.rows_prefix = rows_prefix

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _PHRASES_MARK in seen:
            picks = [n for n, text in options if self.norm(text.strip("«»")) in self.phrases]
        elif _BASKET_MARK in seen:
            picks = [n for n, text in options if text.split(" → ", 1)[-1].startswith(self.rows_prefix)]
        else:
            raise ConnectionError("тест: LLM недоступна для этого места")
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


_LH_AND_GGT = "сколько стоит лютеинизирующий гормон и ггтп"


@pytest.mark.parametrize("mode", ["off", "on"])
def test_basket_journals_weak_phrases_with_what_the_llm_picked(mode, mini_prod_world, monkeypatch):  # noqa: F811
    """Под `on` слабые позиции корзины — фразы, которые выбрала LLM разбора, — в журнал синонимов с выбором
    строк LLM («не нашла» — `found: false`), `source: basket`. Под `off` корзина журнал не пишет, как раньше."""

    _deploy(monkeypatch)
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_TURN_PARSE", mode, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _JournalLLM(("лютеинизирующий гормон", "ггтп"), "ЛГ (LH)"))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog([_LH_AND_GGT], f"journal-basket-{mode}")
        entries = _journal_rows(world._journal)
    if mode == "off":
        assert entries == []
        return
    assert [(e["phrase"], e["found"], e["source"], [s["name"] for s in e["services"]]) for e in entries] == [
        ("лютеинизирующий гормон", True, "basket", ["ЛГ (LH) Лютеинизирующий гормон"]),
        ("ггтп", False, "basket", []),
    ]
    assert "ЛГ (LH) Лютеинизирующий гормон —" in turn["text"] and "Не распознал: ггтп" in turn["text"]


def _price_miss(monkeypatch, catalog, question: str, parse: turn_parse.PhraseParse | None, journal: Path) -> list[dict]:
    """`select_price_rows`, где правила кандидатов не дали, а LLM (прежний промпт) не нашла ничего."""

    from messengers_router.services import _price_select_llm as PS

    monkeypatch.setattr(PS, "_ENABLED", True)
    monkeypatch.setattr(PS, "_journal_path", lambda: journal)

    async def nothing(_question, _pool):
        return [], [], ""

    monkeypatch.setattr(PS, "_ask", nothing)

    async def select():
        return await PS.select_price_rows(question, catalog)

    if parse is None:
        asyncio.run(select())
    else:
        _run_with(monkeypatch, parse, select)
    return _journal_rows(journal)


def test_price_journal_takes_the_phrase_of_the_llm_parse(catalog, tmp_path, monkeypatch):
    """Под `on` фраза в журнал — выбранная LLM разбора (срез реплики), `source: price`; разбор «ни одна фраза не
    услуга» (`none_fit`) — в журнал ничего: «Цена анализа» клинике не синоним. Вне разбора — как раньше."""

    question = "Цена анализа"
    assert [e["phrase"] for e in _price_miss(monkeypatch, catalog, question, None, tmp_path / "off.jsonl")] == ["Цена анализа"]
    none_fit = turn_parse.PhraseParse(reply=question, status="none_fit", source="llm")
    assert _price_miss(monkeypatch, catalog, question, none_fit, tmp_path / "none_fit.jsonl") == []

    question = "а подскажите спирометрия почем"
    picked = _parsed(question, _phrase("спирометрия", question))
    [entry] = _price_miss(monkeypatch, catalog, question, picked, tmp_path / "picked.jsonl")
    assert (entry["phrase"], entry["found"], entry["source"]) == ("спирометрия", False, "price")
    [legacy] = _price_miss(monkeypatch, catalog, question, None, tmp_path / "legacy.jsonl")
    assert "source" not in legacy  # вне `on` строка журнала прежняя


def test_readiness_journals_the_weak_phrase_with_the_llm_choice(catalog, mini_prod_world, monkeypatch):  # noqa: F811
    from messengers_router.choice import Choice, Status
    from messengers_router.services import _readiness_select as RS

    answers = iter([Choice(Status.PICKED, ids=("Ферритин",)), Choice(Status.NONE_FIT), Choice(Status.UNAVAILABLE)])

    async def fake_choose(_task, _question, _options):
        return next(answers)

    monkeypatch.setattr(RS, "choose", fake_choose)
    weak = _parsed(_REPLY_DAYS, _phrase("анализ на ферритин", _REPLY_DAYS))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:  # журнал синонимов мира — во временном каталоге
        for _ in range(3):
            _run_with(monkeypatch, weak, lambda: RS.readiness_by_choice(_REPLY_DAYS, rows=catalog))
        entries = _journal_rows(world._journal)
    assert [(e["phrase"], e["found"], e["source"], [s["name"] for s in e["services"]]) for e in entries] == [
        ("анализ на ферритин", True, "readiness", ["Ферритин"]),
        ("анализ на ферритин", False, "readiness", []),
    ]  # сбой LLM — решения нет, записи нет


@pytest.mark.parametrize("llm", ["down", "live"])
def test_phrase_with_a_qualifier_the_price_lacks_is_not_priced_by_neighbours(llm, mini_prod_world, monkeypatch):  # noqa: F811
    """Решение владельца П2 («по ОМС» — честное «не нашёл», не платный приём) держится и у позиции корзины из
    разбора: ни варианты кода при сбое LLM, ни выбор строк при живой LLM не показывают платные приёмы."""

    _deploy(monkeypatch)
    _parse_on(monkeypatch)
    if llm == "live":
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _JournalLLM(("ферритин", "приём терапевта по ОМС"), "Прием"))
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        [turn] = run_dialog(["сколько стоит ферритин и приём терапевта по ОМС"], f"readers-oms-{llm}")
        row = world.take_telemetry()
    assert "Ферритин — 530" in turn["text"] and "Не распознал: приём терапевта по ОМС" in turn["text"]
    assert "врача-терапевта —" not in turn["text"] and "price_basket" not in [entry["task"] for entry in row["choose"]]
