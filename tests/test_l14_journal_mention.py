"""Журнал синонимов берёт фразу из упоминания хода — L-14, шаг 2 (`[MESSENGER_ROUTER] journal_phrase = mention`).

Было: фразу журнала резал вырезатель правил (`_extract_price_service_from_query`) — «а подскажите пожалуйста
сколько стоит ггтп сдать» → «а ггтп»; реплика без названия услуги («Цена анализа») уходила в журнал целиком
(вырезатель ничего не оставил — бралась реплика). Под `turn_parse = on` без выбора LLM (`single`) — то же.

Стало (под выключателем, по умолчанию `rules` — как было): фраза — упоминание одиночной услуги хода
(`TurnUnderstanding.single`): срез реплики без речевой обвязки с краёв; понимание реплики без упоминания —
ничего. Обвязка — существующие лингвистические списки (`_turn_phrases._WRAPPER`, `_biomaterial._SPEECH_WRAPPER`),
новых слов и регулярок нет. Разбор без выбора LLM (`single`) знает об услуге не больше правил — упоминание правил.

Инвариант класса: журнал (подсказка клинике) пишет то, что пациент назвал услугой, без слов вопроса вокруг; реплика,
в которой услуги не названо, в журнал не попадает; ответы пациенту не меняются.
"""

from __future__ import annotations

import pytest

from evalbench import world as ev_world
from messengers_router.services._common import _normalise_catalog_text
from tests.test_l14_two_turn_consumers import _FOUND_GGT, BASE_CONFIGS, _Scripted, _play
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура модуля

CONFIGS = BASE_CONFIGS


def _source(monkeypatch, value: str = "mention") -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_JOURNAL_PHRASE", value, raising=False)


def _phrases(turn) -> list[str]:
    return [_normalise_catalog_text(str(row["phrase"])) for row in turn.journal]


@pytest.mark.parametrize(("raw", "expected"), [(None, "rules"), ("rules", "rules"), ("mention", "mention"), ("MENTION", "mention"), ("x", "rules")])
def test_setting_is_read_with_a_safe_default(raw, expected, monkeypatch):
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services._price_select_llm import journal_phrase_mode

    if raw is None:
        monkeypatch.delattr(cfg, "MR_JOURNAL_PHRASE", raising=False)
    else:
        monkeypatch.setattr(cfg, "MR_JOURNAL_PHRASE", raw, raising=False)
    assert journal_phrase_mode() == expected


def test_config_default_is_rules():
    from agent_logic_2 import config

    assert config.MR_JOURNAL_PHRASE == "rules"


@pytest.mark.parametrize(
    ("raw", "bare"),
    [
        ("а ггтп", "ггтп"),
        ("а подскажите пожалуйста сколько стоит ггтп", "ггтп"),
        ("ггтп сдать", "ггтп"),
        ("анализ на ГГТП", "ГГТП"),  # регистр и знаки — как написал пациент
        ("общий анализ крови", "общий анализ крови"),  # внутри фразы не трогаем
        ("C-пептид", "C-пептид"),
        ("а", ""),
        ("сколько стоит", ""),
        ("", ""),
    ],
)
def test_bare_phrase_trims_only_speech_at_the_edges(raw, bare):
    from messengers_router.turn_understanding import _bare_phrase

    assert _bare_phrase(raw) == bare


def test_mention_phrase_is_the_patient_phrase_and_nothing_without_a_service(mini_prod_world):  # noqa: F811
    from messengers_router import turn_understanding as tu

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        term = tu.understand_price_turn("а подскажите пожалуйста сколько стоит ггтп сдать")
        bare = tu.understand_price_turn("а цена?")
        general = tu.understand_price_turn("Цена анализа")
        readiness = tu.understand_price_turn("а сколько готовится ггтп")
    assert term.single is not None and term.single.phrase == "ггтп"
    assert bare.services == () and general.services == ()
    assert readiness.facet == "readiness" and readiness.single is not None and readiness.single.phrase == "ггтп"


def test_journal_reads_only_the_understanding_of_this_reply(mini_prod_world, monkeypatch):  # noqa: F811
    """Вход читателя — не реплика хода (дозаполнение, контекст прошлого хода) — прежний путь, а не чужое упоминание."""

    from messengers_router import turn_understanding as tu
    from messengers_router.services._price_select_llm import _mention_phrase

    with ev_world.FrozenWorld(mini_prod_world, "day"):
        understood = tu.understand_price_turn("сколько стоит ггтп")
    token = tu._CURRENT.set(understood)
    try:
        assert _mention_phrase("сколько стоит ггтп") == "ггтп"
        assert _mention_phrase("сколько стоит ферритин") is None
    finally:
        tu._CURRENT.reset(token)
    assert _mention_phrase("сколько стоит ггтп") is None  # понимания нет — прежний путь


_UNKNOWN = lambda: _Scripted(cut=("спирометрия",), terms=("спирография",))  # noqa: E731 — слова нет ни в прайсе, ни у LLM


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize(
    ("said", "llm", "expected"),
    [
        ("а подскажите пожалуйста сколько стоит ггтп сдать", _FOUND_GGT, ["ггтп"]),
        ("а ггтп сколько стоит", _FOUND_GGT, ["ггтп"]),
        ("а сколько стоит спирометрия", _UNKNOWN, ["спирометрия"]),
        ("подскажите пожалуйста спирометрия сколько стоит сдать", _UNKNOWN, ["спирометрия"]),
        ("Цена анализа", _UNKNOWN, []),
        ("подскажите стоимость анализов", _UNKNOWN, []),
    ],
    ids=["ggt_sentence", "ggt_first", "unknown_a", "unknown_wrapped", "general_1", "general_2"],
)
def test_journal_writes_the_mention_phrase_through_the_endpoint(said, llm, expected, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Класс через обработчик эндпоинта: фраза журнала — то, что пациент назвал услугой, без слов вопроса; реплика
    без названия услуги — ничего. Ответ пациенту тот же, что под `rules`."""

    _source(monkeypatch)
    [turn] = _play([said], f"journal-mention-{said[:20]}", mini_prod_world, monkeypatch, config, llm=llm())
    assert _phrases(turn) == expected, turn.journal
    _source(monkeypatch, "rules")
    [before] = _play([said], f"journal-rules-{said[:20]}", mini_prod_world, monkeypatch, config, llm=llm())
    assert turn.text == before.text and turn.handoff == before.handoff


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("followup", ["а цена?", "сколько стоит?", "да"])
def test_two_turns_short_question_writes_nothing(followup, config, mini_prod_world, monkeypatch):  # noqa: F811
    """Два хода: после ответа о ГГТ (LLM по термину) короткий вопрос без названия услуги в журнал не попадает,
    первый ход пишет только «ггтп»."""

    _source(monkeypatch)
    first, second = _play(["а подскажите пожалуйста сколько стоит ггтп сдать", followup], f"journal-two-{followup}", mini_prod_world, monkeypatch, config, llm=_FOUND_GGT())
    assert _phrases(first) == ["ггтп"], first.journal
    assert second.journal == [], second.journal


_WRAPPED_BASKET = "сколько стоит лютеинизирующий гормон и а ггтп"
_WRAPPED_QUESTION = "Здравствуйте, а делаете ли вы лютеинизирующий гормон и ггтп?"


@pytest.mark.parametrize(
    ("said", "picked", "mention", "rules"),
    [
        (_WRAPPED_BASKET, ("лютеинизирующий гормон", "а ггтп"), ["лютеинизирующий гормон", "ггтп"], ["лютеинизирующий гормон", "а ггтп"]),
        (  # «делаете ли вы» нет в списках обвязки — остаётся: новых слов нет (как у одиночной услуги)
            _WRAPPED_QUESTION,
            ("а делаете ли вы лютеинизирующий гормон", "ггтп"),
            ["делаете ли вы лютеинизирующий гормон", "ггтп"],
            ["а делаете ли вы лютеинизирующий гормон", "ггтп"],
        ),
    ],
    ids=["edge_a", "edge_question"],
)
def test_basket_positions_are_journaled_like_a_single_mention(said, picked, mention, rules, mini_prod_world, monkeypatch):  # noqa: F811
    """Третий писатель журнала — слабые позиции корзины под `turn_parse = on` (L-15b, шаг 5): под `mention` их фраза —
    без речевой обвязки с краёв, как у одиночной услуги («а ггтп» → «ггтп»; находка вечернего dev 14.10: «а делаете ли
    вы оам» в журнале). Под `rules` — как было; ответ пациенту в обоих режимах один."""

    from messengers_router import llm_runtime
    from tests.test_price_turns_golden import run_dialog
    from tests.test_turn_parse_readers import _journal_rows, _JournalLLM
    from tests.test_turn_parse_slot import _deploy

    _deploy(monkeypatch)
    seen = {}
    for mode in ("mention", "rules"):
        _source(monkeypatch, mode)
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _JournalLLM(picked, "ЛГ (LH)"))
        with ev_world.FrozenWorld(mini_prod_world, "day") as world:
            [turn] = run_dialog([said], f"journal-basket-edge-{mode}-{said[:12]}")
            entries = _journal_rows(world._journal)
        assert all(e["source"] == "basket" for e in entries), entries
        seen[mode] = ([e["phrase"] for e in entries], [(e["found"], e["services"]) for e in entries], turn["text"])
    assert seen["mention"][0] == mention
    assert seen["rules"][0] == rules
    assert seen["mention"][1] == seen["rules"][1]  # решение LLM по строкам — то же, меняется только фраза
    assert seen["mention"][2] == seen["rules"][2]  # журнал — подсказка клинике, ответ не меняется
