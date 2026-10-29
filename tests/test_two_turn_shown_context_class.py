"""Два хода: короткий следующий вопрос не берёт услугу, которой пациент не видел (контекст = показанное) — характеризация
деплоя 21 перед шагами 1а / 3 / 5 карты Понимания (консилиум 30.10, Q6; мастер-план §4 (б)).

После ответа X (не нашёл, свой вопрос в рамке — вариант Б, список, похожие варианты, перевод, ответ на кнопку) пациент пишет коротко:
«а сколько стоит?», «да», цифру, «а подготовка?». Инварианты класса:
- **услуга в ответе — строка каталога или показанное ботом**: название в «…» после «По услуге» / «подготовке к» / «готовности» —
  строка прайса или то, что бот уже называл (слова вопроса пациента услугой не становятся);
- **реплика целиком не в слоте**, если судья сказал «не ответ» (`domain` / `new_topic`): консилиум Q4 — «реплика целиком в слот —
  больше никогда»;
- **на коротком вопросе без своей услуги нет чужой цены**: цена — только услуги из показанного.

Обязательные диалоги (задание штурмана, карта П-1 / П-2 / Д2 / Д3):
- П-1: «Как подготовиться к анализу» → «Откуда берут биоматериал для ВПЧ?» → «а сколько стоит?»;
- Д3 (П-2): «Как подготовиться к анализу» → «а сколько стоит ТТГ?» → «а как подготовиться?»;
- Д2: два листа с одним вопросом подряд → ответ; звено 2 — оффер оператора поверх вопроса кнопки.

Два режима на диалог: **LLM поняла** (харнесс `tests/understanding_contract.py`, подделка настоящего типа, конфиг деплоя 21) и
**LLM недоступна** (`test_price_turns_golden.run_dialog` под тем же конфигом — эталон деградации). Известные дефекты — xfail strict
с ID: Н-1 (OWNER_DECISIONS 29.10, карта П-1), BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT (звено 2), BUG-2026-10-28-PREPARE-FOLLOWUP-LETTERS.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import pytest

from evalbench import world as ev_world
from tests import understanding_contract as uc
from tests.test_price_turns_golden import FIXTURE, run_dialog
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура

_N1 = "Н-1 (OWNER_DECISIONS 29.10; карта Понимания П-1): свой вопрос в рамке кнопки (`domain`) засчитан ответом — реплика в слоте"
_N1_QF = "Н-1, класс (консилиум Q4; карта §1: quick-fill пишет реплику в `last_entities`): реплика целиком в слоте после «другая тема»"
_D2 = "BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT, звено 2 (консилиум 29.10): оффер «да/нет» поверх вопроса кнопки"
_D3 = "BUG-2026-10-28-PREPARE-FOLLOWUP-LETTERS: переклейка подготовки по буквам после памятки"

_CLAIM = re.compile(r"(?:По услуге|подготовке к|готовности) «([^»]+)»")
_PRICE = re.compile(r"\d\s*руб")


def _norm(text: str | None) -> str:
    return " ".join(re.sub(r"[?!.,;:«»]", " ", str(text or "").lower()).split())


@lru_cache(maxsize=1)
def _catalog() -> frozenset[str]:
    rows = (FIXTURE / "price_by_region").glob("price_region_3_*.jsonl")
    names = {json.loads(line).get("serviceName") for path in rows for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return frozenset(_norm(name) for name in names if name)


def _unseen_claims(record: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(ход, название): услуга, названная ответом бота, — не строка каталога и не то, что бот показал раньше."""

    shown: set[str] = set()
    bad = []
    for k, turn in enumerate(record):
        for name in _CLAIM.findall(turn["text"]):
            if _norm(name) not in _catalog() and _norm(name) not in shown:
                bad.append((k, name))
        shown |= {_norm(name) for name in re.findall(r"«([^»]+)»", turn["text"])}
        shown |= {_norm(line.lstrip("—-• ").strip()) for line in turn["text"].splitlines() if line.lstrip().startswith(("—", "- "))}
    return bad


def _reply_in_slot(turns: list[uc.Turn], record: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(ход, слот): реплика целиком в `service_name` / `test_goal` на ходе, где судья сказал «не ответ на вопрос»."""

    bad = []
    for k, (turn, rec) in enumerate(zip(turns, record)):
        if turn.understood.pending in {"domain", "new_topic"}:
            bad += [(k, slot) for slot in ("service_name", "test_goal") if rec[slot] and _norm(rec[slot]) == _norm(turn.text)]
    return bad


def _down(world_dir, monkeypatch, turns: list[str | tuple[str, str]], session: str) -> list[dict[str, Any]]:
    """Тот же диалог при недоступной LLM (сеть гейта закрыта) под конфигом деплоя 21."""

    from messengers_router.runtime_config import config as cfg

    for key, value in uc.D21_CFG.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    with ev_world.FrozenWorld(world_dir, "day") as world:
        record = run_dialog(turns, f"down-{session}")
        assert list(world.net.attempts) == []
    return record


def _as_down(turns: list[uc.Turn]) -> list[str | tuple[str, str]]:
    return [(t.button, t.text) if t.button else t.text for t in turns]


# --------------------------------------------------------------------------- П-1: свой вопрос в рамке → «а сколько стоит?»


_PREPARE = "menu.tests.prepare"
P1 = [
    uc.press(_PREPARE),
    uc.say("Откуда берут биоматериал для ВПЧ?", pending="domain", services=("ВПЧ",)),
    uc.say("а сколько стоит?", services=(), pending="new_topic", topic="PRICE"),
]
P1_HAVE = [
    uc.press(_PREPARE),
    uc.say("а у вас делают ВПЧ?", pending="domain", services=("ВПЧ",)),
    uc.say("а сколько стоит?", services=(), pending="new_topic", topic="PRICE"),
]


def test_p1_own_question_keeps_the_prepare_frame(d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, P1)
    assert record[1]["label"] == "PREPARE" and record[1]["telemetry"].get("nlu_topic_by_llm") == "button_answer"
    assert record[2]["label"] == "PRICE", record[2]["label"]


@pytest.mark.xfail(strict=True, reason=_N1)
@pytest.mark.parametrize("turns", [P1, P1_HAVE], ids=["biomaterial", "do_you_have"])
def test_p1_own_question_is_not_written_into_the_slot(turns, d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, turns)
    assert _reply_in_slot(turns, record) == []


@pytest.mark.xfail(strict=True, reason=_N1)
def test_p1_short_price_question_does_not_price_the_words_of_the_question(d21_world, monkeypatch):  # noqa: F811
    """Сейчас: «По услуге «Откуда берут биоматериал для ВПЧ» … соединяю» — слова вопроса стали услугой на следующем ходе."""

    record = uc.run(d21_world, monkeypatch, P1)
    assert _unseen_claims(record) == []


def test_p1_do_you_have_prices_only_catalog_rows(d21_world, monkeypatch):  # noqa: F811
    """Встречный: в «а у вас делают ВПЧ?» услуга названа — «а сколько стоит?» даёт варианты строк прайса с ВПЧ, не чужую."""

    record = uc.run(d21_world, monkeypatch, P1_HAVE)
    assert _unseen_claims(record) == [] and "ВПЧ" in record[2]["text"], record[2]["text"]


def test_p1_named_answer_then_price_is_about_the_named_service(d21_world, monkeypatch):  # noqa: F811
    turns = [uc.press(_PREPARE), uc.say("ВПЧ", pending="answer", services=("ВПЧ",)), P1[2]]
    record = uc.run(d21_world, monkeypatch, turns)
    assert record[1]["service_name"] == "ВПЧ" and _unseen_claims(record) == [], record[2]["text"]


def test_p1_llm_down_asks_again_and_prices_nothing(d21_world, monkeypatch):  # noqa: F811
    """Эталон деградации (LLM недоступна): свой вопрос в рамке подготовки — переспрос «к какому анализу», слот не пишется;
    «а сколько стоит?» — переспрос названия. Вред П-1 — только у варианта Б с живым судьёй."""

    record = _down(d21_world, monkeypatch, _as_down(P1), "p1")
    assert [r["label"] for r in record] == ["PREPARE", "PREPARE", "PRICE"], [r["label"] for r in record]
    assert record[1]["service_name"] is None and not _PRICE.search(record[2]["text"]) and _unseen_claims(record) == []


# --------------------------------------------------------------------------- Д3 / П-2: «а сколько стоит ТТГ?» после вопроса подготовки


D3 = [
    uc.press(_PREPARE),
    uc.say("а сколько стоит ТТГ?", pending="new_topic", services=("ТТГ",), topic="PRICE"),
    uc.say("а как подготовиться?", services=(), pending="new_topic", topic="PREPARE"),
]


def test_d3_price_question_after_prepare_button_is_a_price(d21_world, monkeypatch):  # noqa: F811
    """Деплой 21: «другая тема» судьи — цена, не подготовка (переклейка по буквам на этом ходе не срабатывает)."""

    record = uc.run(d21_world, monkeypatch, D3)
    assert record[1]["label"] == "PRICE" and "ТТГ" in record[1]["text"], (record[1]["label"], record[1]["text"])
    assert record[1]["telemetry"].get("tu_pending_outcome") == "new_topic"


def test_d3_then_prepare_does_not_take_an_unseen_service(d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, D3)
    assert record[2]["label"] == "PREPARE" and _unseen_claims(record) == [], record[2]["text"]


@pytest.mark.xfail(strict=True, reason=_N1_QF)
def test_d3_new_topic_reply_is_not_written_into_the_slot(d21_world, monkeypatch):  # noqa: F811
    """Сейчас: `test_goal = «а сколько стоит ТТГ?»` — дозаполнение пишет реплику, хотя судья сказал «другая тема»."""

    record = uc.run(d21_world, monkeypatch, D3)
    assert _reply_in_slot(D3, record) == []


def test_d3_llm_down_price_question_is_lost_into_prepare(d21_world, monkeypatch):  # noqa: F811
    """Эталон деградации (LLM недоступна): правила разобрали «ТТГ» значением слота — ответ на вопрос подготовки, тема PREPARE.
    Чужой услуги нет; цены пациент не получает."""

    record = _down(d21_world, monkeypatch, _as_down(D3[:2]), "d3")
    assert record[1]["label"] == "PREPARE" and not _PRICE.search(record[1]["text"]), (record[1]["label"], record[1]["text"])
    assert _unseen_claims(record) == []


_D3_LETTERS = [
    uc.press("menu.price.test"),
    uc.say("ферритин", pending="answer", services=("ферритин",)),
    uc.say("а как к нему подготовиться?", services=(), topic="PREPARE"),
    uc.say("а где его сдать?", services=(), pending="new_topic", topic="ADDRESS"),
]


def test_d3_letters_name_only_the_shown_service(d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, _D3_LETTERS)
    assert _unseen_claims(record) == [] and "«Ферритин»" in record[2]["text"], record[2]["text"]


@pytest.mark.xfail(strict=True, reason=_D3)
def test_d3_letters_prepare_question_keeps_the_shown_service_in_the_slot(d21_world, monkeypatch):  # noqa: F811
    """Сейчас: «а как к нему подготовиться?» кладёт `service_name = «Нему подготовиться»` — корень Д3 (журнал)."""

    record = uc.run(d21_world, monkeypatch, _D3_LETTERS)
    assert record[2]["service_name"] == "Ферритин", record[2]["service_name"]


@pytest.mark.xfail(strict=True, reason=_D3)
def test_d3_letters_where_after_memo_is_an_address(d21_world, monkeypatch):  # noqa: F811
    """Сейчас: «а где его сдать?» после памятки — снова подготовка («К какому анализу…»), хотя судья сказал «другая тема», а
    классификатор — адрес."""

    record = uc.run(d21_world, monkeypatch, _D3_LETTERS)
    assert record[3]["label"] == "ADDRESS", record[3]["label"]


# --------------------------------------------------------------------------- Д2: два листа с одним вопросом подряд


@pytest.mark.parametrize("pair", [("menu.price.test", "menu.tests.price"), ("menu.tests.price", "menu.price.test")])
def test_d2_two_leaves_with_one_question_then_answer_is_a_price(pair, d21_world, monkeypatch):  # noqa: F811
    turns = [uc.press(pair[0]), uc.press(pair[1]), uc.say("ферритин", pending="answer", services=("ферритин",))]
    record = uc.run(d21_world, monkeypatch, turns)
    assert "Перевести на оператора" not in record[1]["text"], record[1]["text"]
    assert not record[2]["handoff"] and "530 руб" in record[2]["text"], record[2]["text"]


def test_d2_two_leaves_llm_down_then_answer_is_a_price(d21_world, monkeypatch):  # noqa: F811
    record = _down(d21_world, monkeypatch, _as_down([uc.press("menu.price.test"), uc.press("menu.tests.price"), uc.say("ферритин")]), "d2")
    assert "Перевести на оператора" not in record[1]["text"] and not record[2]["handoff"], [r["text"] for r in record]


_RESULTS = ["menu.results.lookup", "menu.results.missing"]


@pytest.mark.xfail(strict=True, reason=_D2)
@pytest.mark.parametrize("presses", [_RESULTS[:1], _RESULTS], ids=["one_leaf", "two_leaves"])
def test_d2_non_answer_to_the_button_question_gets_no_yes_no_offer_on_top(presses, d21_world, monkeypatch):  # noqa: F811
    """Консилиум 29.10 (звено 2, вариант А): на втором повторе — тот же вопрос и строка «напишите «оператор»», не вопрос «да/нет».
    Сейчас: «…Пример: «Иванов, 1989, Бг, 123». Похоже, мне не удаётся помочь с этим в чате. Перевести на оператора?» уже на
    первом не-ответе после нажатия."""

    turns = [*(uc.press(b) for b in presses), uc.say("не помню номер", pending="answer")]
    record = uc.run(d21_world, monkeypatch, turns)
    assert (record[-1]["expects"] or {}).get("type") != "operator_offer", record[-1]["text"]


@pytest.mark.parametrize("judge", ["new_topic", "no"])
def test_d2_answer_to_the_button_question_under_the_offer_is_not_a_handoff_when_judged_not_yes(judge, d21_world, monkeypatch):  # noqa: F811
    """Строка контракта: судья оффера прочёл реквизиты не «да» — перевода нет (вред звена 2 — только когда судья читает «да»)."""

    turns = [uc.press(_RESULTS[0]), uc.say("не помню номер", pending="answer"), uc.say("Иванов, 1989, Бг, 123", pending=judge)]
    record = uc.run(d21_world, monkeypatch, turns)
    assert not record[2]["handoff"], record[2]["text"]


# --------------------------------------------------------------------------- прочие X → короткий вопрос


def test_not_found_then_short_price_asks_the_name_again(d21_world, monkeypatch):  # noqa: F811
    turns = [
        uc.say("сколько стоит анализ на гормон счастья", services=("гормон счастья",), topic="PRICE"),
        uc.say("а сколько стоит?", services=(), pending="new_topic", topic="PRICE"),
    ]
    record = uc.run(d21_world, monkeypatch, turns)
    assert not _PRICE.search(record[1]["text"]) and not record[1]["handoff"] and _unseen_claims(record) == [], record[1]["text"]


def test_list_then_short_prepare_takes_no_single_unseen_service(d21_world, monkeypatch):  # noqa: F811
    """После корзины «а подготовка?» — переспрос «к какому анализу» (не одна услуга из списка и не чужая)."""

    turns = [uc.say("сколько стоит ферритин, глюкоза", services=("ферритин", "глюкоза")), uc.say("а подготовка?", services=(), topic="PREPARE")]
    record = uc.run(d21_world, monkeypatch, turns)
    assert record[1]["label"] == "PREPARE" and record[1]["service_name"] is None and _unseen_claims(record) == [], record[1]


def test_variants_then_digit_prices_nothing_unseen(d21_world, monkeypatch):  # noqa: F811
    """После «нашёл похожие услуги: — …» цифра «1» — без чужой цены (сейчас: «Не совсем понял» — у вопроса нет пунктов)."""

    turns = [uc.say("сколько стоит ттг", services=("ттг",)), uc.say("1", pending="new_topic")]
    record = uc.run(d21_world, monkeypatch, turns)
    assert not _PRICE.search(record[1]["text"]) and _unseen_claims(record) == [], record[1]["text"]


def test_handoff_leaf_then_yes_prices_nothing(d21_world, monkeypatch):  # noqa: F811
    record = uc.run(d21_world, monkeypatch, [uc.press("menu.results.eta"), uc.say("да", pending="yes")])
    assert record[0]["handoff"] and not _PRICE.search(record[1]["text"]) and record[1]["service_name"] is None, record[1]["text"]


def test_button_answer_then_where_keeps_the_shown_service(d21_world, monkeypatch):  # noqa: F811
    turns = [
        uc.press("menu.price.test"),
        uc.say("ферритин", pending="answer", services=("ферритин",)),
        uc.say("а где сдать?", services=(), pending="new_topic", topic="ADDRESS"),
    ]
    record = uc.run(d21_world, monkeypatch, turns)
    assert record[2]["label"] == "ADDRESS" and record[2]["service_name"] == "Ферритин", record[2]
