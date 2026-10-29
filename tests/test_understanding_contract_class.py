"""Контракт «понимание хода → поведение» под конфигом деплоя 21 (ход (б), деплой 22; консилиум 30.10, Q6 — эталон понимание, не текст).

До переключения на единое понимание (шаги 0–5 карты `~/Dev/handoffs/2026-10-30_understanding_map.md`) поведение конвейера при
заданном понимании фиксируется как есть: что LLM решила (исход судьи висящего вопроса `pending_answer.Answer`, вопрос рамки листа,
фразы разметки, поля `TurnUnderstanding`) → что сделал ход (тема, перевод, ответ по данным, состояние). Шаг, который переносит решение
в одно понимание, обязан держать эти строки; меняет — правкой теста с разбором, а не молча. Харнесс — `tests/understanding_contract.py`
(подделка LLM настоящего типа: `choice.Choice` через настоящий разбор; конфиг деплоя 21 — `D21_CFG`).

Строки контракта:
- судья висящего вопроса кнопки: `answer` — тема кнопки без классификатора; `new_topic` — тема классификатора; `domain` (вариант Б)
  — тема кнопки; судья недоступен — правила (значение, разобранное правилами, — ответ без LLM);
- вопрос рамки листа без вопроса («Нужно ли записываться»): `same` — тема листа поверх правила записи; `other` — общий путь;
- поле `facet` понимания: одна и та же реплика при `readiness` — срок из МИС, при `price` (и теме цены) — путь цены, не срок;
- фразы разметки: список из двух — корзина с ценой каждой позиции;
- срочное и явная просьба оператора не зависят от решения судьи (принцип 4: LLM срочность не снимает).

Известный дефект — xfail strict: Н-1 (OWNER_DECISIONS 29.10, карта П-1) — исход `domain` засчитан ответом, реплика целиком
пишется в слот.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from messengers_router import choice as choice_mod
from messengers_router.choice import Choice, Option, Status, Strength
from tests import understanding_contract as uc
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура

_NASON = Path.home() / "Dev" / "dla-private" / "bench" / "runs" / "nason_btn" / "run.json"
_N1 = "Н-1 (OWNER_DECISIONS 29.10; карта Понимания П-1): исход `domain` засчитан ответом — реплика целиком в слоте"


def _norm(text: str | None) -> str:
    return " ".join(str(text or "").lower().replace("?", " ").replace("!", " ").replace(".", " ").split())


# --------------------------------------------------------------------------- харнесс: подделка настоящего типа


def _tasks_with_outcomes() -> list[choice_mod.Task]:
    from messengers_router import button_flow, pending_answer, pending_reply

    found = [pending_answer.FREE_TASK, pending_answer.FREE_BUTTON_TASK, pending_answer.FRAME_BUTTON_TASK, pending_answer.MANY_TASK]
    found += [t for t in vars(pending_reply).values() if isinstance(t, choice_mod.Task) and t.outcomes]
    found += [t for t in vars(button_flow).values() if isinstance(t, choice_mod.Task) and t.outcomes]
    return list(dict.fromkeys(found))


@pytest.mark.parametrize("task", _tasks_with_outcomes(), ids=lambda t: t.name)
def test_fake_outcome_roundtrips_through_the_real_parser(task):
    """Исход, заданный тестом, доходит до кода тем же `Choice`, что дал бы разбор ответа живой модели."""

    for name in task.outcomes:
        raw = uc._choice_json(uc.outcome(name), "", choice_mod.answer_schema(task, []))
        assert choice_mod.parse_answer(task, [], raw) == Choice(Status.OUTCOME, outcome=name), (task.name, name, raw)


def test_fake_outcome_unknown_to_the_task_is_invalid_like_a_live_model():
    """`domain` у задачи без такого исхода (вариант А) — нарушение контракта, как у живой модели, а не тихая подмена."""

    from messengers_router.pending_answer import FREE_BUTTON_TASK

    raw = uc._choice_json(uc.outcome("domain"), "", choice_mod.answer_schema(FREE_BUTTON_TASK, []))
    assert choice_mod.parse_answer(FREE_BUTTON_TASK, [], raw).status is Status.INVALID


@pytest.mark.parametrize("by_text", [False, True])
def test_fake_pick_by_option_text(by_text):
    task = choice_mod.Task(name="probe", by_text=by_text)
    options = [Option("a", "Ферритин", Strength.STRONG), Option("b", "Глюкоза")]
    _system, user = "", "\n".join(f"{i}. {o.text}" for i, o in enumerate(options, 1))
    raw = uc._choice_json(uc.picked("Глюкоза"), user, choice_mod.answer_schema(task, options))
    assert choice_mod.parse_answer(task, options, raw) == Choice(Status.PICKED, ids=("b",))


def test_every_choose_prompt_is_routed_by_its_own_instruction():
    """Задача узнаётся по своей инструкции: у двух задач одна инструкция — подделка ответила бы одной за другую."""

    systems = uc._task_systems()
    from messengers_router import prompt_registry

    assert len(systems) == len(list(prompt_registry._PROMPTS_DIR.glob("choose_*.txt")))
    assert {"pending_free_reply_button_frame", "button_frame", "pending_reply"} <= set(systems.values())


def test_d21_cfg_keys_are_known_to_the_code():
    """Ключ конфига деплоя 21, которого код не знает, ничего не включает (класс Н-7): тест шёл бы не по пути прода."""

    from agent_logic_2 import config as code_cfg

    unknown = [key for key in uc.D21_CFG if not hasattr(code_cfg, key)]
    assert unknown == []


@pytest.mark.skipif(not _NASON.exists(), reason="прогон стенда деплоя 21 — вне репо (облако его не видит)")
def test_d21_cfg_equals_the_bench_run_of_deploy_21():
    cfg = json.loads(_NASON.read_text(encoding="utf-8"))["cfg"]
    known = {key: row["value"] for key, row in cfg.items() if row.get("known")}
    assert known == uc.D21_CFG


# --------------------------------------------------------------------------- судья висящего вопроса кнопки


_PREPARE = "menu.tests.prepare"


def test_answer_keeps_the_button_topic_without_classifier(d21_world, monkeypatch):  # noqa: F811
    turn = uc.run(d21_world, monkeypatch, [uc.press(_PREPARE), uc.say("глюкоза", pending="answer", services=("глюкоза",))])[1]
    assert turn["label"] == "PREPARE" and turn["telemetry"].get("nlu_topic_by_llm") == "button_answer", turn["telemetry"]
    assert "classifier" not in " ".join(turn["asked"]), turn["asked"]
    assert "глюкоз" in _norm(turn["service_name"]), turn["service_name"]


def test_new_topic_takes_the_classifier_topic(d21_world, monkeypatch):  # noqa: F811
    turns = [uc.press(_PREPARE), uc.say("сколько стоит ферритин", pending="new_topic", services=("ферритин",), topic="PRICE")]
    turn = uc.run(d21_world, monkeypatch, turns)[1]
    assert turn["label"] == "PRICE" and turn["telemetry"].get("tu_pending_outcome") == "new_topic", turn["telemetry"]
    assert "530 руб" in turn["text"], turn["text"]


_OWN_QUESTION = "а у вас делают ВПЧ?"


def test_domain_keeps_the_button_topic(d21_world, monkeypatch):  # noqa: F811
    """Вариант Б: свой вопрос в теме кнопки — тема кнопки (рамка держится), классификатора нет."""

    turn = uc.run(d21_world, monkeypatch, [uc.press(_PREPARE), uc.say(_OWN_QUESTION, pending="domain", services=("ВПЧ",))])[1]
    assert turn["label"] == "PREPARE" and turn["telemetry"].get("nlu_topic_by_llm") == "button_answer", turn["telemetry"]
    assert not turn["handoff"]


@pytest.mark.xfail(strict=True, reason=_N1)
def test_domain_does_not_write_the_reply_into_the_slot(d21_world, monkeypatch):  # noqa: F811
    """Консилиум 30.10, Q4: реплика целиком в слот — никогда; свой вопрос в рамке — слот из разбора или пусто."""

    turn = uc.run(d21_world, monkeypatch, [uc.press(_PREPARE), uc.say(_OWN_QUESTION, pending="domain", services=("ВПЧ",))])[1]
    assert _norm(turn["service_name"]) != _norm(_OWN_QUESTION), turn["service_name"]


def test_judge_down_rules_value_is_an_answer_without_llm(d21_world, monkeypatch):  # noqa: F811
    """Судья недоступен, правила разобрали значение — ответ без решения LLM (запасной путь деплоя 21; эталон деградации)."""

    turn = uc.run(d21_world, monkeypatch, [uc.press(_PREPARE), uc.say("а сколько стоит ТТГ?", services=None)])[1]
    tel = turn["telemetry"]
    assert (tel.get("tu_pending_outcome"), tel.get("tu_pending_src")) == ("answer", "fast"), tel
    assert turn["label"] == "PREPARE", turn["label"]


# --------------------------------------------------------------------------- рамка листа без своего вопроса


_LEAF = "menu.tests.booking_needed"
_LEAF_REPLY = "Записаться на хгч"


def test_leaf_frame_same_keeps_the_leaf_topic_against_the_booking_rule(d21_world, monkeypatch):  # noqa: F811
    turn = uc.run(d21_world, monkeypatch, [uc.press(_LEAF), uc.say(_LEAF_REPLY, frame="same", services=("хгч",))])[1]
    assert turn["label"] == "ADDRESS" and turn["telemetry"].get("nlu_topic_by_llm") == "button_answer", turn["telemetry"]
    assert "choose_button_frame" in turn["asked"], turn["asked"]


@pytest.mark.parametrize("frame", ["other", None])
def test_leaf_frame_other_or_down_takes_the_general_path(frame, d21_world, monkeypatch):  # noqa: F811
    turns = [uc.press(_LEAF), uc.say(_LEAF_REPLY, frame=frame, services=("хгч",), topic="APPOINTMENT")]
    turn = uc.run(d21_world, monkeypatch, turns)[1]
    assert turn["label"] == "APPOINTMENT" and turn["telemetry"].get("nlu_topic_by_llm") != "button_answer", turn["telemetry"]


# --------------------------------------------------------------------------- поля понимания: facet, список


_READINESS = "сколько готовится ферритин"


def test_facet_readiness_answers_the_term_from_mis_not_the_price(d21_world, monkeypatch):  # noqa: F811
    turn = uc.run(d21_world, monkeypatch, [uc.say(_READINESS, services=("ферритин",), tu={"facet": "readiness"})])[0]
    assert "Срок готовности" in turn["text"] and "руб" not in turn["text"], turn["text"]
    assert (turn["understanding"] or {}).get("facet") == "readiness"


def test_facet_price_on_the_same_words_takes_the_price_path(d21_world, monkeypatch):  # noqa: F811
    """Те же слова с `facet = price` (и темой цены) — путь цены по пониманию, а не срок. Тему хода решает метка, не `facet`:
    без темы цены (классификатор недоступен) `facet = price` хода в цену не переводит — читатели `facet` пока только срок,
    планировщик цены и П1 (карта §1)."""

    turns = [uc.say(_READINESS, services=("ферритин",), topic="PRICE", tu={"facet": "price"})]
    turn = uc.run(d21_world, monkeypatch, turns)[0]
    assert (turn["understanding"] or {}).get("facet") == "price" and turn["label"] == "PRICE", turn["understanding"]
    assert "planner_price_by_understanding" in turn["understanding"]["acted"], turn["understanding"]
    assert "Срок готовности" not in turn["text"], turn["text"]


def test_two_phrases_price_each_position(d21_world, monkeypatch):  # noqa: F811
    turn = uc.run(d21_world, monkeypatch, [uc.say("сколько стоит ферритин, глюкоза", services=("ферритин", "глюкоза"))])[0]
    assert (turn["understanding"] or {}).get("services") == ["ферритин", "глюкоза"], turn["understanding"]
    assert "Ферритин — 530 руб" in turn["text"] and "Глюкоза — 190 руб" in turn["text"], turn["text"]


# --------------------------------------------------------------------------- что решение LLM не перебивает


@pytest.mark.parametrize("pending", ["answer", "domain", None])
def test_urgent_reply_inside_the_frame_is_urgent_whatever_the_judge_says(pending, d21_world, monkeypatch):  # noqa: F811
    turns = [uc.press("menu.price.test"), uc.say("у меня сильная боль в груди и трудно дышать", pending=pending)]
    turn = uc.run(d21_world, monkeypatch, turns)[1]
    assert turn["label"] == "URGENT" and turn["handoff"], (turn["label"], turn["text"])


@pytest.mark.parametrize("pending", ["answer", "domain", None])
def test_explicit_operator_request_inside_the_frame_is_a_handoff(pending, d21_world, monkeypatch):  # noqa: F811
    turn = uc.run(d21_world, monkeypatch, [uc.press(_PREPARE), uc.say("соедините с оператором", pending=pending)])[1]
    assert turn["handoff"] and turn["telemetry"].get("answer_source") == "precheck:manual_operator", turn["telemetry"]
