"""Класс «ответ на свой список» — BUG-2026-10-30-OWN-LIST-REPLY (регресс деплоя 22 под `facet_by_llm = on`).

Смоук прода 30.10: «сколько стоит оак» → список строк ОАК → «1» → «По этому вопросу у меня нет данных… Перевести?», «2» — чужая
цена; «первый», «давайте второй», название строки своими словами — оффер или подбор анализов и без `facet_by_llm` (стенд 30.10,
набор `evalbench/sets/own_question_reply.jsonl`). Корень: «что значит эта реплика» решали три места, и о показанном списке знало одно
— правило номера строки (`rule_price_family_row`). Тип вопроса (`router._route_by_turn_facet`) судил голую реплику «1» → «ни одно» →
оффер; висящим вопросом после списка был скрытый вторичный оффер, а не сам список — «давайте второй» становилось «да» ему.

Инвариант класса: показанный нумерованный список цен — висящий вопрос одной функции ответа (`pending_answer`, тип `show_all`, пункты
— строки, как их увидел пациент, и «все»). Реплика, которую функция признала выбором строки (номер, порядковое слово, название), —
цена этой строки: не оффер, не чужая строка, не другая тема; тип вопроса такого хода не судится (LLM о нём не спрашивается).
Реплика, которую функция выбором не признала (свой вопрос о показанном, другая услуга), идёт обычным путём, и тип вопроса её судит.

Мир — мини-прод, путь прода (`test_open_bugs_2026_10_14._ProdLLM`: разметка называет услугу, выбор строки признаёт строки ОАК — на
проде так получается список с подсказкой «все»), поверх — конфиг деплоя 22 с `facet_by_llm = on`; тип вопроса и функция ответа
отвечают решениями хода (`_Turn`): на «1» тип вопроса — «ни одно», как у живой модели на проде. Выключатель фикса —
`own_question_reply`: off — путь прода (красное, `xfail strict`), on — фикс.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.choice import Choice
from tests.spans_oracle import keep_llm_breaker_closed
from tests.test_open_bugs_2026_10_14 import _ALL_SETS, _ProdLLM
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.understanding_contract import NONE_FIT, _choice_json, _task_systems, outcome, picked

# Деплой 22 (прод с 30.10, вариант Б) с типом вопроса от LLM — так, как его включат снова.
D22_FACET: dict[str, Any] = {
    "MR_FRAME_POLICY": "on",
    "MR_BUTTON_FRAME_CHECK": "judge_leaf",
    "MR_EXPECTS_BY_UNDERSTANDING": "on",
    "MR_PRICE_NOT_FOUND_BY_DATA": "on",
    "MR_FACET_BY_LLM": "on",
}
PRICE_FACET = picked("Узнать стоимость анализа или услуги")
PREPARE_FACET = picked("Узнать, как подготовиться к анализу или услуге")
_OAK = "сколько стоит оак"
_ROW = re.compile(r"^\d+\. (.+? — \d[\d\s]* руб\.)", re.M)


class _Turn:
    """Задачи выбора хода (тип вопроса, функция ответа) — решения, заданные тестом на каждый ход; разметку и выбор строки цены
    отвечает `_ProdLLM`. Задача без решения на этом ходе — «недоступна»."""

    def __init__(self) -> None:
        self.choices: Mapping[str, Choice] = {}
        self.asked: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, format: Any = None, **_kwargs: Any) -> GenerateResponse:
        task = _task_systems().get(str(system or "").strip()) if system else None
        self.asked.append(f"choose_{task}")
        choice = self.choices.get(task or "")
        if choice is None:
            raise ConnectionError(f"тест: решения задачи {task} на этом ходе нет")
        return GenerateResponse(model="m", response=_choice_json(choice, prompt, format), done=True)


def _dialog(world_dir: Any, monkeypatch: Any, own: str, turns: list[tuple[str, Mapping[str, Choice]]], **cfg: Any) -> list[dict[str, Any]]:
    """Диалог через обработчик эндпоинта в одной сессии: реплика и решения задач выбора на этом ходе."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as rcfg
    from messengers_router.services import Services

    for key, value in {**_ALL_SETS["deploy"], **D22_FACET, "MR_OWN_QUESTION_REPLY": own, "MR_CHOOSE_CACHE_TTL_S": 0, **cfg}.items():
        monkeypatch.setattr(rcfg, key, value, raising=False)
    inner = _Turn()
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _ProdLLM(("оак", "ферритин"), "общий анализ крови", inner=inner))
    keep_llm_breaker_closed(monkeypatch)
    session = f"own-list-{own}-" + json.dumps([text for text, _ in turns], ensure_ascii=False) + json.dumps(cfg, sort_keys=True)

    async def go() -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        for text, choices in turns:
            inner.choices, start = choices, len(inner.asked)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id="")
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            decision = ((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}
            entities = memory.get(session).last_entities
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "handoff": bool(body.get("handoff")),
                    "label": decision.get("label"),
                    "expects": entities.get("_expects"),
                    "asked": inner.asked[start:],
                }
            )
        return out

    with ev_world.FrozenWorld(world_dir, "day"):
        return asyncio.run(go())


def _list(world_dir: Any, monkeypatch: Any, own: str) -> dict[str, Any]:
    return _dialog(world_dir, monkeypatch, own, [(_OAK, {"turn_facet": PRICE_FACET})])[0]


def _rows(world_dir: Any, monkeypatch: Any) -> list[str]:
    """Строки показанного списка, как их увидел пациент («название — цена»); под on — они же пункты вопроса о списке."""

    first = _list(world_dir, monkeypatch, "on")
    rows = _ROW.findall(first["text"])
    assert len(rows) >= 2, first["text"]
    assert first["expects"]["type"] == "show_all" and first["expects"]["options"] == [*rows, "все"], first["expects"]
    return rows


def _row_violations(second: dict[str, Any], row: str) -> list[str]:
    """Нарушения инварианта на ходе выбора строки: цена ровно этой строки, без оффера и без суда типа вопроса."""

    found = []
    if second["handoff"] or "Перевести" in second["text"] or "оператор" in second["text"]:
        found.append("оффер или перевод вместо строки")
    if row.split(" — ")[0] not in second["text"] or row.split(" — ")[1] not in second["text"]:
        found.append(f"нет строки «{row}» в ответе: {second['text'][:200]!r}")
    if second["label"] != "PRICE":
        found.append(f"тема {second['label']} вместо PRICE")
    if "choose_turn_facet" in second["asked"]:
        found.append("тип вопроса судил ответ на свой список")
    return found


# Выбор строки: реплика → (номер строки с 1, решение функции ответа — у номера его нет: без LLM). Тип вопроса — «ни одно».
PICKS = {"1": (1, False), "2": (2, False), "первый": (1, True), "давайте второй": (2, True)}


def _pick_dialog(world_dir: Any, monkeypatch: Any, own: str, reply: str) -> tuple[dict[str, Any], str]:
    rows = _rows(world_dir, monkeypatch)
    index, by_llm = PICKS[reply]
    choices = {"turn_facet": NONE_FIT, **({"pending_reply": picked(rows[index - 1])} if by_llm else {})}
    second = _dialog(world_dir, monkeypatch, own, [(_OAK, {"turn_facet": PRICE_FACET}), (reply, choices)])[1]
    return second, rows[index - 1]


@pytest.mark.parametrize("reply", sorted(PICKS))
def test_row_pick_answers_that_row(reply, mini_prod_world, monkeypatch):  # noqa: F811
    second, row = _pick_dialog(mini_prod_world, monkeypatch, "on", reply)
    assert _row_violations(second, row) == []


@pytest.mark.xfail(strict=True, reason="BUG-2026-10-30-OWN-LIST-REPLY: путь деплоя 22 с facet_by_llm (`own_question_reply = off`)")
@pytest.mark.parametrize("reply", sorted(PICKS))
def test_row_pick_answers_that_row_on_prod_path(reply, mini_prod_world, monkeypatch):  # noqa: F811
    second, row = _pick_dialog(mini_prod_world, monkeypatch, "off", reply)
    assert _row_violations(second, row) == []


@pytest.mark.parametrize("reply", ["все", "покажи все"])
def test_show_all_expands_without_facet(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """«все» на свой список — пункт вопроса: раскрытый список, а не «ни одно» типа вопроса."""

    first, second = _dialog(mini_prod_world, monkeypatch, "on", [(_OAK, {"turn_facet": PRICE_FACET}), (reply, {"turn_facet": NONE_FIT})])
    shown, now = _ROW.findall(first["text"]), _ROW.findall(second["text"])
    assert not second["handoff"] and len(now) > len(shown) and now[: len(shown)] == shown, second["text"]
    assert "choose_turn_facet" not in second["asked"]


def test_list_is_the_hanging_question_not_hidden_offer(mini_prod_world, monkeypatch):  # noqa: F811
    """Под on висящий вопрос после своего списка — сам список (пункты — строки и «все»); под off — как было (подсказка «все»)."""

    _rows(mini_prod_world, monkeypatch)
    off = _list(mini_prod_world, monkeypatch, "off")["expects"]
    assert off is None or off["options"] == ["все"], off


# Встречный свип (anti-over-block): реплика, которую функция ответа выбором строки не признала, — обычный путь; тип вопроса её судит.
def test_own_question_about_shown_goes_as_without_switch(mini_prod_world, monkeypatch):  # noqa: F811
    """«а как к нему подготовиться?» — свой вопрос о показанном: функция ответа — «ничего из этого», ход тот же, что без выключателя
    (подготовка показанного)."""

    def second(own: str) -> dict[str, Any]:
        turn = ("а как к нему подготовиться?", {"pending_reply": NONE_FIT, "turn_facet": PREPARE_FACET})
        return _dialog(mini_prod_world, monkeypatch, own, [(_OAK, {"turn_facet": PRICE_FACET}), turn])[1]

    on, off = second("on"), second("off")
    assert on["label"] == "PREPARE" and (on["label"], on["text"]) == (off["label"], off["text"]), (on["label"], on["text"][:200])


# Ход поглощён ответом на свой вопрос — одно определение для читателей типа вопроса (`turn_understanding.absorbed`).
@pytest.mark.parametrize(
    ("found", "absorbed"),
    [
        (("show_all", "pick", None, "llm"), True),
        (("slot_value", "answer", None, "llm"), True),
        (("operator_offer", "no", None, "fast"), True),
        (("show_all", "none", None, "llm"), False),  # свой вопрос о показанном
        (("show_all", "new_topic", None, "llm"), False),
        (("slot_value", "in_frame", None, "llm"), False),  # свой вопрос в рамке кнопки — его тип вопроса судится
        (("secondary_offer", "none", "ферритин", "fast"), False),  # хвост реплики идёт обычным путём
        (("show_all", "pick", None, "fallback"), False),  # сбой LLM — запасной путь правил
    ],
)
def test_absorbed_only_when_the_reply_answers_the_question(found, absorbed, monkeypatch):
    from messengers_router import pending_answer, turn_understanding
    from messengers_router.runtime_config import config as rcfg

    question, result, rest, source = found
    answer = pending_answer.Answer(question, pending_answer.Outcome(result), rest=rest, source=pending_answer.Source(source), text="x")
    token = pending_answer._CURRENT.set(answer)
    try:
        monkeypatch.setattr(rcfg, "MR_OWN_QUESTION_REPLY", "on", raising=False)
        assert turn_understanding.absorbed() is absorbed
        monkeypatch.setattr(rcfg, "MR_OWN_QUESTION_REPLY", "off", raising=False)
        assert turn_understanding.absorbed() is False
    finally:
        pending_answer._CURRENT.reset(token)


def test_other_service_is_not_a_row(mini_prod_world, monkeypatch):  # noqa: F811
    """«а ферритин?» — другая услуга: функция ответа — «другая тема»; ответ тот же, что без выключателя, — не строка списка ОАК."""

    def second(own: str) -> dict[str, Any]:
        turn = ("а ферритин?", {"pending_reply": outcome("new_topic"), "turn_facet": PRICE_FACET})
        return _dialog(mini_prod_world, monkeypatch, own, [(_OAK, {"turn_facet": PRICE_FACET}), turn])[1]

    on, off = second("on"), second("off")
    assert "Общий анализ крови" not in on["text"] and (on["label"], on["text"]) == (off["label"], off["text"]), on["text"][:200]


@pytest.mark.parametrize("reply", ["1", "2"])
def test_number_without_facet_llm_is_the_same_row(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Прод сейчас (`facet_by_llm = off`): номер — та же строка под on и off выключателя фикса (номер решали и правила)."""

    def second(own: str) -> dict[str, Any]:
        turns = [(_OAK, {}), (reply, {})]
        return _dialog(mini_prod_world, monkeypatch, own, turns, MR_FACET_BY_LLM="off")[1]

    on, off = second("on"), second("off")
    assert (on["label"], on["text"]) == (off["label"], off["text"]) and on["label"] == "PRICE", (on["text"][:160], off["text"][:160])
