"""Класс «ответ не на заданный вопрос» — BUG-2026-10-30-ANSWER-NOT-TO-QUESTION (решение владельца 30.10 по g0673).

«В какой день недели можно приносить анализы кала и мочи?» → памятка Cito ОАМ (стенд, `facet_by_llm = on`: прогоны `d23on_dev`,
`vbon_dev`, `d22s5on_dev`; без него — «К какому анализу нужна подготовка?»). Памятку владелец оставил, дефект — что на сам вопрос
(«когда / в какой день / во сколько») ответа нет: данные показаны, вопрос пациента без ответа.

Инвариант класса: на вопрос «когда / в какой день / во сколько принести или сдать X» в ответе есть ответ на «когда» (время пунктов
забора — часы из данных филиалов) или честный оффер оператора, а не только памятка подготовки и не переспрос «к какому анализу
подготовка». Тему хода задаёт классификатор (PREPARE — биоматериал, «приносить»), а тип вопроса («где или в какое время» —
`turn_understanding.FACETS`, `where`) у темы подготовки не читается: `router._route_by_turn_facet` судит только ход цены, у
`where` нет своего «когда» (читатель — список адресов), а ответ часами есть только у реплики после адресной кнопки
(`address_flow.followup`, исход `hours`).

Харнесс — `tests/understanding_contract.py` (подделка LLM настоящего типа на границе `llm_runtime`): тест задаёт, что LLM поняла —
тему PREPARE и тип вопроса «где или в какое время», — и проверяет ответ. Выбор памятки (не задача `choose`, свой промпт) подделка
отвечает первой памяткой списка: в мини-прод мире памяток МИС нет, есть общая памятка клиники о сдаче крови — для класса важно,
что читатель — памятка, а не какая. Конфиг — деплой 21 харнесса + ключи деплоя 22, `facet_by_llm` — off (прод) и on (кандидат 23).
Текущее поведение — `xfail strict`; контроль — реплика после адресной кнопки: читатель «когда» есть, проверка инварианта его принимает
(ответ часами и честный оффер).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from ollama import GenerateResponse

from tests import understanding_contract as uc
from tests.understanding_contract import d21_world  # noqa: F401 — фикстура

_BUG = "BUG-2026-10-30-ANSWER-NOT-TO-QUESTION: «когда / в какой день» у темы подготовки — памятка или переспрос, не ответ"
# Ключи деплоя 22 поверх деплоя 21 харнесса (как `test_own_list_reply_class.D22_FACET`, без `facet_by_llm` — он параметр).
D22: dict[str, Any] = {
    "MR_FRAME_POLICY": "on",
    "MR_BUTTON_FRAME_CHECK": "judge_leaf",
    "MR_EXPECTS_BY_UNDERSTANDING": "on",
    "MR_PRICE_NOT_FOUND_BY_DATA": "on",
}
WHERE_WHEN = uc.picked("Узнать, где или в какое время можно сдать анализ или пройти услугу")
PREPARE = uc.picked("Узнать, как подготовиться к анализу или услуге")
_MEMO_MARK = "ответить пациенту, как подготовиться к анализу или процедуре"  # промпт `prepare_document_select`
_HOURS = re.compile(r"\b\d{1,2}[.:]\d{2}\b")  # «07:00–18:30» (прод), «7.00-20.00» (мини-прод)


class _MemoLLM(uc.UnderstandingLLM):
    """Подделка харнесса + выбор памятки подготовки (`_prepare_select_llm`, ответ `{"match": номер}`): первая памятка списка."""

    async def generate(self, *, prompt: str = "", system: str | None = None, format: Any = None, **kwargs: Any) -> GenerateResponse:
        if _MEMO_MARK in (prompt or ""):
            self.asked.append("prepare_memo")
            return GenerateResponse(model="m", response=json.dumps({"match": 1}), done=True)
        return await super().generate(prompt=prompt, system=system, format=format, **kwargs)


def _when_violations(record: dict[str, Any]) -> list[str]:
    """Нарушение инварианта на последнем ходе: нет ни ответа на «когда» (часы), ни честного оффера (не о памятке подготовки)."""

    text = record["text"]
    if _HOURS.search(text):
        return []
    low = text.lower()
    offer = record["handoff"] or "перевести на оператора" in low
    if offer and "памятк" not in low and "подготов" not in low:
        return []
    return [f"нет ответа на «когда» (тема {record['label']}, план {record['plan']}): {text[:200]!r}"]


def _run(world: Any, monkeypatch: Any, turns: list[uc.Turn], facet: str) -> dict[str, Any]:
    monkeypatch.setattr(uc, "UnderstandingLLM", _MemoLLM)
    return uc.run(world, monkeypatch, turns, cfg={**D22, "MR_FACET_BY_LLM": facet}, session=f"when-{facet}-{turns[-1].text}")[-1]


# Вопрос «когда» о названном анализе: LLM поняла тему PREPARE и тип вопроса «где или в какое время».
_WHEN = {
    "g0673_day": "В какой день недели можно приносить анализы кала и мочи?",
    "when_urine": "Когда можно принести мочу на общий анализ?",
    "time_blood": "Во сколько можно сдать кровь на ферритин?",
}
_SPANS = {"g0673_day": ("анализы кала", "мочи"), "when_urine": ("мочу",), "time_blood": ("ферритин",)}

DIALOGS: dict[str, list[uc.Turn]] = {
    **{case: [uc.say(text, services=_SPANS[case], topic="PREPARE", choices={"turn_facet": WHERE_WHEN})] for case, text in _WHEN.items()},
    # После памятки подготовки — вопрос «когда» о показанном.
    "after_memo": [
        uc.say("как подготовиться к общему анализу мочи?", services=("общему анализу мочи",), topic="PREPARE", choices={"turn_facet": PREPARE}),
        uc.say("а в какой день её можно принести?", topic="PREPARE", pending="new_topic", choices={"turn_facet": WHERE_WHEN}),
    ],
    # После списка адресов на свободный вопрос (без кнопки: адресной метки нет, `address_followup` не зовётся).
    "after_addresses": [
        uc.say("где можно сдать анализы?", topic="ADDRESS", choices={"turn_facet": WHERE_WHEN}),
        uc.say("а в какой день недели можно принести кал?", services=("кал",), topic="PREPARE", pending="new_topic", choices={"turn_facet": WHERE_WHEN}),
    ],
}


@pytest.mark.xfail(strict=True, reason=_BUG)
@pytest.mark.parametrize("facet", ["off", "on"])
@pytest.mark.parametrize("case", sorted(DIALOGS))
def test_when_question_gets_when_answer(case, facet, d21_world, monkeypatch):  # noqa: F811
    assert _when_violations(_run(d21_world, monkeypatch, DIALOGS[case], facet)) == []


# Контроль (встречная проверка инварианта): после адресной кнопки «когда» читает `address_followup` — часы пунктов забора; чего в
# данных филиалов нет (праздники) — честный оффер. Оба проходят проверку; если она станет красной здесь — сломан контроль, не класс.
_BUTTON_WHERE = uc.press("menu.tests.where")
CONTROLS = {
    "hours": ("В какой день недели можно приносить анализы кала и мочи?", "hours"),
    "no_data_offer": ("А в праздники можно принести кал?", "no_data"),
}


@pytest.mark.parametrize("facet", ["off", "on"])
@pytest.mark.parametrize("case", sorted(CONTROLS))
def test_when_after_address_button_is_answered(case, facet, d21_world, monkeypatch):  # noqa: F811
    text, followup = CONTROLS[case]
    reply = uc.say(text, pending="answer", frame="same", topic="ADDRESS", choices={"address_followup": uc.outcome(followup), "turn_facet": WHERE_WHEN})
    record = _run(d21_world, monkeypatch, [_BUTTON_WHERE, reply], facet)
    assert "choose_address_followup" in record["asked"], record["asked"]
    assert _when_violations(record) == []
