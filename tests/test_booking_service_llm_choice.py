"""Цель записи на услугу выбирает LLM, правила только собирают кандидатов (BUG-2026-10-18-BOOKING-SERVICE-BY-LETTERS).

Найдено встречным свипом 18.10 (подбаг B класса 5): «запишите на гастроскопию» → «Есть возможность записи на Гастрин…»
(анализ, 750 руб.), «запишите на массаж» → «Массаж век». Лексический резолвер прайса решал услугу записи по буквам
(«гаст|роскопия» и «гаст|рин»), его догадка шла дальше как «exact» без суда смысла. У цены этот класс закрыт выбором
строки LLM (решение владельца 30.09, `_price_select_llm`); теперь тот же выбор зовёт и запись
(`services._booking_service_select.judge_booking_service` из `entity_grounder`).

Инварианты класса (через эндпоинт, мир гейта; LLM — оракул выбора строки прайса или недоступна):
- LLM решает по смыслу: подтвердила — догадка правил остаётся; назвала другую услугу — цель она (ФГДС по термину
  второго круга); назвала несколько — названия на выбор, без записи; не нашла — честный отказ с оффером оператора;
- LLM недоступна на слабом совпадении — названия на выбор пациенту (решение владельца 18.10, как L-13 у цены), нет
  названий — отказ с оффером оператора; чужой услуги в записи нет;
- сильное совпадение по данным (дословное название строки) — без LLM; название, написанное из списка, продолжает запись;
- запись к специальности выбор не зовёт.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

import pytest
from ollama import GenerateResponse

from messengers_router import llm_runtime

_PRICE_PROMPT_MARK = "ответить пациенту на вопрос о цене"
_FGDS = "Эзофагогастродуоденоскопия (ФГДС)"
_BRANCH_QUESTION = "Какой филиал вам удобен?"
_CHOICE_HEAD = "Уточните, пожалуйста, на какую услугу вас записать:"
_REFUSAL = "эту услугу я не нашёл в системе онлайн-записи"


class _PriceOnlyLLM:
    """LLM отвечает только на выбор строки прайса; прочие места — «недоступна», как в мире гейта.

    `decide(rows_by_name)` → (номера, термины второго круга) в номерах списка, который видит модель."""

    def __init__(self, decide: Callable[[dict[str, int]], tuple[list[int], list[str]]], *, reason: str = "other_service") -> None:
        self.decide = decide
        self.reason = reason  # код причины пустого выбора
        self.price_calls = 0

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _PRICE_PROMPT_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.price_calls += 1
        rows: dict[str, int] = {}
        for line in seen.splitlines():
            head, dot, rest = line.partition(". ")
            if dot and head.isdigit() and " — " in rest:
                rows[rest.rsplit(" — ", 1)[0]] = int(head)
        picks, terms = self.decide(rows)
        answer = {"picks": picks, "reason": "none" if picks else self.reason, "search_terms": terms}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


def _fgds_by_term(rows: dict[str, int]) -> tuple[list[int], list[str]]:
    if _FGDS in rows:
        return [rows[_FGDS]], []
    return [], ["ФГДС"]


def _dialog(turns: tuple[str, ...], session_id: str) -> list[dict[str, Any]]:
    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.services import Services

    async def run() -> list[dict[str, Any]]:
        memory, services = MemoryStore(), Services()
        out = []
        for text in turns:
            req = MessengerGenerateRequest(session_id=session_id, text=text, debug=True)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            entities = dict(memory.get(session_id).last_entities or {})
            flags = (((body.get("state_update") or {}).get("debug") or {}).get("decision") or {}).get("flags") or []
            out.append({"text": str(body.get("text") or ""), "handoff": bool(body.get("handoff")), "entities": entities, "flags": flags})
        return out

    return asyncio.run(run())


def _choice_names(text: str) -> list[str]:
    return [line[2:].strip() for line in text.splitlines() if line.startswith("— ")]


def test_llm_picks_fgds_not_gastrin_by_letters(monkeypatch):
    llm = _PriceOnlyLLM(_fgds_by_term)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    [turn] = _dialog(("запишите на гастроскопию",), "bsc-fgds")
    assert _FGDS in turn["text"] and _BRANCH_QUESTION in turn["text"], turn["text"]
    assert "Гастрин" not in turn["text"]
    assert turn["entities"].get("service_name") == _FGDS
    assert llm.price_calls == 2  # первый круг — «нет», второй — по термину модели


def test_llm_none_fit_is_an_honest_refusal(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(lambda rows: ([], [])))
    [turn] = _dialog(("запишите на гастроскопию",), "bsc-none")
    assert _REFUSAL in turn["text"] and "Гастрин" not in turn["text"], turn["text"]
    # Фраза пациента в состоянии остаётся (подавитель A′-1 держит то, что резолвер находит по буквам) — но не «Гастрин».
    assert turn["entities"].get("service_name") != "Гастрин"


def test_llm_several_services_are_offered_by_name(monkeypatch):
    def massages(rows: dict[str, int]) -> tuple[list[int], list[str]]:
        return [n for name, n in rows.items() if "массаж" in name.lower()][:3], []

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(massages))
    [turn] = _dialog(("запишите на массаж",), "bsc-massage")
    names = _choice_names(turn["text"])
    assert turn["text"].startswith(_CHOICE_HEAD) and len(names) >= 2, turn["text"]
    assert all("массаж" in name.lower() for name in names), names
    assert not turn["entities"].get("service_name")
    expects = turn["entities"].get("_expects") or {}
    assert expects.get("type") == "booking_service" and expects.get("options") == names, expects


def test_llm_too_general_asks_which_service(monkeypatch):
    # «массаж» — группа услуг: правила молча записывали на «Массаж век»; LLM говорит «слишком общее» — пациента просят
    # назвать услугу, а цель записи не ставится.
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(lambda rows: ([], []), reason="too_general"))
    [turn] = _dialog(("запишите на массаж",), "bsc-general")
    assert turn["text"].startswith("Уточните, пожалуйста, какую именно услугу записать"), turn["text"]
    assert _BRANCH_QUESTION not in turn["text"]
    assert not turn["entities"].get("service_name")
    assert (turn["entities"].get("_pending") or {}).get("label") == "APPOINTMENT"


@pytest.mark.parametrize("opening", ["запишите на гастроскопию", "хочу записаться на гастроскопию", "запишите на узи щитовидки"])
def test_without_llm_a_weak_guess_is_asked_not_booked(opening):
    # Мир гейта: LLM недоступна. Слабая догадка правил («Гастрин») не становится целью записи молча: пациенту — названия
    # на выбор или «Похоже, вы имели в виду …?»; филиалы на догадку не предлагаются.
    [turn] = _dialog((opening,), f"bsc-down-{opening}")
    assert _BRANCH_QUESTION not in turn["text"], turn["text"]
    assert turn["text"].startswith(_CHOICE_HEAD) or "имели в виду" in turn["text"], turn["text"]
    assert not turn["entities"].get("service_name"), turn["entities"]
    assert not turn["handoff"]


def test_without_llm_names_are_offered_and_a_named_one_continues_the_booking():
    first, second, third = _dialog(
        ("запишите на узи почек", "Ультразвуковое исследование почек и надпочечников", "пр.Ленина, 5"), "bsc-down-pick"
    )
    names = _choice_names(first["text"])
    assert first["text"].startswith(_CHOICE_HEAD) and "Ультразвуковое исследование почек и надпочечников" in names, first["text"]
    assert _BRANCH_QUESTION in second["text"], second["text"]
    assert second["entities"].get("service_name") == "Ультразвуковое исследование почек и надпочечников"
    assert "Когда вам удобно прийти?" in third["text"], third["text"]


def test_verbatim_catalog_name_needs_no_llm(monkeypatch):
    llm = _PriceOnlyLLM(lambda rows: ([], []))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    [turn] = _dialog((f"Запишите на {_FGDS}",), "bsc-verbatim")
    assert _FGDS in turn["text"] and _BRANCH_QUESTION in turn["text"], turn["text"]
    assert llm.price_calls == 0


def test_specialty_booking_does_not_call_the_choice(monkeypatch):
    llm = _PriceOnlyLLM(lambda rows: ([], []))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    [turn] = _dialog(("хочу записаться к урологу",), "bsc-specialty")
    assert _BRANCH_QUESTION in turn["text"], turn["text"]
    assert llm.price_calls == 0


# --- «Где сделать …» (ADDRESS): та же услуга из реплики судится LLM, а не буквами ---------------------------------


def test_address_takes_the_service_chosen_by_llm(monkeypatch):
    # «лфк» по буквам — «Комплексная операция по эндопротезированию…» и её адреса; LLM называет занятие ЛФК.
    def lfk(rows: dict[str, int]) -> tuple[list[int], list[str]]:
        picks = [n for name, n in rows.items() if "лечебной физкультур" in name.lower()][:1]
        return (picks, []) if picks else ([], ["лечебная физкультура"])

    llm = _PriceOnlyLLM(lfk)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    [turn] = _dialog(("где можно сделать лфк",), "bsc-address-lfk")
    service = str(turn["entities"].get("service_name") or "")
    assert "физкультур" in service.lower() and "эндопротез" not in service.lower(), turn["entities"]
    assert llm.price_calls >= 1


def test_address_drops_a_service_the_llm_did_not_confirm(monkeypatch):
    # Несколько услуг («узи сердца» — эхо взрослым и плода): адреса не фильтруются по одной чужой строке.
    def echo(rows: dict[str, int]) -> tuple[list[int], list[str]]:
        return [n for name, n in rows.items() if "эхокардиограф" in name.lower()][:3], []

    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _PriceOnlyLLM(echo))
    [turn] = _dialog(("где сделать узи сердца",), "bsc-address-echo")
    assert "плода" not in str(turn["entities"].get("service_name") or ""), turn["entities"]
    assert "entity_dropped_unverified_service_name" in turn["flags"]


def test_address_without_llm_is_as_before():
    # LLM недоступна — как было, по правилам (решение «названия на выбор» — о записи).
    [turn] = _dialog(("где сделать узи щитовидки",), "bsc-address-down")
    assert turn["entities"].get("service_name") == "Ультразвуковое исследование щитовидной железы", turn["entities"]
