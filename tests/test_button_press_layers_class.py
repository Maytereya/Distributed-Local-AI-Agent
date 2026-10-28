"""Нажатие листа меню на конфиге деплоя 19: смысл задаёт кнопка, нажатие — не реплика (проверка всех слоёв кнопок 28.10).

Многоходовая проверка кнопок через эндпоинт на коде деплоя 19 (`~/Dev/handoffs/2026-10-28_buttons_all_layers.md`) нашла два класса:

* BUG-2026-10-28-BOOKING-NEEDED-WALKIN-JUDGE — на ходе нажатия LLM-судья судит слова пациента, которых нет. «Нужно ли
  записываться» (раздел «Сдать анализы», смысл — запись на анализы) под `walkin_by_llm = on` (строка конфига деплоя 19)
  спрашивает судью «без записи?» о пустом сообщении (подпись кнопки, верно, не содержание — `OrchestratorContext.text`),
  судья отвечает «нет», и ответ кнопки теряет главное — «Анализы выполняются без записи» (стенд: деплой 18 — 29 нажатий из
  29 с фразой, деплой 19 — 0 из 29).
  Инвариант: на ходе нажатия листа `slot` / `handoff` / `hint` LLM не зовётся — тему и смысл задаёт кнопка, судить нечего;
  ответ кнопки «Нужно ли записываться» — фраза «без записи» при любом ответе судьи.
* BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT — анти-залип считает нажатие другой кнопки с тем же вопросом «другой репликой
  пациента»: «Сколько стоит анализ» → «Цена анализа» дописывает «Похоже, мне не удаётся помочь… Перевести на оператора?», и
  следующее «ферритин» судья читает согласием — перевод на оператора вместо цены.
  Инвариант: нажатие листа не засчитывается повтором — вопрос кнопки, заданный нажатием, не несёт оффера оператора.

Конфиг — прод деплоя 19 (умолчания кода + 13 строк runbook §3). Через обработчик эндпоинта, срезы гейта; LLM — подделка
настоящего типа: метки промптов, остальное — отказ соединения. Известные, ещё не закрытые провалы — xfail strict (чинить —
по слову владельца).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from ollama import GenerateResponse

from messengers_router import button_menu, choice, llm_runtime
from tests.test_button_actions import GATEWAY_V1_TITLES

# Деплой 19: строки `[MESSENGER_ROUTER]` поверх умолчаний кода (~/Dev/handoffs/2026-10-28_deploy19_runbook.md §3).
PROD19 = {
    "MR_TEST_ASSIST_PRICE": "on",
    "MR_WALKIN_BY_LLM": "on",
    "MR_BARE_NAME_OVERVIEW": "on",
    "MR_BOOKING_OPTIONS_PRICE": "on",
    "MR_ADDRESS_NAMED_STREET": "on",
    "MR_STRONG_FAMILY_LETTER": "on",
    "MR_BUTTON_ANSWER_TOPIC": "slot",
    "MR_BUTTON_REPLY_JUDGE": "on",
    "MR_BUTTON_SIDE_ANSWER": "on",
    "MR_LLM_TURN_DEADLINE_S": 50,
    "MR_CLASSIFIER_TIMEOUT_S": 20,
    "MR_BASKET_FAIL_UNPRICED": "on",
    "MR_LLM_FAIL_OFFER": "address_followup,facet",
}
_WALKIN_MARK = "хочет ли пациент сдать анализы или сделать ЭКГ"
_WALKIN_OTHER = json.dumps({"picks": [], "outcome": "other"})  # как ответил живой судья на пустое сообщение нажатия
_WALKIN_SENTENCE = "без записи, в порядке живой очереди"
_OFFER = "Перевести на оператора?"


class _RecordingLLM:
    """Пишет каждый промпт; отвечает по метке (`answers[метка]`), без метки — отказ соединения (как упавший ollama)."""

    def __init__(self, answers: dict[str, str] | None = None):
        self.answers = answers or {}
        self.prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        self.prompts.append(seen)
        for mark, answer in self.answers.items():
            if mark in seen:
                return GenerateResponse(model="m", response=answer, done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _dialog(monkeypatch, turns: list[Any], llm: _RecordingLLM) -> list[dict[str, Any]]:
    """Ходы через `messenger_generate_once`: `(button_id, подпись)` — нажатие, строка — реплика. На ход — текст, перевод,
    висящий оффер и промпты LLM этого хода."""

    from messengers_router.endpoint import MessengerGenerateRequest, messenger_generate_once
    from messengers_router.memory import MemoryStore
    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import Services

    session = f"layers-{json.dumps(turns, ensure_ascii=False)}"

    async def run() -> list[dict[str, Any]]:
        memory, services, out = MemoryStore(), Services(), []
        services.ensure_background_refresh_started = lambda: None
        for turn in turns:
            button_id, text = turn if isinstance(turn, tuple) else ("", turn)
            seen = len(llm.prompts)
            req = MessengerGenerateRequest(session_id=session, text=text, debug=True, button_id=button_id)
            body = json.loads((await messenger_generate_once(req, memory=memory, services=services)).body)
            state = dict(memory.get(session).last_entities or {})
            out.append(
                {
                    "text": str(body.get("text") or ""),
                    "handoff": bool(body.get("handoff")),
                    "offer_pending": bool(state.get("_operator_offer_pending")),
                    "prompts": llm.prompts[seen:],
                }
            )
        return out

    with monkeypatch.context() as patch:
        for key, value in {"MR_CHOOSE_CACHE_TTL_S": 0, **PROD19}.items():
            patch.setattr(cfg, key, value, raising=False)
        patch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
        patch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
        choice._CACHE.clear()
        return asyncio.run(run())


def _press(button_id: str) -> tuple[str, str]:
    return (button_id, GATEWAY_V1_TITLES[button_id])


# --- класс 1: на ходе нажатия листа со своим смыслом LLM не судит слова пациента — их нет --------------------------------

_JUDGED_ON_PRESS = {"menu.tests.booking_needed"}  # BUG-2026-10-28-BOOKING-NEEDED-WALKIN-JUDGE, открыт
_OWN_MEANING = sorted(b for b, a in button_menu.BUTTONS.items() if a.mode in ("slot", "handoff", "hint"))


@pytest.mark.parametrize(
    "button_id",
    [
        pytest.param(b, marks=pytest.mark.xfail(strict=True, reason="BUG-2026-10-28-BOOKING-NEEDED-WALKIN-JUDGE"))
        if b in _JUDGED_ON_PRESS
        else b
        for b in _OWN_MEANING
    ],
)
def test_press_asks_llm_nothing(button_id, monkeypatch):
    (press,) = _dialog(monkeypatch, [_press(button_id)], _RecordingLLM())

    asked = [p.rsplit("Сообщение пациента:", 1)[-1][:60] for p in press["prompts"]]
    assert not press["prompts"], f"на ходе нажатия LLM судила слова пациента, которых нет: {asked}"


@pytest.mark.xfail(strict=True, reason="BUG-2026-10-28-BOOKING-NEEDED-WALKIN-JUDGE")
def test_booking_needed_press_says_walkin_whatever_judge_says(monkeypatch):
    (press,) = _dialog(monkeypatch, [_press("menu.tests.booking_needed")], _RecordingLLM({_WALKIN_MARK: _WALKIN_OTHER}))

    assert _WALKIN_SENTENCE in press["text"], press["text"]


def test_booking_needed_press_says_walkin_when_llm_is_down(monkeypatch):
    # Встречная сторона: без ответа судьи (сбой LLM) решает детектор — фраза есть, как на деплое 18.
    (press,) = _dialog(monkeypatch, [_press("menu.tests.booking_needed")], _RecordingLLM())

    assert _WALKIN_SENTENCE in press["text"], press["text"]


# --- класс 2: нажатие листа — не повтор реплики (анти-залип) ------------------------------------------------------------

# Листы, чьё нажатие задаёт один и тот же вопрос (одна тема и одна подсказка, без смысла-сущностей): «Сколько стоит анализ» и
# «Цена анализа», «Ввести данные» и «Результат не пришёл».
_ASKING = {b: a for b, a in button_menu.BUTTONS.items() if a.mode == "slot" and not a.entities}
_SAME_QUESTION = sorted(
    (a, b) for a in _ASKING for b in _ASKING if a != b and (_ASKING[a].label, _ASKING[a].text) == (_ASKING[b].label, _ASKING[b].text)
)


def test_same_question_leaves_exist():
    # Без пар класс пуст и тест ниже ничего не проверяет: таблица меню сменилась — пересобрать пары.
    assert ("menu.price.test", "menu.tests.price") in _SAME_QUESTION
    assert ("menu.results.lookup", "menu.results.missing") in _SAME_QUESTION


@pytest.mark.parametrize(
    ("first", "second"),
    [pytest.param(*pair, marks=pytest.mark.xfail(strict=True, reason="BUG-2026-10-28-BUTTON-PRESS-AS-REPEAT")) for pair in _SAME_QUESTION],
)
def test_second_leaf_with_same_question_is_not_a_repeat(first, second, monkeypatch):
    _, pressed = _dialog(monkeypatch, [_press(first), _press(second)], _RecordingLLM())

    assert _OFFER not in pressed["text"], pressed["text"]
    assert not pressed["offer_pending"]


@pytest.mark.parametrize("button_id", sorted(_ASKING))
def test_same_leaf_twice_is_not_a_repeat(button_id, monkeypatch):
    # Встречная сторона, держится и сейчас: та же подпись — «тот же вопрос пациента», анти-залип его не считает.
    first, pressed = _dialog(monkeypatch, [_press(button_id), _press(button_id)], _RecordingLLM())

    assert _OFFER not in pressed["text"], pressed["text"]
    assert pressed["text"] == first["text"]
