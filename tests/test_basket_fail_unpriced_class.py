"""Корзина при сбое выбора LLM — варианты названиями без цен и оффер, итога нет (`basket_fail_unpriced`; решение владельца 27.10,
BUG-2026-10-27-LLM-QUEUE-GIVES-UP, шаг 3).

Проверка ответов без LLM 27.10 (узел 7): при занятой очереди выбор строк корзины сдавался, и код печатал свои варианты слабой
позиции с ценами и «Итого» по ним, включая чужие строки («Са общий» → «Общий белок 190 / … / Общий анализ мочи 280», «Итого
1 170–1 500») — 23 ответа на 1 643 хода. Карточка их не видит: в списке есть и нужная строка.

Инвариант класса (через обработчик эндпоинта, мини-прод, LLM выбора строк недоступна — отказ соединения или очередь): под `on`
строки с ценой в ответе — ровно строки сильных позиций (те же, что при выборе LLM «ни одна»); варианты кода слабой позиции —
названиями, без сумм; «Итого» нет; к ответу — оффер оператора вопросом («да» → оператор). Выбор LLM состоялся — ответ тот же, что
под off. off — как было (варианты с ценами).
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.response_builder import UNSUPPORTED_CATALOG_OPERATOR_OFFER
from tests.test_price_turns_golden import FIXTURE, run_dialog

_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"
_PRICE_LINE = re.compile(r"^(?:\d+\.\s*)?(.+?) — \d[\d\s]*(?:руб|₽)?", re.M)
# Реплика → слабая позиция, у которой при сбое код находит два варианта и больше.
_REPLIES = {
    "ttg_glucose": ("сколько стоит ттг и глюкоза", "ТТГ"),
    "ttg_ferritin": ("сколько стоит ттг и ферритин", "ТТГ"),
}


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


class _BasketLLM:
    """Выбор строк корзины: `fail` — сбой (отказ соединения или очередь), иначе ничего не выбрано («ни одна») или всё."""

    def __init__(self, fail: type[Exception] | None = None, pick_all: bool = False) -> None:
        self.fail, self.pick_all = fail, pick_all

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _BASKET_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        if self.fail is not None:
            raise self.fail("тест: выбор строк недоступен")
        picks = [int(head) for head, dot, _rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        return GenerateResponse(model="m", response=json.dumps({"picks": picks if self.pick_all else []}), done=True)


def _turns(world, monkeypatch, replies: list[str], *, mode: str, llm: _BasketLLM, session: str) -> list[dict]:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_BASKET_FAIL_UNPRICED", mode, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    with ev_world.FrozenWorld(world, "day"):
        return run_dialog(replies, f"bfu-{mode}-{session}")


def _priced(text: str) -> set[str]:
    return {m.group(1).strip() for m in _PRICE_LINE.finditer(text)}


@pytest.mark.parametrize("fail", [ConnectionError, llm_runtime.LLMQueueTimeoutError], ids=["down", "queue"])
@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_failed_row_choice_names_variants_without_prices_and_offers_operator(reply, fail, mini_prod_world, monkeypatch):
    text, weak = _REPLIES[reply]
    [off] = _turns(mini_prod_world, monkeypatch, [text], mode="off", llm=_BasketLLM(fail), session=f"off-{reply}")
    [on] = _turns(mini_prod_world, monkeypatch, [text], mode="on", llm=_BasketLLM(fail), session=f"on-{reply}-{fail.__name__}")
    [rejected] = _turns(mini_prod_world, monkeypatch, [text], mode="on", llm=_BasketLLM(), session=f"none-{reply}")

    guessed = {name for name in _priced(off["text"]) if name.startswith(weak) or weak in name}
    assert len(guessed) >= 2, off["text"]  # off — как было: варианты кода с ценами (дефект класса)
    assert _priced(on["text"]) == _priced(rejected["text"]), on["text"]  # с ценой — только сильные позиции
    for name in guessed:
        assert name in on["text"], (name, on["text"])  # вариант назван — без суммы
    assert "Итого" not in on["text"], on["text"]
    assert on["text"].rstrip().endswith(UNSUPPORTED_CATALOG_OPERATOR_OFFER), on["text"]


def test_yes_to_the_basket_offer_hands_off(mini_prod_world, monkeypatch):
    first, second = _turns(
        mini_prod_world, monkeypatch, [_REPLIES["ttg_glucose"][0], "да"], mode="on", llm=_BasketLLM(ConnectionError), session="yes"
    )

    assert first["text"].rstrip().endswith(UNSUPPORTED_CATALOG_OPERATOR_OFFER)
    assert second["handoff"], second


@pytest.mark.parametrize("reply", sorted(_REPLIES))
def test_when_the_model_answers_the_basket_is_the_same_as_off(reply, mini_prod_world, monkeypatch):
    text, _weak = _REPLIES[reply]
    [off] = _turns(mini_prod_world, monkeypatch, [text], mode="off", llm=_BasketLLM(pick_all=True), session=f"ok-off-{reply}")
    [on] = _turns(mini_prod_world, monkeypatch, [text], mode="on", llm=_BasketLLM(pick_all=True), session=f"ok-on-{reply}")

    assert on["text"] == off["text"]


def test_switch_defaults_to_off():
    from agent_logic_2 import config as host_config

    assert host_config.settings.messenger_router.basket_fail_unpriced == "off"
