"""Корзина на модуле выбора (ход (в), L-15a) — через обработчик эндпоинта, мини-прод.

Выключателя `choose_basket` нет с 16.10 (решение владельца, §5 итога облака): строки слабых позиций выбирает LLM всегда.

Инварианты класса:
- у позиции корзины — только строки той услуги: вторая строка «на всякий случай» (чужая — «Глюкоза в
  моче» к глюкозе) не показывается;
- сильное совпадение по данным (дословное название) — без LLM;
- слабая позиция — выбор модели одним вызовом на весь список; строки, которые модель не выбрала, не
  показываются; позиция без выбора — честное «Не распознал», а не цена по буквам;
- сбой LLM: сильные позиции — их строки по данным, слабые — строки корзины правил; сильных нет —
  корзина правил целиком; off — прежняя корзина правил.
"""

from __future__ import annotations

import json

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from tests.test_price_turns_golden import FIXTURE, _golden, run_dialog

_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory):
    return ev_world.build_world(FIXTURE, "20261008", tmp_path_factory.mktemp("mini_prod") / "world")


class _BasketLLM:
    """Отвечает только на выбор строк корзины: номера строк, чьё название начинается с заданного."""

    def __init__(self, wanted: tuple[str, ...] = (), fail: bool = False) -> None:
        self.wanted, self.fail, self.prompts = wanted, fail, []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _BASKET_MARK not in seen:
            raise ConnectionError("тест: LLM недоступна для этого места")
        self.prompts.append(seen)
        if self.fail:
            raise ConnectionError("ollama down")
        picks = []
        for line in prompt.splitlines():
            head, dot, rest = line.partition(". ")
            name = rest.split(" → ", 1)[-1]
            if dot and head.isdigit() and any(name.startswith(w) for w in self.wanted):
                picks.append(int(head))
        return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)


def _basket(monkeypatch, llm: _BasketLLM | None = None) -> None:
    if llm is not None:
        monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)


def _ask(question: str, session: str, world_root) -> str:
    with ev_world.FrozenWorld(world_root, "day"):
        [turn] = run_dialog([question], session)
    return turn["text"]


def test_strong_positions_show_only_their_rows_without_llm(mini_prod_world, monkeypatch):
    llm = _BasketLLM()
    _basket(monkeypatch, llm)
    text = _ask("сколько стоит ферритин и глюкоза", "basket-strong", mini_prod_world)
    assert "Ферритин — 530" in text and "Глюкоза — 190" in text
    assert "в моче" not in text
    assert llm.prompts == []


def test_weak_position_shows_only_the_rows_the_model_picked(mini_prod_world, monkeypatch):
    llm = _BasketLLM(wanted=("ТТГ (TSH)",))
    _basket(monkeypatch, llm)
    text = _ask("сколько стоит ттг и ферритин", "basket-weak", mini_prod_world)
    assert "ТТГ (TSH) тиреотропный гормон — 400" in text and "Ферритин — 530" in text
    assert "Антитела к рецепторам ТТГ" not in text
    assert len(llm.prompts) == 1 and "«ттг» → ТТГ (TSH)" in llm.prompts[0]


def test_position_the_model_rejected_is_named_not_priced_by_letters(mini_prod_world, monkeypatch):
    _basket(monkeypatch, _BasketLLM(wanted=()))
    text = _ask("сколько стоит ттг и ферритин", "basket-reject", mini_prod_world)
    assert "Ферритин — 530" in text
    assert "ТТГ (TSH) тиреотропный гормон — 400" not in text and "Не распознал: ттг" in text


def test_llm_failure_keeps_strong_positions_by_data(mini_prod_world, monkeypatch):
    # Разбор реплики — единственный режим (T7 шаг 3б): при сбое LLM сильная позиция — её строки по данным; прежней
    # корзины правил со второй строкой «на всякий случай» («Глюкоза в моче») нет.
    _basket(monkeypatch, _BasketLLM(fail=True))
    after = _ask("сколько стоит ттг и глюкоза", "basket-fail2", mini_prod_world)
    assert "Глюкоза — 190" in after and "Глюкоза в моче" not in after
    assert "ТТГ (TSH) тиреотропный гормон — 400" in after  # слабая позиция — варианты кода


def test_no_basket_switch():
    """Выключателя `choose_basket` нет с 16.10 (§5 итога облака): строки слабых позиций всегда выбирает LLM."""

    from agent_logic_2 import config
    from messengers_router.services import _basket_select

    assert not hasattr(config, "MR_CHOOSE_BASKET") and not hasattr(_basket_select, "basket_mode")


def test_default_is_the_golden_basket(mini_prod_world):
    """Умолчание = прод (`on`), LLM недоступна: эталон `list_text_and` — сильные позиции по данным."""

    text = _ask("сколько стоит ферритин и глюкоза", "basket-default", mini_prod_world)
    assert text == _golden()["list_text_and"][0]["text"]
    assert "в моче" not in text
