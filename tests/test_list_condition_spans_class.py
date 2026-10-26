"""Условие у позиции списка на пути прода: разметка слов (`turn_spans_llm = on`) оторвала «по ОМС» от названия — позиция
получает розничную цену (RISK-2026-10-26-LIST-CONDITION-CUT-BY-SPANS, класс П2).

Суд условия у сильной позиции корзины (`phrase_price_items`, BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS) читает
`ParsedPhrase.beside` — слова фрагмента вне выбранной фразы. Поле заполняет только прежний генератор окон
(`_turn_phrases.result` → `_with_beside`); разметка слов (`_turn_spans._result`, прод с деплоя 17) его не заполняет, а гарда
П2 на уровне реплики у корзины нет (`prices._multi_price_payload` идёт раньше всех гардов одной услуги). Разметка по правилу 3
своего промпта держит условие внутри фразы — тогда позиция слабая и её судят (честное «Не распознал»); оторвала условие
(«ферритин» из «ферритин по ОМС», живая LLM у одной услуги — 59 из 82 свипа П2, карточка Ш7) — сильная позиция печатает
цену по данным, LLM строк о ней не спрашивают вовсе; слабая — идёт к LLM без пометки «[добавлено: …]», правило 5 промпта
корзины не срабатывает.

Инварианты класса (через обработчик эндпоинта, мини-прод, выключатели прода деплоя 17; `price_by_parse` off — прод
деплоя 17, on — деплой 18; оракул LLM: разметка — фразы случая, выбор строк корзины — правило 5 промпта):
- условие внутри фразы — позиция без цены, «Не распознал» словами пациента, соседняя — с ценой (держится);
- условие оторвано разметкой — позиция без розничной цены (xfail strict: дефект, не исправлен);
- соседняя позиция при оторванном условии — с ценой (встречный: фикс не должен глушить весь список);
- LLM недоступна — у позиции с условием розничной цены нет (держится);
- список без условия и со словом разговора («первый раз») — цены, как есть (встречный).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_stage2_turns import PROD_D17
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

SWITCH = ("off", "on")  # `price_by_parse`: off — прод деплоя 17, on — набор деплоя 18
_SPANS_MARK = "Слова сообщения с номерами:"
_BASKET_MARK = "ответить на вопрос о ценах нескольких услуг"
_ONE_MARK = "ответить пациенту на вопрос о цене"
_CONDITION = re.compile(r"омс|полис", re.IGNORECASE)  # условия класса: в прайсе их нет (П2)
_FERRITIN = "Ферритин — 530"
_GLUCOSE = "Глюкоза — 190"
_TSH = "ТТГ (TSH) тиреотропный гормон — 400"

# реплика, фразы разметки по правилу 3 (условие внутри), фразы с оторванным условием, цена позиции с условием, цена соседа
CLASS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str, str]] = {
    "omc_comma": ("ферритин по ОМС, глюкоза сколько стоит", ("ферритин по ОМС", "глюкоза"), ("ферритин", "глюкоза"), _FERRITIN, _GLUCOSE),
    "polis_and": ("сколько стоит глюкоза и ферритин по полису", ("глюкоза", "ферритин по полису"), ("глюкоза", "ферритин"), _FERRITIN, _GLUCOSE),
    "tsh_omc": ("ферритин, ТТГ по омс", ("ферритин", "ТТГ по омс"), ("ферритин", "ТТГ"), _TSH, _FERRITIN),
}
# без условия класса: что разметка ни отметит, цены остаются (фикс не должен глушить разговор и чистые списки)
CONTROLS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "plain_and": ("сколько стоит ферритин и глюкоза", ("ферритин", "глюкоза"), (_FERRITIN, _GLUCOSE)),
    "plain_comma": ("ферритин, ТТГ", ("ферритин", "ТТГ"), (_FERRITIN, _TSH)),
    "conversational": ("сколько стоит ферритин первый раз, глюкоза", ("ферритин", "глюкоза"), (_FERRITIN, _GLUCOSE)),
}


class _Oracle:
    """Разметка слов — ровно `phrases`; выбор строк корзины — правило 5 её промпта: позицию с условием класса (в фразе или
    в «[добавлено: …]») не выбирает, остальные — все предложенные строки; выбор строки одной услуги — «той услуги с условием
    нет», если условие в вопросе. Прочие места — «недоступна» (как в гейте)."""

    def __init__(self, phrases: tuple[str, ...]) -> None:
        self.phrases = phrases
        self.asked: list[str] = []

    def _spans(self, prompt: str) -> dict[str, Any]:
        numbered = prompt.split(_SPANS_MARK, 1)[1].strip().splitlines()[0].split()
        words = [_normalise_catalog_text(item.partition(":")[2]) for item in numbered]
        groups = []
        for phrase in self.phrases:
            want = _normalise_catalog_text(phrase).split()
            for k in range(len(words) - len(want) + 1):
                if words[k : k + len(want)] == want:
                    groups.append([k + 1, k + len(want)])
                    break
        return {"phrases": groups}

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if _SPANS_MARK in prompt:
            self.asked.append("spans")
            return GenerateResponse(model="m", response=json.dumps(self._spans(prompt)), done=True)
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if _BASKET_MARK in seen:
            self.asked.append("basket")
            picks = [n for n, text in options if not _CONDITION.search(text.split(" → ", 1)[0])]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if _ONE_MARK in seen:
            self.asked.append("rows")
            question = next((line for line in prompt.splitlines() if line.startswith("Сообщение пациента:")), "")
            absent = bool(_CONDITION.search(question))
            picks = [] if absent else [n for n, _text in options[:1]]
            answer: dict[str, Any] = {"picks": picks} if system else {"match": picks, "search_terms": []}
            if absent:
                answer["reason"] = "qualifier_absent"
            return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


class _Down:
    async def generate(self, **_kwargs):
        raise ConnectionError("тест: LLM недоступна")


def _answer(reply: str, llm: Any, switch: str, world_dir, monkeypatch, session: str) -> str:
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D17, "MR_PRICE_BY_PARSE": switch}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([reply], session)
        attempts = list(world.net.attempts)
    assert attempts == []
    return str(turn["text"])


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_condition_inside_the_phrase_takes_the_price_away(case, switch, mini_prod_world, monkeypatch):  # noqa: F811
    reply, inside, _cut, priced, neighbour = CLASS[case]
    llm = _Oracle(inside)
    text = _answer(reply, llm, switch, mini_prod_world, monkeypatch, f"lcs-in-{case}-{switch}")
    assert priced not in text and neighbour in text, text
    unrecognized = re.search(r"Не распознал: ([^.]+)\.", text)
    assert unrecognized and _CONDITION.search(unrecognized.group(1)), text  # словами пациента, с условием


@pytest.mark.xfail(
    strict=True,
    reason="RISK-2026-10-26-LIST-CONDITION-CUT-BY-SPANS: разметка слов не заполняет `beside`, суд условия сильной позиции "
    "корзины не зовётся, у слабой — нет пометки «[добавлено: …]»; гарда П2 на уровне реплики у корзины нет",
)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_condition_cut_by_spans_never_gets_the_retail_price(case, switch, mini_prod_world, monkeypatch):  # noqa: F811
    """Инвариант класса: позиция списка с условием, которого нет в прайсе, не получает розничную цену без оговорки —
    при любой границе фразы, которую поставила LLM разметки."""

    reply, _inside, cut, priced, _neighbour = CLASS[case]
    text = _answer(reply, _Oracle(cut), switch, mini_prod_world, monkeypatch, f"lcs-cut-{case}-{switch}")
    assert priced not in text, text


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_neighbour_keeps_its_price_when_spans_cut_the_condition(case, switch, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: условие относится к одной позиции — соседняя без условия остаётся с ценой (фикс не глушит весь список)."""

    reply, _inside, cut, _priced, neighbour = CLASS[case]
    text = _answer(reply, _Oracle(cut), switch, mini_prod_world, monkeypatch, f"lcs-nb-{case}-{switch}")
    assert neighbour in text, text


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_without_llm_the_conditioned_position_has_no_retail_price(case, switch, mini_prod_world, monkeypatch):  # noqa: F811
    reply, _inside, _cut, priced, _neighbour = CLASS[case]
    text = _answer(reply, _Down(), switch, mini_prod_world, monkeypatch, f"lcs-down-{case}-{switch}")
    assert priced not in text, text


@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CONTROLS))
def test_lists_without_the_condition_keep_their_prices(case, switch, mini_prod_world, monkeypatch):  # noqa: F811
    reply, phrases, prices = CONTROLS[case]
    text = _answer(reply, _Oracle(phrases), switch, mini_prod_world, monkeypatch, f"lcs-ctl-{case}-{switch}")
    assert all(price in text for price in prices), text
