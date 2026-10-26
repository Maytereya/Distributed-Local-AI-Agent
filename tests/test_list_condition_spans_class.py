"""Условие у позиции списка на пути прода: разметка слов (`turn_spans_llm = on`) оторвала «по ОМС» от названия — позиция
получала розничную цену (BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS, класс П2; выключатель `list_condition_beside`).

Суд условия у позиции корзины (`phrase_price_items`, BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS) читает `ParsedPhrase.beside` —
слова фрагмента вне выбранной фразы. Поле заполнял только прежний генератор окон (`_turn_phrases.result` → `_with_beside`);
разметка слов (`_turn_spans._result`, прод с деплоя 17) — нет, а гарда П2 на уровне реплики у корзины нет
(`prices._multi_price_payload` идёт раньше всех гардов одной услуги). Разметка по правилу 3 своего промпта держит условие внутри
фразы — тогда позиция слабая и её судят (честное «Не распознал»); оторвала условие («ферритин» из «ферритин по ОМС», живая LLM
у одной услуги — 59 из 82 свипа П2, карточка Ш7) — сильная позиция печатала цену по данным, LLM строк о ней не спрашивали
вовсе; слабая шла к LLM без пометки «[добавлено: …]», правило 5 промпта корзины не срабатывало.

Фикс под `list_condition_beside = on`: разметка отдаёт фразам списка слова их отрезка вне фраз (`_turn_spans._with_beside`),
слабая позиция с условием вне прайса несёт его в вариант LLM корзины, сбой LLM корзины — «Не распознал», не цена.

Инварианты класса (через обработчик эндпоинта, мини-прод, выключатели прода деплоя 17; `price_by_parse` off — прод
деплоя 17, on — деплой 18; оракул LLM: разметка — фразы случая, выбор строк корзины — правило 5 промпта):
- условие оторвано разметкой — позиция без розничной цены: on — держится, off — известный дефект (xfail strict);
- то же при сбое LLM выбора строк (разметка ответила): on — держится, off — xfail strict;
- встречные, в обоих положениях выключателя: условие внутри фразы — без цены и «Не распознал» с условием; соседняя позиция
  при оторванном условии — с ценой (фикс не глушит весь список); LLM недоступна — без цены позиции; списки без условия и со
  словом разговора («первый раз») — с ценами.
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
from tests.test_basket_calc_turns import _load as basket_golden
from tests.test_basket_calc_turns import _turns as basket_turns
from tests.test_basket_calc_turns import basket_world  # noqa: F401 — фикстура
from tests.test_price_stage2_turns import PROD_D17
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура

SWITCH = ("off", "on")  # `price_by_parse`: off — прод деплоя 17, on — набор деплоя 18
BESIDE = ("off", "on")  # `list_condition_beside`
_DEFECT = "BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS: под `list_condition_beside = off` разметка слов не заполняет `beside`"
BESIDE_FIX = (pytest.param("off", marks=pytest.mark.xfail(strict=True, reason=_DEFECT)), "on")
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
    "tsh_omc": ("ферритин, ТТГ по омс", ("ферритин", "ТТГ по омс"), ("ферритин", "ТТГ"), _TSH, _FERRITIN),  # «ТТГ» — слабая фраза
}
# без условия класса: что разметка ни отметит, цены остаются (фикс не должен глушить разговор и чистые списки)
CONTROLS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "plain_and": ("сколько стоит ферритин и глюкоза", ("ферритин", "глюкоза"), (_FERRITIN, _GLUCOSE)),
    "plain_comma": ("ферритин, ТТГ", ("ферритин", "ТТГ"), (_FERRITIN, _TSH)),
    "conversational": ("сколько стоит ферритин первый раз, глюкоза", ("ферритин", "глюкоза"), (_FERRITIN, _GLUCOSE)),
}


class _Oracle:
    """Разметка слов — ровно `phrases`; выбор строк корзины — правило 5 её промпта: позицию с условием класса (в фразе или
    в «[добавлено: …]») не выбирает, остальные — все предложенные строки (`rows_down` — выбор строк недоступен); выбор строки
    одной услуги — «той услуги с условием нет», если условие в вопросе. Прочие места — «недоступна» (как в гейте)."""

    def __init__(self, phrases: tuple[str, ...], *, rows_down: bool = False) -> None:
        self.phrases = phrases
        self.rows_down = rows_down
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
        if self.rows_down:
            raise ConnectionError("тест: выбор строк недоступен")
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


def _answer(reply: str, llm: Any, switch: str, beside: str, world_dir, monkeypatch, session: str) -> str:
    from messengers_router.runtime_config import config as cfg

    for key, value in {**PROD_D17, "MR_PRICE_BY_PARSE": switch, "MR_LIST_CONDITION_BESIDE": beside}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    with ev_world.FrozenWorld(world_dir, "day") as world:
        [turn] = run_dialog([reply], session)
        attempts = list(world.net.attempts)
    assert attempts == []
    return str(turn["text"])


@pytest.mark.parametrize("beside", BESIDE_FIX)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_condition_cut_by_spans_never_gets_the_retail_price(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    """Инвариант класса: позиция списка с условием, которого нет в прайсе, не получает розничную цену без оговорки —
    при любой границе фразы, которую поставила LLM разметки."""

    reply, _inside, cut, priced, _neighbour = CLASS[case]
    text = _answer(reply, _Oracle(cut), switch, beside, mini_prod_world, monkeypatch, f"lcs-cut-{case}-{switch}-{beside}")
    assert priced not in text, text
    unrecognized = re.search(r"Не распознал: ([^.]+)\.", text)
    assert unrecognized and _CONDITION.search(unrecognized.group(1)), text  # словами пациента, с условием


@pytest.mark.parametrize("beside", BESIDE_FIX)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_condition_cut_and_failed_row_choice_give_no_price(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    """Разметка ответила, выбор строк LLM — сбой: позиция с условием без цены (не варианты с ценой — П2)."""

    reply, _inside, cut, priced, _neighbour = CLASS[case]
    llm = _Oracle(cut, rows_down=True)
    text = _answer(reply, llm, switch, beside, mini_prod_world, monkeypatch, f"lcs-rd-{case}-{switch}-{beside}")
    assert priced not in text, text


@pytest.mark.parametrize("beside", BESIDE)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_condition_inside_the_phrase_takes_the_price_away(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    reply, inside, _cut, priced, neighbour = CLASS[case]
    text = _answer(reply, _Oracle(inside), switch, beside, mini_prod_world, monkeypatch, f"lcs-in-{case}-{switch}-{beside}")
    assert priced not in text and neighbour in text, text
    unrecognized = re.search(r"Не распознал: ([^.]+)\.", text)
    assert unrecognized and _CONDITION.search(unrecognized.group(1)), text  # словами пациента, с условием


@pytest.mark.parametrize("beside", BESIDE)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_neighbour_keeps_its_price_when_spans_cut_the_condition(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный: условие относится к одной позиции — соседняя без условия остаётся с ценой (фикс не глушит весь список)."""

    reply, _inside, cut, _priced, neighbour = CLASS[case]
    text = _answer(reply, _Oracle(cut), switch, beside, mini_prod_world, monkeypatch, f"lcs-nb-{case}-{switch}-{beside}")
    assert neighbour in text, text


@pytest.mark.parametrize("beside", BESIDE)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CLASS))
def test_without_llm_the_conditioned_position_has_no_retail_price(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    reply, _inside, _cut, priced, _neighbour = CLASS[case]
    text = _answer(reply, _Down(), switch, beside, mini_prod_world, monkeypatch, f"lcs-down-{case}-{switch}-{beside}")
    assert priced not in text, text


@pytest.mark.parametrize("beside", BESIDE)
@pytest.mark.parametrize("switch", SWITCH)
@pytest.mark.parametrize("case", sorted(CONTROLS))
def test_lists_without_the_condition_keep_their_prices(case, switch, beside, mini_prod_world, monkeypatch):  # noqa: F811
    reply, phrases, prices = CONTROLS[case]
    text = _answer(reply, _Oracle(phrases), switch, beside, mini_prod_world, monkeypatch, f"lcs-ctl-{case}-{switch}-{beside}")
    assert all(price in text for price in prices), text


@pytest.mark.parametrize("reply", [CLASS["omc_comma"][0], CONTROLS["plain_and"][0], CONTROLS["conversational"][0]])
def test_off_parse_items_carry_no_beside(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Откат: под off разбор разметки слов — как было, без `beside` у фраз списка; под on — слова отрезка вне фраз."""

    import asyncio

    from agent_logic_2.nayka_api import api_price

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _turn_spans as S

    phrases = ("ферритин", "глюкоза")
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", _Oracle(phrases))
    monkeypatch.setattr(llm_runtime, "_BREAKER", llm_runtime._Breaker())
    got = {}
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        for beside in BESIDE:
            monkeypatch.setattr(cfg, "MR_LIST_CONDITION_BESIDE", beside, raising=False)
            parse = asyncio.run(S.parse_turn_spans(reply, facet=None, catalog=rows))
            got[beside] = {item.text: item.beside for item in parse.items}
    assert set(got["off"].values()) == {""}, got
    outside = {word for text in got["on"].values() for word in text.split()}
    assert outside == set(reply.replace(",", " ").split()) - {"ферритин", "глюкоза"}, got


# --- эталон корзины «ферритин или железо в сыворотке — какой лучше сдать?» (`test_basket_calc_turns`, `same_service_asked`)


@pytest.mark.parametrize("calc", SWITCH)
def test_basket_golden_with_a_conversational_tail_is_unchanged_under_off(calc, basket_world, monkeypatch):  # noqa: F811
    """Откат: под `list_condition_beside = off` ответ — эталон `basket_calc_turns.json` (оба положения `basket_calc`)."""

    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_LIST_CONDITION_BESIDE", "off", raising=False)
    record = json.loads(json.dumps(basket_turns("same_service_asked", calc, basket_world, monkeypatch), ensure_ascii=False))
    assert record == basket_golden()[f"same_service_asked-{calc}"]


@pytest.mark.parametrize("rows_answer", (True, False))
def test_basket_conversational_tail_goes_to_the_judge_under_on(rows_answer, basket_world, monkeypatch):  # noqa: F811
    """Под on хвост «какой лучше сдать» (отрезок после тире, без фраз) — у «железо в сыворотке» и идёт на суд LLM корзины:
    ответила (разговор, правило 5) — ответ эталона; выбор строк недоступен — «Не распознал» (П2: без суда цены нет)."""

    from dataclasses import replace

    from messengers_router.runtime_config import config as cfg
    from tests import test_basket_calc_turns as calc_turns

    monkeypatch.setattr(cfg, "MR_LIST_CONDITION_BESIDE", "on", raising=False)
    case = calc_turns.CASES["same_service_asked"]
    if rows_answer:
        monkeypatch.setitem(calc_turns.CASES, "same_service_asked", replace(case, rows=("Железо в сыворотке",)))
    [turn] = basket_turns("same_service_asked", "off", basket_world, monkeypatch)["turns"]
    golden = basket_golden()["same_service_asked-off"]["turns"][0]["text"]
    if rows_answer:
        assert turn["text"] == golden, turn["text"]
    else:
        assert "Железо в сыворотке — 240" not in turn["text"] and "Не распознал: железо в сыворотке" in turn["text"], turn["text"]
        assert _FERRITIN in turn["text"], turn["text"]
