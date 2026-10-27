"""Гард П2 у сильной позиции корзины из фраз разбора (BUG-2026-10-16-P2-STRONG-PHRASE-BYPASS, корзина; `strong_phrase_guard`).

Под окнами разбора (`parse_fragment_windows` / `parse_complete_windows = on`) перечень кода предлагает «ферритин» внутри
фрагмента «ферритин по омс», LLM разбора выбирает его и «глюкозу» — сильная позиция корзины печатала «Ферритин — 530 руб.»
мимо гарда П2 (условие «по ОМС» клиника не оказывает — честное «Не распознал»). Без окон фрагмент целиком — слабая
позиция с гардом. Решение владельца 16.10: окна не включать, пока сильная позиция корзины не судит условие.

Фикс — LLM-first, как у пути одной услуги: разбор отдаёт слова фрагмента вне выбранной фразы (`ParsedPhrase.beside`);
слово, которого нет в прайсе, — повод спросить LLM корзины, не вердикт: позиция идёт в её вызов с кандидатами — своими
строками и словами пациента в варианте («ферритин» [добавлено: «по омс»]), правило 5 промпта корзины решает, условие
это или разговор. Сбой LLM — «Не распознал», не цена.

Инварианты класса (через обработчик эндпоинта, мини-прод, прод-набор 16.10 + окна, оракул LLM с правилом 5):
- `off` — как было: сильная позиция с ценой, корзина LLM о ней не спрашивается;
- `llm` + условие — позиция без цены, «Не распознал» словами пациента; соседняя позиция — с ценой;
- `llm` + сбой выбора строк — та же позиция без цены (не варианты с ценой);
- разговорное слово рядом с сильной фразой — цена, как под `off`;
- список без слов вне прайса — ответ под `llm` тот же, что под `off`;
- без окон («ферритин по омс» — слабая позиция) — ответ не меняется;
- два хода: короткий вопрос после такого ответа не получает цену позиции, которой пациент не видел.

Путь прода с 27.10 в гейте (`turn_spans_llm = on`, деплой 17; STEP-2026-10-15-DEFAULTS-B): фразы размечает LLM номерами слов, окон
нет. Подделка отвечает на разметку теми же фразами. Разметка, оторвавшая «по омс» от «ферритин» (ошибка модели против правила 3
промпта `turn_spans`), — известный дефект BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS: `beside` заполнял только генератор окон,
разметка — нет, суда условия нет, пациент видит «Ферритин — 530 руб.» (xfail strict ниже; чинит дизайн С3). Разметка по правилу 3
(«ферритин по омс» целиком) — слабая позиция с судом П2, цены нет. Откат `off` (ответ 15.10) удалён 27.10 — путь не прода.
"""

from __future__ import annotations

import json
import re

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services._common import _normalise_catalog_text
from tests.spans_oracle import is_spans, keep_llm_breaker_closed, spans_response
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _PHRASES_MARK, _ROWS_MARKS, _assert_follow_up_stays_on_what_was_shown

_PROD_1016 = {
}
_WINDOWS = {"MR_PARSE_FRAGMENT_WINDOWS": "on", "MR_PARSE_COMPLETE_WINDOWS": "on"}

_P2_BASKET = (
    "сколько стоит ферритин по омс, глюкоза",
    "сколько стоит глюкоза и ферритин по омс",
)
_CONVERSATIONAL = ("сколько стоит ферритин первый раз, глюкоза",)
_OUTSIDE_THE_CLASS = ("сколько стоит ферритин, глюкоза", "сколько стоит ферритин и глюкоза")
_FERRITIN = "Ферритин — 530"
_GLUCOSE = "Глюкоза — 190"
_BUG_1 = "BUG-2026-10-26-LIST-CONDITION-CUT-BY-SPANS: условие у сильной позиции списка под разметкой turn_spans не судится — чинит С3"
# Разметка «ферритин по омс» по правилу 3 промпта `turn_spans` — условие при названии (слабая позиция, суд П2).
_WHOLE = ("ферритин по омс", "глюкоза")


class _RuleFive:
    """LLM с правилом 5 промпта корзины: разбор выбирает фразы `phrases` (сильные, без условия — как прод 16.10);
    выбор строк отмечает все варианты, кроме позиций, к которым пациент добавил условие «… омс»; `down` — выбор
    строк недоступен (разбор ответил)."""

    def __init__(self, phrases: tuple[str, ...] = ("ферритин", "глюкоза"), *, down: bool = False) -> None:
        self.phrases = {_normalise_catalog_text(text) for text in phrases}
        self.down = down
        self.asked: list[str] = []
        self.rows_prompts: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **_kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        if is_spans(prompt):  # разметка слов `turn_spans` (прод с деплоя 17) — те же фразы; раньше метки старого промпта
            self.asked.append("phrases")
            return spans_response(prompt, sorted(self.phrases, key=lambda text: -len(text.split())))
        if _PHRASES_MARK in seen:
            self.asked.append("phrases")
            picks = [n for n, text in options if _normalise_catalog_text(text.strip("«»")) in self.phrases]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}), done=True)
        if any(mark in seen for mark in _ROWS_MARKS) and not self.down:
            self.asked.append("rows")
            self.rows_prompts.append(prompt)
            picks = [n for n, text in options if not re.search(r"\[добавлено: «[^»]*омс", text.split(" → ", 1)[0].lower())]
            return GenerateResponse(model="m", response=json.dumps({"picks": picks}, ensure_ascii=False), done=True)
        raise ConnectionError("тест: LLM недоступна для этого места")


def _run(turns, world, monkeypatch, *, guard: str, llm, windows: bool = True, session: str):
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    for key, value in {**_PROD_1016, **(_WINDOWS if windows else {})}.items():
        monkeypatch.setattr(cfg, key, value, raising=False)
    monkeypatch.setattr(cfg, "MR_STRONG_PHRASE_GUARD", guard, raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", llm)
    keep_llm_breaker_closed(monkeypatch)
    with ev_world.FrozenWorld(world, "day") as frozen:
        record = run_dialog(list(turns), session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return record, rows


def _sid(*parts: str) -> str:
    return "p2spb-" + "-".join(re.sub(r"\W+", "_", part)[-16:] for part in parts)


def test_precondition_windows_offer_the_phrase_without_the_condition(mini_prod_world):  # noqa: F811
    """Предусловие: под окнами перечень предлагает «ферритин» внутри «ферритин по омс», без окон — нет; выбранная
    фраза списка несёт слова фрагмента вне себя (`beside`)."""

    from agent_logic_2.nayka_api import api_price

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _prices_helpers as H
    from messengers_router.services import _turn_phrases as P
    from messengers_router.turn_parse import ParseStatus

    with ev_world.FrozenWorld(mini_prod_world, "day"), pytest.MonkeyPatch.context() as mp:
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        for reply in _P2_BASKET:
            for key, value in _WINDOWS.items():
                mp.setattr(cfg, key, "off", raising=False)
            assert "ферритин" not in [o.raw for o in P.phrase_candidates(reply, facet=None, catalog=rows).options], reply
            for key, value in _WINDOWS.items():
                mp.setattr(cfg, key, value, raising=False)
            cands = P.phrase_candidates(reply, facet=None, catalog=rows)
            picked = [o for o in cands.options if o.raw in ("ферритин", "глюкоза")]
            parse = P.result(cands, ParseStatus.PICKED, "llm", [P.parsed(o, P.phrase_rows(o.raw, rows)) for o in picked])
            beside = {item.text: item.beside for item in parse.items}
            assert beside == {"ферритин": "омс", "глюкоза": ""}, beside  # «по» — обвязка (`_FILLER`)


@pytest.mark.xfail(strict=True, reason=_BUG_1)
@pytest.mark.parametrize("reply", _P2_BASKET)
def test_the_condition_takes_the_price_away(reply, mini_prod_world, monkeypatch):  # noqa: F811
    llm = _RuleFive()
    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=llm, session=_sid("llm", reply))
    assert _FERRITIN not in turn["text"] and _GLUCOSE in turn["text"], turn["text"]
    assert "Не распознал: ферритин омс" in turn["text"], turn["text"]  # словами пациента, не «ферритин»
    assert "[добавлено: «омс»]" in llm.rows_prompts[0]
    assert row["strong_phrase_guard"] == "basket_not_found"


@pytest.mark.xfail(strict=True, reason=_BUG_1)
@pytest.mark.parametrize("reply", _P2_BASKET)
def test_a_failed_row_choice_gives_no_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Сбой LLM выбора строк — позиция на суде без цены и без вариантов с ценой (П2); сильная соседняя — с ценой."""

    [turn], [row] = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=_RuleFive(down=True), session=_sid("down", reply))
    assert _FERRITIN not in turn["text"] and "Ферритин" not in turn["text"] and _GLUCOSE in turn["text"], turn["text"]
    assert row["strong_phrase_guard"] == "basket_fallback"


@pytest.mark.parametrize("reply", _CONVERSATIONAL)
def test_a_conversational_word_keeps_the_price(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Встречный инвариант: слово разговора рядом с сильной фразой («первый раз») гард по буквам считал бы
    уточнением; цена обеих позиций остаётся, «Не распознал» нет.

    Под разметкой (путь прода) суд не спрашивается вовсе: `beside` разметка не заполняет (корень BUG-2026-10-26-LIST-CONDITION-
    CUT-BY-SPANS), поэтому прежняя проверка механизма окон («выбор строк спрошен», `strong_phrase_guard = basket_rows`) и
    сравнение с откатом `off` сняты 27.10; идёт ли разговорное слово на суд — решит С3 (карточка ветки дефекта: суд разговорных
    слов терял верные цены), ответ пациенту от этого не должен меняться."""

    [after], _rows = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=_RuleFive(), session=_sid("cllm", reply))
    assert _FERRITIN in after["text"] and _GLUCOSE in after["text"] and "Не распознал" not in after["text"], after["text"]


@pytest.mark.parametrize("reply", _OUTSIDE_THE_CLASS)
def test_lists_without_outside_words_do_not_change(reply, mini_prod_world, monkeypatch):  # noqa: F811
    [off], _rows = _run([reply], mini_prod_world, monkeypatch, guard="off", llm=_RuleFive(), session=_sid("ooff", reply))
    llm = _RuleFive()
    [on], [row] = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=llm, session=_sid("oon", reply))
    assert on["text"] == off["text"] and _FERRITIN in on["text"], on["text"]
    assert "rows" not in llm.asked and "strong_phrase_guard" not in row


@pytest.mark.parametrize("reply", _P2_BASKET)
def test_without_windows_nothing_changes(reply, mini_prod_world, monkeypatch):  # noqa: F811
    """Без окон фраза с условием — слабая позиция («ферритин по омс») с гардом П2: выключатель ответа не меняет. Под разметкой
    (путь прода) — то же при разметке по правилу 3 промпта `turn_spans`: условие не оторвано от названия."""

    [off], _rows = _run([reply], mini_prod_world, monkeypatch, guard="off", llm=_RuleFive(_WHOLE), windows=False, session=_sid("woff", reply))
    [on], [row] = _run([reply], mini_prod_world, monkeypatch, guard="llm", llm=_RuleFive(_WHOLE), windows=False, session=_sid("won", reply))
    assert on["text"] == off["text"] and _FERRITIN not in on["text"] and _GLUCOSE in on["text"], on["text"]
    # Слабую позицию с сильным названием внутри судит LLM корзины (суд слабой фразы — всегда с 22.10, выключатель
    # `basket_weak_phrase_judge` снят), поэтому строка хода может нести вердикт суда; ответ — тот же, цены по ОМС нет.
    _ = row


@pytest.mark.parametrize(
    "phrases",
    [
        pytest.param(("ферритин", "глюкоза"), marks=pytest.mark.xfail(strict=True, reason=_BUG_1), id="torn"),
        pytest.param(_WHOLE, id="whole"),
    ],
)
@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_stays_on_what_was_shown(follow_up, phrases, mini_prod_world, monkeypatch):  # noqa: F811
    """Разметка оторвала условие (`torn`) — дефект №1: первый ход уже с ценой ферритина; разметка по правилу 3 (`whole`) —
    первый ход без неё, короткий вопрос следом не получает цену, которой пациент не видел."""

    (first, second), _rows = _run(
        [_P2_BASKET[0], follow_up], mini_prod_world, monkeypatch, guard="llm", llm=_RuleFive(phrases), session=_sid("fu", follow_up, *phrases)
    )
    assert _FERRITIN not in first["text"], first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
