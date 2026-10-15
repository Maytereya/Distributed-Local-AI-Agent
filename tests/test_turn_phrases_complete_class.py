"""Окно-кандидат разбора, оборванное перед словом названия (g0724, `parse_complete_windows`, 15.10).

«Подскажите, нужно ли дополнительно платить за взятие крови из вены»: окна сегментации (в) и сильные окна (г) — не
длиннее трёх слов, предлог «из» занимает место, и кандидат — «взятие крови из»; «вены» правила приземляют на ту же
услугу и выбрасывают повтором. W — с обвязкой «дополнительно платить за», слабая. LLM разбора не видела ни одной фразы
с названием целиком и отвечала `none_fit` → «Уточните» вместо 195 руб.

В мини-проде g0724 не воспроизводится (сегментации не на что опереться в «дополнительно платить за», а без живого
классификатора реплика не ценовая); тот же механизм — «Сколько стоит ОАК взятие крови из вены»: сегментация режет
«ОАК» и «взятие крови из», «вены» выбрасывается повтором, сильного окна (г) нет — нет и промежутков (г), которые
отдали бы название целиком (как в «ферритин взятие крови из пальца»).

Инварианты класса «кандидат обрезан, полный — сильный» (мини-прод):
- `off` — кандидаты как были (эталоны `price_turns*.json` и `test_turn_phrases*.py` — без изменений);
- `on` — слабое окно, которое соседними словами реплики (не через знак препинания) дополняется до сильного по данным
  МИС, получает вариант с полным названием; разрезы, обход без LLM и политика сбоя прежние;
- `on` ничего не меняет, где окна сильные или дополнять нечем;
- через эндпоинт с оракулом LLM — цена «Взятие крови из вены»; два хода: короткий вопрос после ответа не получает цену
  того, чего пациент не видел.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router.services import _turn_phrases as TP
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_turns_golden import mini_prod_world  # noqa: F401 — фикстура
from tests.test_turn_parse_t3 import _FOLLOW_UPS, _PHRASES_MARK, _Oracle, _assert_follow_up_stays_on_what_was_shown, _dialog

_REPLY = "Сколько стоит ОАК взятие крови из вены"
_FULL = "взятие крови из вены"
_VEIN = "Взятие крови из вены — 195"


def _mode(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PARSE_COMPLETE_WINDOWS", mode, raising=False)


def _cands(world_dir, monkeypatch, mode: str, reply: str) -> tuple[TP.Candidates, list[dict[str, Any]]]:
    from agent_logic_2.nayka_api import api_price

    from messengers_router.services import _prices_helpers as H

    _mode(monkeypatch, mode)
    with ev_world.FrozenWorld(world_dir, "day"):
        rows = [row for row in api_price.load_price_by_region(H.SAMARA_PRICE_REGION_ID) if isinstance(row, dict)]
        return TP.phrase_candidates(reply, facet="price", catalog=rows), rows


def _texts(cands: TP.Candidates) -> list[str]:
    return [_normalise_catalog_text(p.text) for p in cands.options]


def test_off_keeps_the_truncated_window(mini_prod_world, monkeypatch):  # noqa: F811
    """Как было: кандидат сегментации — обрывок «взятие крови из», полного названия среди кандидатов нет."""

    cands, _rows = _cands(mini_prod_world, monkeypatch, "off", _REPLY)
    texts = _texts(cands)
    assert "взятие крови из" in texts and _FULL not in texts, texts


@pytest.mark.parametrize(
    ("reply", "full"),
    [
        (_REPLY, _FULL),
        ("ОАК ОАМ взятие крови из пальца", "взятие крови из пальца"),
    ],
)
def test_on_offers_the_window_completed_to_a_strong_name(reply, full, mini_prod_world, monkeypatch):  # noqa: F811
    on, rows = _cands(mini_prod_world, monkeypatch, "on", reply)
    off, _rows = _cands(mini_prod_world, monkeypatch, "off", reply)
    completed = [p for p in on.options if _normalise_catalog_text(p.text) == full]
    assert completed and completed[0].source == TP.SOURCE_VARIANT, _texts(on)
    assert completed[0].raw == reply[completed[0].start : completed[0].end]  # фраза ⊂ реплика
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        assert TP.phrase_rows(completed[0].raw, rows)  # полное — сильное по данным МИС
    # Вариант не режет реплику: разрезы прежние, прежние кандидаты на месте.
    assert on.partitions == off.partitions
    assert set(_texts(off)) <= set(_texts(on))
    assert set(_texts(on)) - set(_texts(off)) == {full}


@pytest.mark.parametrize(
    "reply",
    [
        "ферритин, ттг, витамин д",  # сильные фрагменты — дополнять нечего
        "сколько стоит ферритин, ттг и витамин д",
        "Стоимость анализов: ферритин, кальпротектин, лактозная непереносимость",
        "ОАК ОАМ общий белок глюкоза",  # сплошной список: окна сильные
        "сколько стоит взятие крови из вены",  # одна сильная фраза — `strong_whole`, выбора нет
        "сколько стоит ферритин взятие крови из пальца",  # название целиком — промежуток между сильными окнами (г)
        "сколько стоит ОАК, взятие крови из вены",  # разделитель: фрагмент (б) — название целиком
    ],
)
def test_on_changes_nothing_where_windows_are_strong_or_nothing_completes(reply, mini_prod_world, monkeypatch):  # noqa: F811
    on, _rows = _cands(mini_prod_world, monkeypatch, "on", reply)
    off, _rows = _cands(mini_prod_world, monkeypatch, "off", reply)
    assert _texts(on) == _texts(off)


def test_on_keeps_the_decision_without_llm(mini_prod_world, monkeypatch):  # noqa: F811
    """Вариант разрезом не служит: обход без LLM и разрез политики сбоя — те же, что под off."""

    on, rows = _cands(mini_prod_world, monkeypatch, "on", _REPLY)
    off, _rows = _cands(mini_prod_world, monkeypatch, "off", _REPLY)
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        assert TP.decide_without_llm(on, rows) == TP.decide_without_llm(off, rows) is None
        assert TP.fallback_split(on, rows) == TP.fallback_split(off, rows)


class _Seeing(_Oracle):
    """Оракул, который запоминает фразы, предложенные LLM разбора."""

    def __init__(self) -> None:
        super().__init__(("ОАК", _FULL), ("Общий анализ крови (Le", "Взятие крови из вены"))
        self.offered: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        if _PHRASES_MARK in (f"{system}\n\n{prompt}" if system else prompt):
            lines = (line.partition(". ") for line in prompt.splitlines())
            self.offered += [_normalise_catalog_text(rest.strip("«»")) for head, dot, rest in lines if dot and head.isdigit()]
        return await super().generate(prompt=prompt, system=system, **kwargs)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_the_model_sees_the_full_name_only_when_on(mode, mini_prod_world, monkeypatch):  # noqa: F811
    """Через эндпоинт: под off LLM разбора видит обрывок «взятие крови из» и не видит названия целиком; под on —
    видит, выбирает, и пациент получает цену «Взятие крови из вены». (Под off в мини-проде цену находят правила пути
    одной услуги; в проде g0724 LLM отвечала `none_fit`, и ответ был «Уточните».)"""

    _mode(monkeypatch, mode)
    oracle = _Seeing()
    [turn], _rows = _dialog([_REPLY], mini_prod_world, monkeypatch, f"complete-{mode}", oracle)
    assert "взятие крови из" in oracle.offered, oracle.offered
    assert (_FULL in oracle.offered) is (mode == "on"), oracle.offered
    if mode == "on":
        assert _VEIN in turn["text"], turn["text"]


@pytest.mark.parametrize("follow_up", _FOLLOW_UPS)
def test_follow_up_after_the_completed_window_stays_on_what_was_shown(follow_up, mini_prod_world, monkeypatch):  # noqa: F811
    _mode(monkeypatch, "on")
    oracle = _Seeing()
    first, second = _dialog([_REPLY, follow_up], mini_prod_world, monkeypatch, f"complete-two-{follow_up}", oracle)[0]
    assert _VEIN in first["text"], first["text"]
    _assert_follow_up_stays_on_what_was_shown(first, second)
