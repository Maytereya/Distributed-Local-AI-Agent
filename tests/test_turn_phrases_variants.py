"""Генератор фраз: дополнительные варианты для выбора LLM (L-15b, шаг 6; разбор 95 ответов, п. 6; T4).

Правила разреза теряли позиции, и LLM выбора фраз их не видела: короткие обозначения («лг», «щф») —
сплиттер правил выбрасывает фрагмент, которого нет в словарях кода (g0281, g0382, g0481); перечисление в
скобках («TORCH (токсоплазмоз, краснуха, ЦМВ, герпес)», «(RW, ВИЧ, HBsAg, HCV)», g0647, g0339); хвост вопроса
у последней позиции («витамин D — какая цена анализов», g0348); родовая голова («анализов на ттг», g0040).
Генератор добавляет варианты (д) — выбирает по-прежнему LLM. Инварианты:
- нужная фраза среди вариантов (класс выше), каждая — срез реплики без обвязки
  (`test_turn_phrases.py::test_every_phrase_is_a_slice_of_the_reply`, ≥500 реплик);
- варианты не режут реплику: разрезы (б)/(в)/(г), обход без LLM и политика сбоя прежние;
- одна фраза без разделителей остаётся `single` — без нового вызова LLM;
- вариант не называет несколько позиций сразу («ТТГ и TORCH») и не состоит из одних родовых слов;
- через эндпоинт с оракулом «лг и фсг» и «сколько стоит лг и ггтп» — две позиции со своими ценами; второй
  короткий вопрос не берёт услугу, которой пациент не видел.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime, turn_understanding
from messengers_router.services import _turn_phrases as TP
from messengers_router.services._common import _normalise_catalog_text
from tests.test_price_one_or_many_class import Case, _configure, _OracleLLM, _violations
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура модуля

_LH = "ЛГ (LH) Лютеинизирующий гормон"
_FSH = "ФСГ (FSH) Фолликулостимулирующий гормон"
_PRICED = re.compile(r"^\s*\d+[.)]\s*(?:Розничная цена:\s*)?(?P<name>.+) — \d[\d\s]* руб", re.M)


@pytest.fixture(scope="module")
def catalog(mini_prod_world) -> list[dict[str, Any]]:  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day"):
        from messengers_router.services import _prices_helpers as H

        return H._catalog_rows(None)


@pytest.fixture
def world(mini_prod_world):  # noqa: F811
    with ev_world.FrozenWorld(mini_prod_world, "day") as frozen:
        yield frozen


def _cands(reply: str, catalog):
    return TP.phrase_candidates(reply, facet=turn_understanding.price_turn_facet(reply)[0], catalog=catalog)


def _texts(cands) -> list[str]:
    return [_normalise_catalog_text(p.text) for p in cands.options]


@pytest.mark.parametrize(
    ("reply", "wanted"),
    [
        ("лг, фсг, пролактин", ("лг", "фсг", "пролактин")),  # короткое обозначение вне словарей кода
        ("сколько стоит лг и фсг", ("лг", "фсг")),
        ("Госпитальный скрининг (RW, ВИЧ, HBsAg, HCV)", ("rw", "вич", "hbsag", "hcv")),  # перечисление в скобках
        ("Стоимость: ФСГ, ЛГ, пролактин и TORCH (токсоплазмоз, краснуха, ЦМВ, герпес)", ("лг", "torch", "краснуха", "цмв")),
        ("Ферритин, железо, ОАК, ТТГ, Т4 свободный, витамин D — какая цена анализов?", ("витамин d",)),  # хвост вопроса
        ("Нужно узнать стоимость анализов на ттг ; т3 ; т4", ("ттг", "т3", "т4")),  # родовая голова
    ],
)
def test_lost_positions_are_offered_to_the_model(reply, wanted, catalog, world):
    texts = _texts(_cands(reply, catalog))
    assert [w for w in wanted if w not in texts] == [], texts


@pytest.mark.parametrize(
    "reply",
    [
        "лг, фсг, пролактин",
        "Стоимость: ФСГ, ЛГ, пролактин и TORCH (токсоплазмоз, краснуха, ЦМВ, герпес)",
        "Ферритин, железо, ОАК, ТТГ, Т4 свободный, витамин D — какая цена анализов?",
        "сколько стоит ферритин, ттг и витамин д",
    ],
)
def test_variants_do_not_cut_the_reply(reply, catalog, world):
    cands = _cands(reply, catalog)
    assert all(p.source != TP.SOURCE_VARIANT for cut in cands.partitions.values() for p in cut)
    assert TP.fallback_split(cands, catalog) == tuple(
        TP.parsed(p, TP.phrase_rows(p.raw, catalog)) for p in cands.partitions.get(TP.SOURCE_SPLIT, ())
    ) or not TP.fallback_split(cands, catalog)


@pytest.mark.parametrize("reply", ["сколько стоит лг", "сколько стоит анализ на ттг", "лг", "анализ на ферритин"])
def test_one_phrase_without_delimiters_stays_single(reply, catalog, world):
    cands = _cands(reply, catalog)
    decided = TP.decide_without_llm(cands, catalog)
    assert decided is not None and decided.status in ("single", "strong_whole"), (decided, _texts(cands))


def test_a_variant_names_one_position_and_not_only_generic_words(catalog, world):
    for reply in (
        "Стоимость: ФСГ, ЛГ, пролактин, ТТГ и TORCH (токсоплазмоз, краснуха, ЦМВ, герпес)",
        "Ферритин, железо, ОАК, ТТГ, Т4 свободный, витамин D — какая цена анализов?",
        "Подскажите стоимость анализов: ферритин, ттг",
    ):
        for p in _cands(reply, catalog).options:
            if p.source != TP.SOURCE_VARIANT:
                continue
            assert not TP._names_several(p.raw), (reply, p.text)
            assert not all(t.key in TP._FILLER for t in TP._tokens(p.raw)), (reply, p.text)


def _priced(text: str) -> set[str]:
    return {" ".join(m.group("name").split()) for m in _PRICED.finditer(text)}


@pytest.mark.parametrize(
    ("reply", "positions"),
    [
        ("сколько стоит лг и фсг", {"лг": (_LH,), "фсг": (_FSH,)}),
        # T4: «лг» сплиттер правил выбрасывает — без варианта (д) LLM его не видит. («сколько стоят …» без
        # живого классификатора правила ценой не считают; «стоят» — не обвязка генератора: фраза «стоят лг».)
        # «ггтп» в мини-проде кодом не находится («ГГТ гамма-глютамилтрансфераза», синонимов МИС в гейте нет) —
        # позиция без строки: «Не распознал», не чужая цена.
        ("сколько стоит лг и ггтп", {"лг": (_LH,), "ггтп": ()}),
    ],
)
def test_short_designations_get_their_own_prices_through_the_endpoint(reply, positions, mini_prod_world, monkeypatch):  # noqa: F811
    case = Case(reply, "list", positions)
    _configure(monkeypatch, "deploy_parse")
    oracle = _OracleLLM(case, tuple(positions))
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(mini_prod_world, "day") as world:
        first, second = run_dialog([reply, "а сколько?"], f"variants-{reply}")
        assert list(world.net.attempts) == []
    assert "phrases" in oracle.calls
    violations = _violations(case, first["text"], "oracle", oracle.picked)
    assert not violations, "\n".join([*violations, "--- ответ:", first["text"]])
    assert {_LH} <= _priced(first["text"])
    assert sorted(_priced(second["text"]) - _priced(first["text"])) == []  # контекст = показанное
