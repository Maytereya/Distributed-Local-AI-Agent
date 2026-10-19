"""Данные МИС помогают LLM судить строку прайса (BUG-2026-10-19-PRICE-ROUND-TWO-POOL, остаток g0542).

«Сколько стоит мазок на микрофлору с поверхности задней стенки глотки» → «Не нашёл»: верная строка «Посев на микрофлору
(…)» (1200 и 1120 руб.) была в списке выбора, но LLM видела только «название — цена». В названии материала нет, и по
правилу 2 промпта («учитывай материал») LLM отвергала строку в обоих кругах. А МИС знает: в `biomatNames` строки есть
«соскоб с задней стенки глотки», в `serviceSynonyms` — «мазок на флору».

Фикс — под `[MESSENGER_ROUTER] price_option_mis_notes = on` (умолчание off — как было): у позиции в выборе строки — что
МИС знает о ней и назвал пациент: материал и синоним клиники, только совпавшие со словами реплики, не длиннее 120
знаков. Ни нового правила, ни словаря: решает по-прежнему LLM, данные — из справочника клиники.

Инварианты класса (через обработчик эндпоинта, мини-прод, словарь МИС подставлен в тесте, оракул LLM настоящего типа:
судит строку по её тексту, материала не видит — отвергает):
- строка принимает материал, который назвал пациент, — LLM видит его в тексте позиции и может выбрать строку: цена этой
  строки, а не «Не нашёл» (на старом коде — «Не нашёл»);
- в пометке — только совпавшее со словами реплики: «соскоб с зева» на вопрос о глотке не показывается; позиция без
  данных МИС или без совпадения — текст как раньше; пометка не длиннее `_MIS_NOTE_MAX`;
- список кандидатов от пометки не меняется (те же строки в том же порядке): чужих цен она не добавляет;
- выключатель off — текст всех позиций байт в байт прежний, ответ прежний (без цены).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest
from ollama import GenerateResponse

from evalbench import world as ev_world
from tests.test_price_turns_golden import FIXTURE
from tests.test_qualifier_round_two_class import _priced, _run
from tests.test_turn_parse_t3 import _ROWS_MARKS, _Oracle

_THROAT = "соскоб с задней стенки глотки"
_CERVIX = "соскоб с шейки матки"
_POSEV = "Посев на микрофлору (идентификация агента и чувствительность к а\\б)"

# Выдержка справочника МИС для мира теста. У «Посев на микрофлору» материалы «соскоб с задней стенки глотки», «соскоб с
# зева» и синоним «мазок на флору» — из среза МИС (журнал, g0542); остальное — синтетика теста: тот же материал у чужой
# строки, другой материал у второй строки класса.
_SERVICE_INFO = [
    {"serviceName": _POSEV, "biomatNames": f"{_THROAT}, соскоб с зева, моча, отделяемое уретры", "serviceSynonyms": "мазок на флору"},
    {"serviceName": "Мазок из зева и носа на патогенный стафилококк", "biomatNames": f"{_THROAT}, отделяемое носа", "serviceSynonyms": None},
    {"serviceName": "Цитология (мазок на атипичные клетки)", "biomatNames": f"{_CERVIX}, соскоб из цервикального канала", "serviceSynonyms": None},
    {"serviceName": "Общий анализ крови", "biomatNames": "кровь из вены", "serviceSynonyms": "оак"},
]

# Реплика · фраза разбора · строка, которую LLM выбрала бы, увидев материал · материал, который назвал пациент.
_CASES = {
    "throat": ("Сколько стоит мазок на микрофлору с поверхности задней стенки глотки", "мазок на микрофлору с поверхности задней стенки глотки", "Посев на микрофлору", _THROAT),
    "cervix": ("сколько стоит онкоцитология мазок с шейки матки", "онкоцитология мазок с шейки матки", "Цитология (мазок на атипичные клетки)", _CERVIX),
}


class _ByMaterial(_Oracle):
    """Разбор — фраза `phrase`. Выбор строки — как LLM, которая судит по тексту позиции: строка `pick` подходит, только
    если в тексте позиции есть материал пациента `material`; иначе «ни одна» (`other_service`) с терминами, которые
    находят ту же строку во втором круге. Тексты позиций каждого круга — в `pools`."""

    def __init__(self, phrase: str, pick: str, material: str) -> None:
        super().__init__((phrase,))
        self.pick = pick
        self.material = material
        self.pools: list[list[str]] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if not any(mark in seen for mark in _ROWS_MARKS):
            return await super().generate(prompt=prompt, system=system, **kwargs)
        options = [(int(head), rest) for head, dot, rest in (line.partition(". ") for line in prompt.splitlines()) if dot and head.isdigit()]
        self.pools.append([text for _n, text in options])
        picks = [n for n, text in options if text.startswith(self.pick) and self.material in text]
        answer = {"picks": picks, "reason": "none" if picks else "other_service", "search_terms": [] if picks else [self.pick]}
        return GenerateResponse(model="m", response=json.dumps(answer, ensure_ascii=False), done=True)


# Слова реплики о глотке есть в полном прайсе (на проде гард П2 «уточнения нет в прайсе» реплику пропускает, и её судит
# LLM), а в мини-прайсе их нет — гард отсёк бы реплику до выбора строки. Две строки вместо полного прайса: первая —
# строка, на которую вопрос приземлялся до 10.08 (`_biomaterial`, «Зачем модуль существует»), вторая — синтетика со
# словом «поверхности»; цены и коды — синтетика теста.
_EXTRA_ROWS = [
    {"serviceId": 990001, "priceUnitId": 155, "serviceHomecode": "T-0001", "serviceName": "Обработка задней стенки глотки расфокусированным лучом СО2 лазера", "deadline": "", "cost": 1500.0, "promotion": 0},
    {"serviceId": 990002, "priceUnitId": 155, "serviceHomecode": "T-0002", "serviceName": "Криодеструкция образований на поверхности кожи", "deadline": "", "cost": 900.0, "promotion": 0},
]


@pytest.fixture(scope="module")
def mini_prod_world(tmp_path_factory) -> Path:
    """Мир мини-прода и строки `_EXTRA_ROWS` в прайсе региона."""

    source = tmp_path_factory.mktemp("mis_notes_src") / "snapshots"
    shutil.copytree(FIXTURE, source)
    [price] = source.glob("price_by_region/price_region_3_*.jsonl")
    with price.open("a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in _EXTRA_ROWS)
    return ev_world.build_world(source, "20261008", tmp_path_factory.mktemp("mis_notes") / "world")


@pytest.fixture
def mis_vocab(monkeypatch):
    """Словарь МИС мира теста (поверх герметичного `_hermetic_mis_synonyms`: тест подставляет свой явно)."""

    from messengers_router.services import _biomaterial as bio

    vocab = bio._build_vocabularies(_SERVICE_INFO)
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    return vocab


def _sid(*parts: str) -> str:
    return "mis-" + "-".join(re.sub(r"\W+", "_", part)[-14:] for part in parts)


def _turn(case: str, world, monkeypatch, notes: str):
    reply, phrase, pick, material = _CASES[case]
    oracle = _ByMaterial(phrase, pick, material)
    [turn], [row] = _run([reply], world, monkeypatch, llm=oracle, session=_sid(notes, case), MR_PRICE_OPTION_MIS_NOTES=notes)
    return oracle, turn, row


@pytest.mark.parametrize("case", _CASES)
def test_named_material_reaches_the_llm_and_the_row_is_priced(case, mini_prod_world, mis_vocab, monkeypatch):
    """Строка принимает материал пациента по МИС — LLM видит его у позиции и выбирает её: цена этой строки (на старом
    коде LLM видела только «название — цена», отвергала строку в обоих кругах — «Не нашёл»)."""

    _reply, _phrase, pick, material = _CASES[case]
    oracle, turn, row = _turn(case, mini_prod_world, monkeypatch, "on")
    marked = [text for pool in oracle.pools for text in pool if text.startswith(pick)]
    assert marked and all(f"материал — {material}" in text for text in marked), oracle.pools
    assert pick in turn["text"] and _priced(turn["text"]), turn["text"]
    assert row.get("price_policy") == "picked", row.get("price_policy")
    assert row.get("price_mis_notes", 0) >= 1


def test_note_shows_only_what_the_patient_named(mini_prod_world, mis_vocab, monkeypatch):
    """В пометке — только материал и синоним, совпавшие со словами реплики: «соскоб с зева», «моча», «кровь из вены» на
    вопрос о глотке не показываются; позиция без совпадения — текст как раньше; пометка не длиннее предела."""

    from messengers_router.services import _price_select_llm as S

    oracle, _turn_record, _row = _turn("throat", mini_prod_world, monkeypatch, "on")
    texts = [text for pool in oracle.pools for text in pool]
    posev = [text for text in texts if text.startswith("Посев на микрофлору")]
    assert posev and all("другое название — мазок на флору" in text for text in posev), posev
    for text in texts:
        assert "соскоб с зева" not in text and "моча" not in text and "кровь из вены" not in text, text
        name, _dash, tail = text.partition(" — ")
        price, _space, note = tail.partition(" руб.")
        assert price.isdigit(), text
        assert note == "" or (note.startswith(" (по справочнику клиники: ") and len(note) <= S._MIS_NOTE_MAX + 3), text


@pytest.mark.parametrize("case", _CASES)
def test_switch_off_is_byte_identical_and_pool_unchanged(case, mini_prod_world, mis_vocab, monkeypatch):
    """off — тексты позиций прежние («название — цена»), ответ прежний: без цены (о глотке — «Не нашёл», как на проде
    19.10); on — те же строки в том же порядке: пометка не добавляет кандидатов и не меняет их порядок."""

    off, turn_off, row_off = _turn(case, mini_prod_world, monkeypatch, "off")
    on, _turn_on, _row_on = _turn(case, mini_prod_world, monkeypatch, "on")
    assert all(re.fullmatch(r".+ — \d+ руб\.", text) for pool in off.pools for text in pool), off.pools
    assert not _priced(turn_off["text"]), turn_off["text"]
    assert case != "throat" or "Не нашёл" in turn_off["text"], turn_off["text"]
    assert "price_mis_notes" not in row_off
    first_on = [text.split(" (по справочнику клиники: ", 1)[0] for text in on.pools[0]]
    assert first_on == off.pools[0], (first_on, off.pools[0])


def test_note_is_bounded_by_dropping_items_not_cutting_words(monkeypatch):
    """Длинные данные МИС — пометка не длиннее предела: уходят названия целиком (сначала лишние, затем синоним), слова
    не режутся; не влезает ни одно — пометки нет. Строки нет в справочнике — пометки нет."""

    from messengers_router.services import _biomaterial as bio
    from messengers_router.services import _price_select_llm as S
    from messengers_router.services._prices_helpers import _price_query_tokens

    wanted = set(_price_query_tokens(_CASES["throat"][0]))
    long = "мазок с поверхности задней стенки глотки " + "и очень длинное уточнение " * 5
    for biomat, synonyms in ((f"{_THROAT}, {long.strip()}", "мазок на флору из задней стенки глотки"), (long.strip(), long.strip())):
        vocab = bio._build_vocabularies([{"serviceName": _POSEV, "biomatNames": biomat, "serviceSynonyms": synonyms}])
        monkeypatch.setattr(bio, "_vocabularies", lambda vocab=vocab: vocab)
        materials, names = bio.service_mis_names(_POSEV)
        note = S._mis_note({"serviceName": _POSEV, "cost": 1200}, wanted)
        assert len(note) <= S._MIS_NOTE_MAX, note
        shown = [item for part in note.removeprefix("по справочнику клиники: ").split("; ") if part for item in part.split(" — ", 1)[1].split(", ")]
        assert all(item in materials + names for item in shown), (note, materials, names)
    assert note == ""  # единственные материал и синоним длиннее предела
    assert S._mis_note({"serviceName": "Строки нет в МИС", "cost": 1}, wanted) == ""


def test_notes_of_one_list_are_bounded(monkeypatch):
    """Материал, общий для всех строк («кровь на ТТГ» → «кровь из вены» у каждой), — пометки всего списка не длиннее
    `_MIS_NOTES_POOL_MAX`, первыми помечены строки в начале списка (ответ правил); off — ни одной пометки."""

    from messengers_router.runtime_config import config as cfg
    from messengers_router.services import _biomaterial as bio
    from messengers_router.services import _price_select_llm as S

    pool = [{"serviceName": f"Тиреотропный гормон, вариант {n}", "cost": 300 + n} for n in range(24)]
    vocab = bio._build_vocabularies([{"serviceName": row["serviceName"], "biomatNames": "кровь из вены, кровь капиллярная", "serviceSynonyms": None} for row in pool])
    monkeypatch.setattr(bio, "_vocabularies", lambda: vocab)
    monkeypatch.setattr(cfg, "MR_PRICE_OPTION_MIS_NOTES", "on", raising=False)
    texts = [option.text for option in S._options("сколько стоит кровь на ттг", pool)]
    plain = [f"{row['serviceName']} — {row['cost']} руб." for row in pool]
    added = sum(len(text) - len(base) for text, base in zip(texts, plain))
    marked = [text != base for text, base in zip(texts, plain)]
    assert 0 < added <= S._MIS_NOTES_POOL_MAX + 3 * sum(marked), added
    assert marked == sorted(marked, reverse=True) and not all(marked), marked
    assert "по справочнику клиники: материал — кровь из вены, кровь капиллярная" in texts[0], texts[0]
    monkeypatch.setattr(cfg, "MR_PRICE_OPTION_MIS_NOTES", "off", raising=False)
    assert [option.text for option in S._options("сколько стоит кровь на ттг", pool)] == plain
