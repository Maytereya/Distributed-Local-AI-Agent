"""Уточнения позиции, этап 3 — пул выбора строки по материалу (DESIGN-2026-10-28-POSITION-QUALIFIERS; `qualifier_pool = on`) — через
обработчик эндпоинта, мини-прод, день, сеть закрыта.

Класс (решение владельца 27.10, п. 7; g0372 «посев кала на микрофлору» → общий «Посев на микрофлору» 1 120 / 1 200 руб.): материал,
который назвал пациент, теряется при сборе кандидатов по буквам названия. Есть ли материал в строке, решают данные МИС
(`biomatNames` справочника услуг; у строки без материалов в справочнике — слова её названия), сверка — по основам слов.

Инварианты класса:
- строка, которая по данным МИС материал позиции не принимает, не попадает в список выбора LLM и не получает цену — ни в первом круге,
  ни во втором (по терминам LLM), ни у позиции списка (корзина);
- строки с этим материалом попадают в список выбора, хотя слов фразы пациента в их названии нет;
- встречные: вопрос без материала — кандидаты как были; выключатель off — как было; материал у строки без данных справочника — по
  словам её названия («вен» у «Дуплексного сканирования вен»).

Справочник МИС в тесте — свой (`_build_vocabularies` из строк ниже): гейт держит синонимы герметичными, а материалы в CI без среза пусты.
"""

from __future__ import annotations

import re

import pytest

from evalbench import world as ev_world
from messengers_router import llm_runtime
from messengers_router.services import _biomaterial, _qualifier_pool
from tests.spans_oracle import SpansOracle, keep_llm_breaker_closed
from tests.test_price_turns_golden import mini_prod_world, run_dialog  # noqa: F401 — фикстура
from tests.test_turn_parse_slot import _deploy
from tests.test_turn_parse_t3 import _ROWS_MARKS, _Oracle

_GENERIC = "Посев на микрофлору (идентификация агента и чувствительность к а\\б)"
_MIS = [
    {"serviceName": _GENERIC, "biomatNames": "соскоб с зева, мазок из носоглотки, отделяемое ран, ректальное отделяемое, желчь"},
    {"serviceName": "Посев на грибы рода Candida (чувствительность к антимикотическим препаратам)", "biomatNames": "кал, моча, соскоб с зева"},
    {"serviceName": "Кал на яйца глист", "biomatNames": "кал"},
    {"serviceName": "Исследование кала на простейшие (бластоцисты, амебы, лямблии и др.)", "biomatNames": "кал"},
    {"serviceName": "Посев мочи с идентификацией культур", "biomatNames": "левая почка, правая почка, мочевой пузырь, моча"},
]


@pytest.fixture
def mis_materials(monkeypatch):
    vocab = _biomaterial._build_vocabularies(_MIS)
    monkeypatch.setattr(_biomaterial, "_vocabularies", lambda: vocab)


class _Rows(_Oracle):
    """Выбор строк — строки с началом названия `rows`; запоминает названия, показанные LLM выбора (`offered`)."""

    def __init__(self, rows: tuple[str, ...]) -> None:
        super().__init__((), rows)
        self.offered: list[str] = []

    async def generate(self, *, prompt: str = "", system: str | None = None, **kwargs):
        seen = f"{system}\n\n{prompt}" if system else prompt
        if any(mark in seen for mark in _ROWS_MARKS):
            for line in prompt.splitlines():
                head, dot, rest = line.partition(". ")
                if dot and head.isdigit():
                    self.offered.append(rest.split(" → ", 1)[-1].rsplit(" — ", 1)[0])
        return await super().generate(prompt=prompt, system=system, **kwargs)


def _run(reply: str, oracle: SpansOracle, world, monkeypatch, session: str, *, pool_on: bool = True) -> tuple[dict, dict]:
    from messengers_router.runtime_config import config as cfg

    _deploy(monkeypatch)
    keep_llm_breaker_closed(monkeypatch)
    monkeypatch.setattr(cfg, "MR_POSITION_QUALIFIERS", "on", raising=False)
    monkeypatch.setattr(cfg, "MR_QUALIFIER_POOL", "on" if pool_on else "off", raising=False)
    monkeypatch.setattr(llm_runtime, "_OLLAMA_CLIENT", oracle)
    with ev_world.FrozenWorld(world, "day") as frozen:
        [turn] = run_dialog([reply], session)
        rows = [row for row in frozen.telemetry if row.get("kind", "turn") == "turn"]
        attempts = list(frozen.net.attempts)
    assert attempts == []
    return turn, rows[-1]


def _priced(text: str, name_start: str) -> bool:
    return bool(re.search(rf"{re.escape(name_start)}[^\n]*— [\d ]+ руб", text))


def _sid(*parts: str) -> str:
    return "qp-" + "-".join(re.sub(r"\W+", "_", part)[-16:] for part in parts)


_STOOL = "сколько стоит посев кала на микрофлору"


def test_material_the_row_lacks_keeps_it_out_of_the_choice(mis_materials, mini_prod_world, monkeypatch):  # noqa: F811
    inner = _Rows(("Посев",))
    oracle = SpansOracle(("посев кала на микрофлору",), inner=inner, qualifiers=[("посев кала на микрофлору", "кала", "material")])
    turn, row = _run(_STOOL, oracle, mini_prod_world, monkeypatch, _sid("stool"))
    assert inner.offered and _GENERIC not in inner.offered, inner.offered
    assert any(name.startswith("Кал на яйца глист") for name in inner.offered), inner.offered  # строка с материалом — без слов фразы
    assert not _priced(turn["text"], "Посев на микрофлору"), turn["text"]
    assert row["pool_material"].split("/")[1] != "0"  # общий посев был в пуле по буквам и ушёл по данным


def test_list_position_with_material_keeps_the_generic_row_out(mis_materials, mini_prod_world, monkeypatch):  # noqa: F811
    """Позиция списка (корзина) — тот же пул по материалу; соседняя позиция — своей ценой."""

    inner = _Rows(("Посев",))
    oracle = SpansOracle(
        ("посев кала на микрофлору", "ферритин"), inner=inner, qualifiers=[("посев кала на микрофлору", "кала", "material")]
    )
    turn, _row = _run("сколько стоит посев кала на микрофлору и ферритин", oracle, mini_prod_world, monkeypatch, _sid("list"))
    assert inner.offered and _GENERIC not in inner.offered, inner.offered
    assert any(name.startswith("Кал на яйца глист") for name in inner.offered), inner.offered
    assert not _priced(turn["text"], "Посев на микрофлору") and _priced(turn["text"], "Ферритин"), turn["text"]


def test_without_material_the_generic_rows_stay(mis_materials, mini_prod_world, monkeypatch):  # noqa: F811
    inner = _Rows(("Посев на микрофлору",))
    oracle = SpansOracle(("посев на микрофлору",), inner=inner, qualifiers=[])
    turn, row = _run("сколько стоит посев на микрофлору", oracle, mini_prod_world, monkeypatch, _sid("plain"))
    assert _GENERIC in inner.offered and _priced(turn["text"], "Посев на микрофлору"), turn["text"]
    assert "pool_material" not in row


def test_off_offers_the_generic_rows_as_before(mis_materials, mini_prod_world, monkeypatch):  # noqa: F811
    inner = _Rows(("Посев на микрофлору",))
    oracle = SpansOracle(("посев кала на микрофлору",), inner=inner, qualifiers=[("посев кала на микрофлору", "кала", "material")])
    turn, row = _run(_STOOL, oracle, mini_prod_world, monkeypatch, _sid("off"), pool_on=False)
    assert _GENERIC in inner.offered and "pool_material" not in row


# --------------------------------------------------------------------------- данные: сверка материала


@pytest.mark.parametrize(
    ("a", "b", "same"),
    [("кала", "кал", True), ("мочи", "моча", True), ("крови", "кровь", True), ("вен", "вены", True),
     ("кала", "кальций", False), ("мочи", "мочевой", False), ("вен", "венерических", False), ("ка", "ка", False)],
)
def test_same_word_is_a_word_form_not_a_prefix(a, b, same):
    assert _qualifier_pool._same_word(a, b) is same


def test_accepts_by_mis_materials_then_by_the_name(mis_materials):
    assert _qualifier_pool.accepts({"serviceName": _GENERIC}, "кала") is False
    assert _qualifier_pool.accepts({"serviceName": "Кал на яйца глист"}, "кала") is True
    assert _qualifier_pool.accepts({"serviceName": "Посев мочи с идентификацией культур"}, "мочи") is True
    # Нет в справочнике — по словам названия.
    assert _qualifier_pool.accepts({"serviceName": "Дуплексное сканирование вен нижних конечностей"}, "вен") is True
    assert _qualifier_pool.accepts({"serviceName": "УЗИ слюнных желез с доплерографией (УЗДГ)"}, "вен") is False
