"""Класс: правила промпта выбора строки — только на пути ответа о цене (BUG-2026-10-24-FOREIGN-PRICE-D17; выключатель
`price_select_rules` снят 31.10 — режим `general` на проде с деплоя 17, единственный).

Карточка 24.10: «консультация врача» (4 варианта из 4) — LLM выбирала приёмы всех специальностей (правило 5 «группа услуг —
все позиции»), и пациент видел одну случайную цену или список чужих приёмов (против решения 25.09); «эхо сердца», «УЗИ
сердца» — ЭхоКГ плода при отсутствии обычной в списке. Первый вариант правил (раздел прайса, «только похоже — нет», марка)
давал новую чужую цену (ЛОР → фониатр), был строже нужного и менял запись («запись к неврологу» → уточнение) — снят.

Инварианты:
- путь ответа о цене — промпты `*_general`: слишком общее название («анализы», приём врача без специальности) — не группа правила 5; прочие строки промпта —
  прежние (режим `groups` — «плод, беременность, до года» — снят: карточка 24.10, test — две новые чужие цены; первая
  формулировка `general` делала «анализы» группой — «стоимость анализов» → список из 8 анализов);
- суд записи (`price_answer=False`) — прежний промпт;
- промпты — в двух локациях, одинаковые.
"""

from __future__ import annotations

import pathlib

import pytest

from messengers_router.services import _price_select_llm as sel

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASES = ["choose_price_service", "choose_price_service_qualifier"]


@pytest.mark.parametrize(("judge", "task"), [(False, "price_service_general"), (True, "price_service_qualifier_general")])
def test_price_answer_reads_the_rules_prompt(judge, task) -> None:
    assert sel.price_task(judge_qualifier=judge).name == task


@pytest.mark.parametrize(("judge", "task"), [(False, "price_service"), (True, "price_service_qualifier")])
def test_booking_judge_keeps_the_old_prompt(judge, task) -> None:
    assert sel.price_task(judge_qualifier=judge, price_answer=False).name == task


def test_booking_judge_passes_price_answer_false() -> None:
    source = (_ROOT / "messengers_router/services/_booking_service_select.py").read_text(encoding="utf-8")
    assert "price_answer=False" in source


def _lines(name: str) -> list[str]:
    return (_ROOT / f"messengers_router/prompts/{name}.txt").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("base", _BASES)
def test_rules_prompt_differs_from_the_old_one_only_by_its_rule(base) -> None:
    old, general = _lines(base), _lines(f"{base}_general")
    changed = [i for i, (a, b) in enumerate(zip(old, general)) if a != b]
    assert len(old) == len(general)
    assert [old[i][:2] for i in changed] == ["5.", "- "], "правило 5 и код причины too_general"
    rule5, reason = general[changed[0]], general[changed[1]]
    for word in ("слишком общее название — не группа", "«анализы»", "консультация врача"):
        assert word in rule5, word
    assert "приём врача без специальности" in reason
    for line in general:
        assert "раздел прайса" not in line and "производител" not in line and "для плода" not in line  # снятые правила


@pytest.mark.parametrize("base", _BASES)
def test_rules_prompt_lives_in_both_places(base) -> None:
    repo = (_ROOT / f"messengers_router/prompts/{base}_general.txt").read_text(encoding="utf-8")
    assert repo == (_ROOT / f"app_data/prompts/mr_{base}_general.txt").read_text(encoding="utf-8")


def test_options_carry_no_price_section(monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_OPTION_MIS_NOTES", "off", raising=False)
    rows = [{"serviceName": "Эхокардиография", "cost": 3500, "priceUnitId": 175}]
    assert [o.text for o in sel._options("эхо сердца", rows)] == ["Эхокардиография — 3500 руб."]
