"""Класс: правила промпта выбора строки — только на пути ответа о цене (BUG-2026-10-24-FOREIGN-PRICE-D17, `price_select_rules`).

Карточка 24.10: «консультация врача» (4 варианта из 4) — LLM выбирала приёмы всех специальностей (правило 5 «группа услуг —
все позиции»), и пациент видел одну случайную цену или список чужих приёмов (против решения 25.09); «эхо сердца», «УЗИ
сердца» — ЭхоКГ плода при отсутствии обычной в списке. Первый вариант правил (раздел прайса, «только похоже — нет», марка)
давал новую чужую цену (ЛОР → фониатр), был строже нужного и менял запись («запись к неврологу» → уточнение) — снят.

Инварианты:
- `general` — приём врача без специальности — «слишком общее»; `groups` — то же и особая группа пациентов (плод,
  беременность, до года) — только если её назвал пациент; прочие строки промптов — прежние;
- суд записи (`price_answer=False`) — прежний промпт при любом режиме;
- промпты — в двух локациях, одинаковые; off — прежние задачи.
"""

from __future__ import annotations

import pathlib

import pytest

from messengers_router.services import _price_select_llm as sel

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BASES = ["choose_price_service", "choose_price_service_qualifier"]


def _mode(monkeypatch, mode: str) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_SELECT_RULES", mode, raising=False)
    monkeypatch.setattr(cfg, "MR_PRICE_CANDIDATES_BY_MEANING", "on", raising=False)


@pytest.mark.parametrize(
    ("mode", "judge", "task"),
    [
        ("off", False, "price_service"),
        ("off", True, "price_service_qualifier"),
        ("general", False, "price_service_general"),
        ("general", True, "price_service_qualifier_general"),
        ("groups", False, "price_service_groups"),
        ("groups", True, "price_service_qualifier_groups"),
        ("мусор", False, "price_service"),
    ],
)
def test_price_answer_reads_the_rules_prompt_of_its_mode(mode, judge, task, monkeypatch) -> None:
    _mode(monkeypatch, mode)
    assert sel.price_task(judge_qualifier=judge).name == task


@pytest.mark.parametrize("mode", ["off", "general", "groups"])
@pytest.mark.parametrize("judge", [False, True])
def test_booking_judge_keeps_the_old_prompt(mode, judge, monkeypatch) -> None:
    _mode(monkeypatch, mode)
    assert sel.price_task(judge_qualifier=judge, price_answer=False).name in {"price_service", "price_service_qualifier"}


def test_booking_judge_passes_price_answer_false() -> None:
    source = (_ROOT / "messengers_router/services/_booking_service_select.py").read_text(encoding="utf-8")
    assert "price_answer=False" in source


def _lines(name: str) -> list[str]:
    return (_ROOT / f"messengers_router/prompts/{name}.txt").read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("base", _BASES)
def test_rules_prompts_differ_from_the_old_one_only_by_their_rules(base) -> None:
    old, general, groups = _lines(base), _lines(f"{base}_general"), _lines(f"{base}_groups")
    changed_general = [i for i, (a, b) in enumerate(zip(old, general)) if a != b]
    changed_groups = [i for i, (a, b) in enumerate(zip(general, groups)) if a != b]
    assert len(old) == len(general) == len(groups)
    assert [old[i][:2] for i in changed_general] == ["5.", "- "], "general: правило 5 и код причины too_general"
    assert "консультация врача" in general[changed_general[0]] and "приём врача без специальности" in general[changed_general[1]]
    assert [general[i][:2] for i in changed_groups] == ["4."], "groups: только правило 4"
    assert "для плода, при беременности, до 1 года" in groups[changed_groups[0]]
    for line in (*general, *groups):
        assert "раздел прайса" not in line and "производител" not in line  # снятые правила первого варианта


@pytest.mark.parametrize("base", _BASES)
@pytest.mark.parametrize("mode", ["general", "groups"])
def test_rules_prompts_live_in_both_places(base, mode) -> None:
    repo = (_ROOT / f"messengers_router/prompts/{base}_{mode}.txt").read_text(encoding="utf-8")
    assert repo == (_ROOT / f"app_data/prompts/mr_{base}_{mode}.txt").read_text(encoding="utf-8")


def test_options_carry_no_price_section(monkeypatch) -> None:
    from messengers_router.runtime_config import config as cfg

    monkeypatch.setattr(cfg, "MR_PRICE_OPTION_MIS_NOTES", "off", raising=False)
    _mode(monkeypatch, "groups")
    rows = [{"serviceName": "Эхокардиография", "cost": 3500, "priceUnitId": 175}]
    assert [o.text for o in sel._options("эхо сердца", rows)] == ["Эхокардиография — 3500 руб."]
