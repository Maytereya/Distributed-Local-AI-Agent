"""Тесты для `messengers_router.services._unit_canonicalisation`.

Проверяем:
1. Известные unit-композиты НЕ матчатся на чужие базовые специальности
   (мануальный терапевт ≠ терапевт, стоматолог-ортопед ≠ ортопед, и т.п.).
2. Множественные роли матчатся правильно (травматолог-ортопед матчит
   и «травматолог» и «ортопед»).
3. Неизвестные unit-имена возвращают `None` (caller использует fallback).
4. УЗИ helper различает «УЗИ-врачей» и врачей, использующих УЗИ как
   инструмент.
5. Все канонические значения в карте присутствуют в `SPECIALTY_CANONICAL`
   (или явно расширяют его) — sanity на консистентность.
"""

from __future__ import annotations

import pytest

from messengers_router.services._unit_canonicalisation import (
    _all_canonical_targets,
    _all_known_unit_keys,
    canonicalise_unit,
    is_uzi_unit,
    known_units_count,
    unit_matches_specialty,
)
from messengers_router.specialty_parser import SPECIALTY_CANONICAL


# ---------------------------------------------------------------------------
# 1. Композитные специальности НЕ должны матчиться на базовый термин
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit_name, base_specialty",
    [
        # Арцыбашев — «Врач-мануальный терапевт» НЕ должен попадать в «терапевт».
        ("Врач-мануальный терапевт", "терапевт"),
        # Сидоров — «Врач-стоматолог-ортопед» НЕ должен попадать в «ортопед».
        ("Врач-стоматолог-ортопед", "ортопед"),
        ("Врач-стоматолог-ортопед", "травматолог"),
        # Пластические хирурги НЕ должны попадать в «хирург».
        ("Врач-пластический хирург", "хирург"),
        # Гинеколог-эндокринолог — гинеколог, НЕ general endocrinologist.
        ("Врач гинеколог-эндокринолог", "эндокринолог"),
        # Нейрохирург — самостоятельная специальность, НЕ general хирург.
        ("Врач-нейрохирург", "хирург"),
    ],
)
def test_composite_unit_does_not_match_base_specialty(
    unit_name: str, base_specialty: str
) -> None:
    assert unit_matches_specialty(unit_name, base_specialty) is False


# ---------------------------------------------------------------------------
# 2. Композиты и единичные роли матчатся СВОЕЙ специальностью
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit_name, target_specialty",
    [
        # — single-role —
        ("Врач-терапевт", "терапевт"),
        ("Врач терапевт", "терапевт"),  # variant без дефиса
        ("Врач-кардиолог", "кардиолог"),
        ("Врач-уролог", "уролог"),
        ("Врач-онколог", "онколог"),
        ("Врач-эндокринолог", "эндокринолог"),
        ("Врач-стоматолог", "стоматолог"),
        ("Врач-хирург", "хирург"),
        # — composite multi-role: оба компонента легитимны —
        ("Врач акушер-гинеколог", "гинеколог"),
        ("Врач акушер-гинеколог", "акушер"),
        ("Врач-травматолог-ортопед", "травматолог"),
        ("Врач-травматолог-ортопед", "ортопед"),
        ("Врач-анестезиолог-реаниматолог", "анестезиолог"),
        ("Врач-анестезиолог-реаниматолог", "реаниматолог"),
        ("Врач уролог-андролог", "уролог"),
        ("Врач уролог-андролог", "андролог"),
        ("Врач аллерголог-иммунолог", "аллерголог"),
        ("Врач аллерголог-иммунолог", "иммунолог"),
        # — composite NOT-subset: матчится своим именем —
        ("Врач-мануальный терапевт", "мануальный терапевт"),
        ("Врач-пластический хирург", "пластический хирург"),
        ("Врач-стоматолог-ортопед", "стоматолог"),  # это всё ещё стоматолог
        ("Врач гинеколог-эндокринолог", "гинеколог"),
        ("Врач-нейрохирург", "нейрохирург"),
        # — синонимы / расширения —
        ("Врач-оториноларинголог", "лор"),
        ("Врач-оториноларинголог", "оториноларинголог"),
        ("Врач-дерматовенеролог", "дерматолог"),
        ("Врач-колопроктолог", "проктолог"),
    ],
)
def test_unit_matches_target_specialty(unit_name: str, target_specialty: str) -> None:
    assert unit_matches_specialty(unit_name, target_specialty) is True


# ---------------------------------------------------------------------------
# 3. Неизвестные unit-имена возвращают None (fallback к legacy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unit_name",
    [
        "Врач-кардиоторакальный хирург",  # нового хирургического профиля нет
        "Врач-космонавт",  # явно отсутствующее
        "Random Doctor 42",
        "",  # пустая строка обработана отдельно — ниже
    ],
)
def test_unknown_unit_returns_none_or_empty(unit_name: str) -> None:
    result = canonicalise_unit(unit_name)
    if not unit_name.strip():
        # Пустая строка — это «known empty», не fallback.
        assert result == frozenset()
    else:
        assert result is None
        # Для unit_matches_specialty это означает None — caller обязан
        # использовать legacy substring matcher.
        assert unit_matches_specialty(unit_name, "хирург") is None


# ---------------------------------------------------------------------------
# 4. УЗИ helper строго различает профильных vs использующих как инструмент
# ---------------------------------------------------------------------------

def test_is_uzi_unit_for_dedicated_uzi_unit() -> None:
    assert is_uzi_unit("Врач ультразвуковой диагностики") is True
    assert is_uzi_unit("УЗИ (центр)") is True


def test_is_uzi_unit_for_non_uzi_units() -> None:
    # Эндокринолог делает УЗИ как инструмент — но это не УЗИ-врач.
    assert is_uzi_unit("Врач-эндокринолог") is False
    # Гинеколог тоже работает с УЗИ при приёме — но не профильный.
    assert is_uzi_unit("Врач акушер-гинеколог") is False
    # Кардиолог не УЗИ-врач.
    assert is_uzi_unit("Врач-кардиолог") is False


def test_is_uzi_unit_for_unknown_returns_none() -> None:
    assert is_uzi_unit("Какой-то новый unit") is None


# ---------------------------------------------------------------------------
# 5. Sanity: канонические targets уважают глобальный enum
# ---------------------------------------------------------------------------

def test_canonical_targets_subset_of_specialty_canonical() -> None:
    """Все канонические специальности из карты должны быть либо в
    `SPECIALTY_CANONICAL`, либо явно расширяющие enum."""
    canonical = _all_canonical_targets()
    expected_canonical = {s.lower() for s in SPECIALTY_CANONICAL}

    # Дополнительно разрешаем технические маркеры, которых может не
    # быть в SPECIALTY_CANONICAL (синонимы / составные).
    extras = {
        "узи",
        "лор",
        "экг",
        "функциональная диагностика",
        "репродуктолог",  # новая специальность, ещё не в enum
        "акушер",  # без «-гинеколог» — пациент может спросить отдельно
        "маммолог",  # пациент может искать маммолога самостоятельно
        "челюстно-лицевой хирург",  # отдельная специальность из live-кэша, ещё не в enum
    }
    allowed = expected_canonical | extras

    unknown = canonical - allowed
    assert not unknown, (
        f"Канонизация ссылается на специальности, не зарегистрированные "
        f"в SPECIALTY_CANONICAL: {sorted(unknown)}. Добавь в enum или "
        f"в `extras` теста с обоснованием."
    )


def test_known_units_have_reasonable_count() -> None:
    """Sanity: карта должна покрывать сегодняшние данные (~45 unit'ов)."""
    assert known_units_count() >= 35


# ---------------------------------------------------------------------------
# 6. Smoke на текущих живых данных doctors_*.jsonl
# ---------------------------------------------------------------------------

def test_smoke_against_live_doctors_cache() -> None:
    """Все unit_names из текущего doctors-кэша должны быть либо в
    карте канонизации, либо известно отсутствовать там (для ранее
    неизвестных будет лог)."""
    import glob
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    candidates = sorted(
        glob.glob(str(repo / "agent_logic_2/nayka_api/apidata/doctors_*.jsonl"))
        + glob.glob(str(repo / "app_data/nayka_api/apidata/doctors_*.jsonl"))
    )
    if not candidates:
        pytest.skip("doctor cache file not present in this checkout")

    latest = candidates[-1]
    seen_units: set[str] = set()
    with open(latest, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for ul in d.get("unit_links") or []:
                if isinstance(ul, dict):
                    n = (ul.get("company_unit_name") or "").strip()
                    if n:
                        seen_units.add(n)
            for n in d.get("units") or []:
                if isinstance(n, str) and n.strip():
                    seen_units.add(n.strip())

    known = _all_known_unit_keys()
    # Нормализуем seen_units по тому же ключу, что карта
    import re

    from messengers_router.russian_nlu import normalize_ru

    def _key(t: str) -> str:
        return re.sub(r"\s+", " ", normalize_ru(t)).strip()

    seen_keys = {_key(u) for u in seen_units}
    missing = seen_keys - known
    # На текущем cache мы хотим, чтобы НИ ОДНО unit-имя не оставалось
    # неизвестным карте — иначе legacy substring fallback пропустит
    # ложные срабатывания вроде Решетова → УЗИ.
    assert not missing, (
        f"Unit names в живом doctors-кэше отсутствуют в карте "
        f"канонизации: {sorted(missing)}. Добавь в _UNIT_TO_CANONICAL_RAW."
    )
