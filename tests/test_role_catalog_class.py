"""Каталог ролей из среза врачей МИС (ход (г), L-17, этап 1) — тесты на класс.

Инвариант класса (принцип 1 и «данные из МИС, не списки в коде»): какие роли приёма есть у клиники, знает
только срез врачей. Роль, которую МИС завела, бот узнаёт по её же названию без правки кода (AUT-5); роль,
которой нет у врачей региона, в каталоге не появляется; общая роль не подменяет роль с уточнением.
Журнал: STEP-2026-10-20-G-ROLES-CATALOG.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from messengers_router.services._role_catalog import build_role_catalog

_EXCERPT = Path(__file__).resolve().parent / "fixtures" / "doctors_excerpt" / "doctors_20261008.jsonl"
_HERE, _AWAY = 101, 909  # филиал региона бота и филиал другого города


def _doctor(*units: tuple[int, str], branch: int = _HERE) -> dict:
    return {
        "region_ids": [branch],
        "unit_links": [{"company_unit_id": uid, "company_unit_name": name, "main": False} for uid, name in units],
    }


def _here(doctor: dict) -> bool:
    return _HERE in (doctor.get("region_ids") or [])


def _names(roles) -> list[str]:
    return [role.name for role in roles]


def test_new_mis_role_is_recognised_without_code_change():
    """AUT-5: специальности, которой нет ни в одном списке кода, достаточно появиться в МИС."""

    without = build_role_catalog([_doctor((1, "Врач-кардиолог"))], in_region=_here)
    with_new = build_role_catalog([_doctor((1, "Врач-кардиолог")), _doctor((777, "Врач-сомнолог"))], in_region=_here)

    assert _names(without.find("Есть ли у вас сомнолог?").exact) == []
    assert _names(with_new.find("Есть ли у вас сомнолог?").exact) == ["сомнолог"]
    assert _names(with_new.find("запишите к сомнологу").exact) == ["сомнолог"]
    assert with_new.by_unit(777).name == "сомнолог"


def test_role_of_other_region_only_is_not_in_catalog():
    catalog = build_role_catalog(
        [_doctor((1, "Врач-кардиолог")), _doctor((2, "Врач-репродуктолог"), branch=_AWAY)], in_region=_here
    )

    assert _names(catalog.roles) == ["кардиолог"]
    assert catalog.find("к репродуктологу") == catalog.find("")


def test_units_with_one_display_name_are_one_role():
    catalog = build_role_catalog(
        [_doctor((109, "Врач-ревматолог")), _doctor((323, "Ревматолог (центр)")), _doctor((109, "Врач-ревматолог"), (323, "Ревматолог (центр)"))],
        in_region=_here,
    )

    (role,) = catalog.roles
    assert (role.name, role.unit_ids, role.doctors) == ("ревматолог", (109, 323), 3)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("запишите к сердечно-сосудистому хирургу", ["сердечно-сосудистый хирург, флеболог"]),
        ("акции пластическая хирургия", ["пластический хирург"]),
        ("сколько стоит приём хирурга", ["хирург"]),
        ("к урологу-андрологу", ["уролог-андролог"]),
        ("к урологу", ["уролог"]),
        ("нужен флеболог", ["сердечно-сосудистый хирург, флеболог"]),
    ],
)
def test_qualified_role_is_not_replaced_by_general_one(text, expected):
    catalog = build_role_catalog(
        [
            _doctor((50, "Врач-хирург")),
            _doctor((120, "Врач-пластический хирург")),
            _doctor((126, "Врач-сердечно-сосудистый хирург, флеболог")),
            _doctor((17, "Врач уролог")),
            _doctor((18, "Врач уролог-андролог")),
        ],
        in_region=_here,
    )

    assert _names(catalog.find(text).exact) == expected


def test_part_of_compound_role_is_candidate_not_answer():
    """«Похожий, но другой врач» автоматически не подставляется: часть составной роли — кандидат для выбора."""

    catalog = build_role_catalog([_doctor((136, "Врач гинеколог-маммолог")), _doctor((120, "Врач-пластический хирург"))], in_region=_here)

    to_mammolog = catalog.find("можно к маммологу на завтра?")
    assert _names(to_mammolog.exact) == []
    assert _names(to_mammolog.partial) == ["гинеколог-маммолог"]
    # Прилагательное без существительного роли — не кандидат.
    assert catalog.find("пластическая операция носа").partial == ()


def test_service_role_is_not_candidate_for_every_appointment_phrase():
    """Служебные роли МИС («<фамилия> -прием (центр)», медсестра) — не роль приёма и не кандидат на «приём»."""

    catalog = build_role_catalog(
        [_doctor((326, "Тестов -прием (центр)")), _doctor((380, "Медицинская сестра по физиотерапии"))], in_region=_here
    )

    assert [role.reception for role in catalog.roles] == [False, False]
    for text in ("Приём психолога", "Прием врача и медицинской сестры на дому", "приём рефлексотерапевта"):
        assert catalog.find(text).partial == ()


def test_short_role_word_matches_whole_word_only():
    catalog = build_role_catalog([_doctor((325, "УЗИ (центр)"))], in_region=_here)

    assert _names(catalog.find("узи почек").exact) == ["узи"]
    assert catalog.find("узине").exact == ()


def test_catalog_does_not_depend_on_snapshot_order_and_empty_snapshot_is_empty():
    doctors = [_doctor((1, "Врач-кардиолог")), _doctor((22, "Врач невролог")), _doctor((22, "Врач невролог"))]

    assert build_role_catalog(doctors, in_region=_here) == build_role_catalog(doctors[::-1], in_region=_here)
    assert build_role_catalog([], in_region=_here).roles == ()
    assert build_role_catalog([None, {"unit_links": None}], in_region=lambda d: True).roles == ()


def _inflect(atom: str, noun_end: str, adj_end: str) -> str:
    def one(word: str) -> str:
        if len(word) > 5 and word.endswith(("ый", "ий", "ой")):
            return word[:-2] + adj_end
        return word + noun_end

    return " ".join("-".join(one(part) for part in word.split("-")) for word in atom.split())


def test_every_reception_role_of_snapshot_is_found_by_its_own_name():
    """Встречный свип по данным: каждую роль приёма из среза точный слой узнаёт по её названию в трёх
    формах — «есть ли у вас X», «запишите к X-у», «приём X-а» — и не подменяет общей ролью."""

    doctors = [json.loads(line) for line in _EXCERPT.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog = build_role_catalog(doctors, in_region=lambda d: True)
    reception = [role for role in catalog.roles if role.reception]
    assert len(reception) >= 10

    missed = []
    for role in reception:
        for atom in role.atoms:
            for text in (f"есть ли у вас {atom}", f"запишите к {_inflect(atom, 'у', 'ому')}", f"приём {_inflect(atom, 'а', 'ого')}"):
                if role not in catalog.find(text).exact:
                    missed.append((role.name, text))
    assert missed == []
