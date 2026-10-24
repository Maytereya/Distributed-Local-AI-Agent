"""Характеристики показа врачей по услуге: что сейчас выводят функции, которые ещё читают списки (консилиум 25.10, п. 2-5).

Показ пути off (`doctors.doctors_info`) берёт описание врача по буквам услуги: `_pick_display_specialization` (через
`_doctor_matches_service`: `_SERVICE_FILTER_STOPWORDS`, `_PROCEDURE_GENERIC_TOKENS`) и `_compact_specialization_service_lines`.
Списки пока остаются (решение ведущей 25.10: показ снимать отдельно); эти тесты фиксируют вывод **как есть**, в том числе
дефекты класса «буквы» (помечены), — чтобы снятие списков или переход показа на данные было видно поимённо, а не молча.
Рядом — показ пути `on` (`_doctors_by_parse._display_specialization`): блок роли, под которой врач делает услугу по данным МИС.
Описания — выдуманные.
"""

from __future__ import annotations

import pytest

from messengers_router.services._doctors_by_parse import _display_specialization
from messengers_router.services._doctors_helpers import (
    _compact_specialization_service_lines,
    _doctor_matches_service,
    _pick_display_specialization,
)

_UZI = (
    "• УЗИ брюшной полости:\n- печени\n- желчного пузыря\n• УЗИ мочеполовой системы:\n- почек\n"
    "- предстательной железы и органов мошонки (с ЦДК сосудов)\n• Гинекологическое УЗИ:\n- органов малого таза\n- молочных желез"
)
_HEAD = "• УЗИ брюшной полости:\n- печени\n- желчного пузыря\n• УЗИ мочеполовой системы:"
_SCROTUM = "- предстательной железы и органов мошонки (с ЦДК сосудов)"

_SURGEON, _ONCOLOGIST = "• Амбулаторная хирургия\n• Удаление новообразований кожи", "• Биопсия новообразований\n• Удаление папиллом радиоволновым методом"
_DOC = {
    "specialization": "Общая хирургия",
    "main_specializations": ["Общая хирургия"],
    "unit_links": [
        {"company_unit_id": 40, "company_unit_name": "Врач-хирург", "main": True, "specialization": _SURGEON},
        {"company_unit_id": 92, "company_unit_name": "Врач-онколог", "main": False, "specialization": _ONCOLOGIST},
    ],
}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("УЗИ органов мошонки", _SCROTUM),  # кейс 08.07: строка про ЭТУ процедуру
        ("узи почек", "- почек"),
        ("узи органов", _HEAD),  # только общие стволы — голова описания
        ("процедура узи", _HEAD),
        ("исследование органов малого таза", "- органов малого таза"),
        # Дефекты класса «буквы» (как есть): «железы» щитовидной находит мошонку, «желез» молочных — тоже её.
        ("кто делает узи молочных желез", f"{_SCROTUM}\n- молочных желез"),
        ("узи щитовидной железы", _SCROTUM),
    ],
)
def test_service_lines_of_a_description_as_is(query: str, expected: str) -> None:
    assert _compact_specialization_service_lines(_UZI, query) == expected


@pytest.mark.parametrize(
    ("specialty", "service", "expected"),
    [
        ("", "", "Общая хирургия"),
        ("", "удаление папиллом", _ONCOLOGIST),  # блок по словам услуги, хотя он не основной
        ("", "биопсия", _ONCOLOGIST),
        ("онколог", "", _ONCOLOGIST),
        ("хирург", "удаление папиллом", _SURGEON),  # специальность побеждает слова услуги
        ("", "процедура", "Общая хирургия"),  # слово списка — не услуга
        ("", "узи", "Общая хирургия"),  # общий ствол без различающего токена
    ],
)
def test_display_block_by_letters_as_is(specialty: str, service: str, expected: str) -> None:
    assert _pick_display_specialization(_DOC, preferred_specialty=specialty, preferred_service=service) == expected


@pytest.mark.parametrize(
    ("service", "expected"),
    [("удаление папиллом", True), ("биопсия", True), ("амбулаторная хирургия", True), ("кожи", True), ("процедура узи", False), ("узи органов", False)],
)
def test_doctor_matches_service_by_letters_as_is(service: str, expected: bool) -> None:
    assert _doctor_matches_service(_DOC, service) is expected


@pytest.mark.parametrize(
    ("units", "specialty", "expected"),
    [
        (frozenset({92}), "", _ONCOLOGIST),  # роль, под которой врач делает услугу по данным МИС
        (frozenset(), "онколог", _ONCOLOGIST),
        (frozenset(), "", "Общая хирургия"),
        (frozenset({999}), "хирург", _SURGEON),  # роли услуги у врача нет — блок названной специальности
    ],
)
def test_display_block_by_data_on_path(units: frozenset[int], specialty: str, expected: str) -> None:
    assert _display_specialization(_DOC, unit_ids=units, specialty=specialty) == expected
