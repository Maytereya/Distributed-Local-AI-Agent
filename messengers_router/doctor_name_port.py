"""Локальный порт для doctor-name matching функций.

Изолирует прямой импорт `agent_logic_2.doctor_name_matching` от core-слоя.
"""

from __future__ import annotations

from typing import Any

from agent_logic_2.doctor_name_matching import (
    extract_doctor_name_candidate as _extract_doctor_name_candidate,
    resolve_schedule_surname as _resolve_schedule_surname,
    surname_variants as _surname_variants,
)


def extract_doctor_name_candidate(text: str, *, prefer_schedule: bool = False) -> str | None:
    return _extract_doctor_name_candidate(text, prefer_schedule=prefer_schedule)


def resolve_schedule_surname(value: str | None, doctors: list[dict[str, Any]]) -> str | None:
    return _resolve_schedule_surname(value, doctors)


def surname_variants(value: str) -> list[str]:
    return _surname_variants(value)

